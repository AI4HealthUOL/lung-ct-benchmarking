"""Two tasks (``--task``):
  cv        Nested cross-validation on lung1 (NSCLC-Radiomics).  This is the
            model-selection benchmark; it also trains the per-fold covariance
            bottlenecks consumed by the external task.
  external  Train on lung1, test on lung2 (NSCLC-Radiogenomics).  Uses the best
            fold/config per task as recorded in runs.csv (an MLflow export of
            the cv task) and the bottlenecks trained during cv.

Four feature backends (mutually exclusive; ``--all`` is external-only):
  --radiomics   Pre-extracted PyRadiomics tabular features.
  --curia       CURIA foundation-model patch tokens.
  --curia2      CURIA-2 foundation-model patch tokens.
  --dinov3      DINOv3 foundation-model patch tokens.

Common flags:
  --skip_extraction  Reuse cached token .pt files (skip CT encoding).
  --skip_tabpfn      Do not run TabPFN.
  --hf_token TOKEN   HuggingFace token (else $HF_TOKEN).
  --device DEV       torch device (default: cuda if available else cpu).
  --dinov3_model ID  timm model id for DINOv3.

All paths and credentials come from config.py / environment variables.
"""

import os
import sys
import gc
import json
import glob
import random
import argparse
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd

import code.config as config

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# cli

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--task", choices=["cv", "external"], default="cv")

backend = parser.add_mutually_exclusive_group(required=True)
backend.add_argument("--radiomics", action="store_true")
backend.add_argument("--curia", action="store_true")
backend.add_argument("--curia2", action="store_true")
backend.add_argument("--dinov3", action="store_true")
backend.add_argument(
    "--all",
    action="store_true",
    help="external task only: run radiomics + all FM backends",
)

parser.add_argument("--skip_extraction", action="store_true")
parser.add_argument("--skip_tabpfn", action="store_true")
parser.add_argument("--hf_token", type=str, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--dinov3_model", type=str, default=None)
args = parser.parse_args()

TASK = args.task
SKIP_TABPFN = args.skip_tabpfn

if args.all and TASK != "external":
    sys.exit("--all is only valid with --task external")

if args.all:
    BACKENDS = ["radiomics", "curia", "curia2", "dinov3"]
else:
    BACKENDS = [
        b for b in ("radiomics", "curia", "curia2", "dinov3") if getattr(args, b)
    ]
FM_BACKENDS = [b for b in BACKENDS if b != "radiomics"]
NEEDS_TORCH = len(FM_BACKENDS) > 0

GLOBAL_SEED = config.GLOBAL_SEED
GBDT_MAX_ROUNDS = config.GBDT_MAX_ROUNDS
EARLY_STOPPING_ROUNDS = config.EARLY_STOPPING_ROUNDS
TABPFN_N_ESTIMATORS = config.TABPFN_N_ESTIMATORS
CT_HU_MIN, CT_HU_MAX = config.CT_HU_MIN, config.CT_HU_MAX
COV_DIM_PRIME = config.COV_DIM_PRIME
PATCH_OVERLAP_THRESHOLD = config.PATCH_OVERLAP_THRESHOLD
CONDITIONS = config.CONDITIONS
SLICE_AGG_STRATEGIES = config.SLICE_AGG_STRATEGIES
N_OUTER_FOLDS = config.N_OUTER_FOLDS
N_INNER_FOLDS = config.N_INNER_FOLDS

# optional imports

from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    StratifiedShuffleSplit,
    ShuffleSplit,
)
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost not available.")
try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("Warning: catboost not available.")
try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False
    print("Warning: tabpfn not available.")
try:
    from tabicl import TabICLClassifier, TabICLRegressor

    HAS_TABICL = True
except ImportError:
    HAS_TABICL = False
    print("Warning: tabicl not available.")

import SimpleITK as sitk
from tqdm import tqdm

torch = nn = None
Image = None
DEVICE = None
if NEEDS_TORCH:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        sys.exit("PyTorch not found.  pip install torch")
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow not found.  pip install Pillow")
    HF_TOKEN = args.hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
    _dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device(_dev)
    print("Device: %s" % DEVICE)
else:
    HF_TOKEN = None

# reproducibility


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if NEEDS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(GLOBAL_SEED)
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

# geometry

PATCH_DIM = N_PATCHES = GRID_SIZE = PATCH_SIZE = INPUT_SIZE = None
DINOV3_N_PREFIX = 1  # CLS (+ registers) for DINOv3

# results directories

RESULTS_BASE = os.path.join(config.ROOT, "results")


def results_dir_for(tag):
    d = os.path.join(RESULTS_BASE, TASK, tag)
    os.makedirs(os.path.join(d, "predictions"), exist_ok=True)
    return d


# mlflow

MLFLOW = config.setup_mlflow(
    os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", "lung_cv" if TASK == "cv" else "lung_external"
    )
)


def log_mlflow(run_name, params, metrics, primary_fold_vals=None):
    if MLFLOW is None:
        return
    try:
        with MLFLOW.start_run(run_name=run_name):
            MLFLOW.log_params(params)
            for k, v in metrics.items():
                if not (isinstance(v, float) and np.isnan(v)):
                    MLFLOW.log_metric(k, v)
            for fi, v in enumerate(primary_fold_vals or []):
                if not np.isnan(v):
                    MLFLOW.log_metric("fold_%d_primary" % fi, v)
    except Exception as e:
        print("      MLflow warning: %s" % e)


# shared: scaling


def scale(X_train, *rest):
    scaler = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(len(X_train), 1000),
        random_state=GLOBAL_SEED,
    )
    out = [
        np.nan_to_num(scaler.fit_transform(X_train), nan=0.0, posinf=0.0, neginf=0.0)
    ]
    for arr in rest:
        out.append(
            np.nan_to_num(scaler.transform(arr), nan=0.0, posinf=0.0, neginf=0.0)
        )
    return out


# shared: metrics


def clf_metrics(y_true, y_pred, y_proba, n_classes):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        if n_classes == 2:
            proba = y_proba[:, 1] if y_proba is not None else None
            out["auc"] = (
                float(roc_auc_score(y_true, proba)) if proba is not None else np.nan
            )
        elif y_proba is not None:
            # If a class is absent from the test split, OvR AUC is undefined;
            # one dummy row per missing class (prob 0) so the metric is computable.
            missing = sorted(set(range(n_classes)) - set(np.unique(y_true).tolist()))
            if missing:
                yt, yp = list(y_true), list(y_proba)
                for c in missing:
                    dummy = np.full(n_classes, 1.0 / (n_classes - 1))
                    dummy[c] = 0.0
                    yt.append(c)
                    yp.append(dummy)
                yt, yp = np.array(yt), np.vstack(yp)
            else:
                yt, yp = y_true, y_proba
            out["auc"] = float(
                roc_auc_score(yt, yp, multi_class="ovr", average="macro")
            )
        else:
            out["auc"] = np.nan
    except Exception:
        out["auc"] = np.nan
    try:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        if n_classes == 2:
            tn, fp, fn, tp = cm.ravel()
            out["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            out["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        else:
            out["sensitivity"] = float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            )
            specs = []
            for i in range(n_classes):
                tn = cm.sum() - (cm[i].sum() + cm[:, i].sum() - cm[i, i])
                fp = cm[:, i].sum() - cm[i, i]
                specs.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
            out["specificity"] = float(np.nanmean(specs))
    except Exception:
        out["sensitivity"] = out["specificity"] = np.nan
    return out


def reg_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def aggregate(fold_metrics):
    agg = {}
    for k, vals in fold_metrics.items():
        arr = np.array([v for v in vals if not np.isnan(v)])
        agg["%s_mean" % k] = float(arr.mean()) if len(arr) else np.nan
        agg["%s_std" % k] = float(arr.std()) if len(arr) else np.nan
    return agg


# shared: model factories


def build_clf(name, seed, n_est=None):
    if name == "LogisticRegression":
        return LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=10_000,
            class_weight="balanced",
            n_jobs=-1,
        )
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300,
            max_leaf_nodes=15_000,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "XGBoost" and HAS_XGB:
        return XGBClassifier(
            n_estimators=n_est or GBDT_MAX_ROUNDS,
            learning_rate=0.1,
            booster="gbtree",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    if name == "CatBoost" and HAS_CB:
        return CatBoostClassifier(
            iterations=n_est or GBDT_MAX_ROUNDS,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=seed,
            allow_writing_files=False,
        )
    if name == "TabPFN" and HAS_TABPFN:
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_6, n_estimators=TABPFN_N_ESTIMATORS, random_state=seed
        )
    if name == "TabICL" and HAS_TABICL:
        return TabICLClassifier()
    raise ValueError("Unknown or unavailable clf: %s" % name)


def build_reg(name, seed, n_est=None):
    if name == "Ridge":
        return Ridge(alpha=1.0, fit_intercept=True, solver="auto")
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=300, max_leaf_nodes=15_000, random_state=seed, n_jobs=-1
        )
    if name == "XGBoost" and HAS_XGB:
        return XGBRegressor(
            n_estimators=n_est or GBDT_MAX_ROUNDS,
            learning_rate=0.1,
            objective="reg:squarederror",
            booster="gbtree",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    if name == "CatBoost" and HAS_CB:
        return CatBoostRegressor(
            iterations=n_est or GBDT_MAX_ROUNDS,
            learning_rate=0.05,
            verbose=0,
            random_seed=seed,
            allow_writing_files=False,
        )
    if name == "TabPFN" and HAS_TABPFN:
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_6, n_estimators=TABPFN_N_ESTIMATORS
        )
    if name == "TabICL" and HAS_TABICL:
        return TabICLRegressor()
    raise ValueError("Unknown or unavailable reg: %s" % name)


def clf_model_names():
    names = []
    names += ["LogisticRegression", "RandomForest"]
    if HAS_TABPFN and not SKIP_TABPFN:
        names.append("TabPFN")
    if HAS_XGB:
        names.append("XGBoost")
    if HAS_CB:
        names.append("CatBoost")
    if HAS_TABICL:
        names.append("TabICL")
    return names


def reg_model_names():
    names = []
    names += ["Ridge", "RandomForest"]
    if HAS_TABPFN and not SKIP_TABPFN:
        names.append("TabPFN")
    if HAS_XGB:
        names.append("XGBoost")
    if HAS_CB:
        names.append("CatBoost")
    if HAS_TABICL:
        names.append("TabICL")
    return names


# shared: early stopping


def es_clf(name, X_tr, y_tr, X_val, y_val, seed):
    try:
        n_classes = len(np.unique(y_tr))
        if name == "XGBoost" and HAS_XGB:
            objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
            params = dict(
                n_estimators=GBDT_MAX_ROUNDS,
                learning_rate=0.1,
                objective=objective,
                booster="gbtree",
                tree_method="hist",
                eval_metric="auc" if n_classes == 2 else "mlogloss",
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
            )
            if n_classes > 2:
                params["num_class"] = n_classes
            if n_classes == 2:
                neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
                params["scale_pos_weight"] = neg / pos if pos > 0 else 1.0
            m = XGBClassifier(**params)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            return m.best_iteration
        if name == "CatBoost" and HAS_CB:
            m = CatBoostClassifier(
                iterations=GBDT_MAX_ROUNDS,
                learning_rate=0.05,
                auto_class_weights="Balanced",
                verbose=0,
                eval_metric="AUC" if n_classes == 2 else "MultiClass",
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                random_seed=seed,
                allow_writing_files=False,
            )
            m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            return m.best_iteration_
    except Exception as e:
        print("      [es_clf] %s - using default n_est" % e)
    return None


def es_reg(name, X_tr, y_tr, X_val, y_val, seed):
    try:
        if name == "XGBoost" and HAS_XGB:
            m = XGBRegressor(
                n_estimators=GBDT_MAX_ROUNDS,
                learning_rate=0.1,
                objective="reg:squarederror",
                booster="gbtree",
                tree_method="hist",
                eval_metric="rmse",
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
            )
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            return m.best_iteration
        if name == "CatBoost" and HAS_CB:
            m = CatBoostRegressor(
                iterations=GBDT_MAX_ROUNDS,
                learning_rate=0.05,
                eval_metric="RMSE",
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=0,
                random_seed=seed,
                allow_writing_files=False,
            )
            m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            return m.best_iteration_
    except Exception as e:
        print("      [es_reg] %s - using default n_est" % e)
    return None


# shared: prediction i/o


def run_already_completed(results_dir, run_label):
    pattern = os.path.join(
        results_dir, "predictions", "predictions_%s_*.csv" % run_label
    )
    return len(glob.glob(pattern)) > 0


def save_predictions(results_dir, rows, run_label, timestamp):
    if not rows:
        return
    path = os.path.join(
        results_dir, "predictions", "predictions_%s_%s.csv" % (run_label, timestamp)
    )
    pd.DataFrame(rows).to_csv(path, index=False)
    print("    Predictions -> %s" % path)


# shared: cv folds


def load_cv_folds(path=config.FOLD_JSON):
    with open(path) as fh:
        data = json.load(fh)
    folds = sorted(data["folds"], key=lambda f: f["fold"])
    parsed = [
        {"fold": f["fold"], "train": set(f["train"]), "test": set(f["test"])}
        for f in folds
    ]
    print("Loaded %d outer folds from %s" % (len(parsed), path))
    return parsed


# shared: sitk + mask helpers


def load_sitk(path):
    return sitk.ReadImage(path) if (path and os.path.exists(path)) else None


def align_mask(mask_img, ct_img):
    if mask_img.GetSize() != ct_img.GetSize():
        r = sitk.ResampleImageFilter()
        r.SetReferenceImage(ct_img)
        r.SetInterpolator(sitk.sitkNearestNeighbor)
        r.SetDefaultPixelValue(0)
        mask_img = r.Execute(mask_img)
    return mask_img


def _combine_sides(arrays):
    if not arrays:
        return None
    out = np.zeros_like(arrays[0])
    for a in arrays:
        out = np.maximum(out, a)
    return out


def load_lung1_masks(pid, ct_img):
    """(tumor_zyx, lung_zyx) from GTV-1 + left/right lung segs (NSCLC-Radiomics)."""
    base = os.path.join(config.LUNG1_NII_BASE, pid)
    tumor = None
    t_img = load_sitk(os.path.join(base, "seg-GTV-1.nii.gz"))
    if t_img is not None:
        tumor = sitk.GetArrayFromImage(align_mask(t_img, ct_img)).astype(np.uint8)
    arrays = []
    for side in ("Left", "Right"):
        img = load_sitk(os.path.join(base, "seg-Lung-%s.nii.gz" % side))
        if img is not None:
            arrays.append(
                sitk.GetArrayFromImage(align_mask(img, ct_img)).astype(np.uint8)
            )
    return tumor, _combine_sides(arrays)


def load_lung2_masks(pid, ct_img):
    """(tumor_zyx, lung_zyx) for NSCLC-Radiogenomics: tumor from the first
    seg-*.nii.gz; lung from CLIP lung segmentations if available."""
    base = os.path.join(config.LUNG2_NII_BASE, pid)
    tumor = None
    segs = sorted(glob.glob(os.path.join(base, "seg-*.nii.gz")))
    if segs:
        t_img = load_sitk(segs[0])
        if t_img is not None:
            arr = sitk.GetArrayFromImage(align_mask(t_img, ct_img)).astype(np.uint8)
            tumor = (arr > 0).astype(np.uint8)
    arrays = []
    clip_base = os.path.join(config.LUNG2_CLIP_SEG_PATH, pid)
    for fname in ("%s_Left Lung.nii.gz" % pid, "%s_Right Lung.nii.gz" % pid):
        img = load_sitk(os.path.join(clip_base, fname))
        if img is not None:
            arrays.append(
                sitk.GetArrayFromImage(align_mask(img, ct_img)).astype(np.uint8)
            )
    return tumor, _combine_sides(arrays)


def load_masks(dataset, pid, ct_img):
    return (
        load_lung1_masks(pid, ct_img)
        if dataset == "lung1"
        else load_lung2_masks(pid, ct_img)
    )


def mask_to_patch_indices(mask_slice_yx):
    """Patch indices whose mask overlap >= PATCH_OVERLAP_THRESHOLD."""
    if mask_slice_yx.sum() == 0:
        return np.array([], dtype=np.int64)
    pil = Image.fromarray(mask_slice_yx.astype(np.uint8), mode="L")
    pil = pil.resize((INPUT_SIZE, INPUT_SIZE), Image.NEAREST)
    resized = np.array(pil)
    patch_area = PATCH_SIZE * PATCH_SIZE
    sel = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            r0, c0 = row * PATCH_SIZE, col * PATCH_SIZE
            block = resized[r0 : r0 + PATCH_SIZE, c0 : c0 + PATCH_SIZE]
            if block.sum() / patch_area >= PATCH_OVERLAP_THRESHOLD:
                sel.append(row * GRID_SIZE + col)
    return np.array(sel, dtype=np.int64)


# foundation models: loading + encoding


def _set_geometry(patch_size, grid_size, patch_dim, n_prefix=1):
    global PATCH_DIM, N_PATCHES, GRID_SIZE, PATCH_SIZE, INPUT_SIZE, DINOV3_N_PREFIX
    PATCH_SIZE = int(patch_size)
    GRID_SIZE = int(grid_size)
    N_PATCHES = GRID_SIZE**2
    INPUT_SIZE = GRID_SIZE * PATCH_SIZE
    PATCH_DIM = int(patch_dim)
    DINOV3_N_PREFIX = int(n_prefix)
    print(
        "  Geometry: patch_size=%d grid=%dx%d n_patches=%d dim=%d n_prefix=%d"
        % (PATCH_SIZE, GRID_SIZE, GRID_SIZE, N_PATCHES, PATCH_DIM, DINOV3_N_PREFIX)
    )


def load_model(approach):
    if approach in ("curia", "curia2"):
        from transformers import AutoModel, AutoImageProcessor

        hf_id = config.CURIA_HF_ID if approach == "curia" else config.CURIA2_HF_ID
        print("\nLoading %s from '%s' ..." % (approach, hf_id))
        hf_kw = {"trust_remote_code": True}
        if HF_TOKEN:
            hf_kw["token"] = HF_TOKEN
        processor = AutoImageProcessor.from_pretrained(hf_id, **hf_kw)
        model = AutoModel.from_pretrained(hf_id, **hf_kw).to(DEVICE).eval()
        grid = model.config.image_size // model.config.patch_size
        _set_geometry(model.config.patch_size, grid, model.config.hidden_size, 1)
        return model, processor, "curia"

    try:
        import timm, timm.data
    except ImportError:
        sys.exit("timm not found.  pip install timm")
    timm_id = args.dinov3_model or config.DINOV3_TIMM_ID_DEFAULT
    print("\nLoading DINOv3 via timm: '%s' ..." % timm_id)
    model = timm.create_model(timm_id, pretrained=True, num_classes=0).to(DEVICE).eval()
    ps = model.patch_embed.patch_size
    ps = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
    isz = model.patch_embed.img_size
    isz = isz[0] if isinstance(isz, (tuple, list)) else int(isz)
    n_prefix = int(getattr(model, "num_prefix_tokens", 1))
    _set_geometry(ps, isz // ps, model.num_features, n_prefix)
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    return model, transform, "dinov3"


def infer_geometry_from_tokens(token_dir):
    pts = [f for f in os.listdir(token_dir) if f.endswith(".pt")]
    if not pts:
        sys.exit("No .pt files in %s. Run without --skip_extraction." % token_dir)
    data = torch.load(
        os.path.join(token_dir, pts[0]), map_location="cpu", weights_only=False
    )
    tokens = data["tokens"]
    grid = int(round((tokens.shape[1] - 1) ** 0.5))
    _set_geometry(int(data.get("patch_size", 14)), grid, tokens.shape[2], 1)


def _encode_slice(model, processor, kind, ct_slice_yx):
    with torch.no_grad():
        if kind == "curia":
            proc = processor(images=ct_slice_yx.astype(np.float32), return_tensors="pt")
            proc = {
                k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in proc.items()
            }
            hidden = model(**proc).last_hidden_state[0]
            return hidden.cpu().float().numpy()
        arr = np.clip(ct_slice_yx, CT_HU_MIN, CT_HU_MAX)
        arr = (arr - CT_HU_MIN) / (CT_HU_MAX - CT_HU_MIN) * 255.0
        pil = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
        x = processor(pil).unsqueeze(0).to(DEVICE)
        feat = model.forward_features(x)
        tokens = torch.cat([feat[0, 0:1], feat[0, DINOV3_N_PREFIX:]], dim=0)
        return tokens.cpu().float().numpy()


def extract_tokens(approach, pids, nii_base, token_dir):
    """Encode every CT slice of every patient and cache <pid>.pt token tensors."""
    os.makedirs(token_dir, exist_ok=True)
    missing = [
        p for p in pids if not os.path.exists(os.path.join(token_dir, "%s.pt" % p))
    ]
    if not missing:
        print("  %s: all %d token files present." % (approach, len(pids)))
        return
    model, processor, kind = load_model(approach)
    print("  %s: encoding %d patients -> %s" % (approach, len(missing), token_dir))
    for pid in tqdm(missing, desc="Encode %s" % approach):
        ct = load_sitk(os.path.join(nii_base, pid, "image.nii.gz"))
        if ct is None:
            continue
        ct_zyx = sitk.GetArrayFromImage(ct)
        toks = []
        for z in range(ct_zyx.shape[0]):
            try:
                toks.append(_encode_slice(model, processor, kind, ct_zyx[z]))
            except Exception as e:
                print("  [%s] z=%d error: %s" % (pid, z, e))
                toks.append(np.zeros((1 + N_PATCHES, PATCH_DIM), dtype=np.float32))
        torch.save(
            {"tokens": np.stack(toks).astype(np.float32), "patch_size": PATCH_SIZE},
            os.path.join(token_dir, "%s.pt" % pid),
        )
    del model
    if DEVICE is not None and DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def get_tokens(token_dir, pid):
    path = os.path.join(token_dir, "%s.pt" % pid)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)["tokens"]


# foundation models: covariance pooling

COV_RECON_EPOCHS = 100
COV_RECON_LR = 1e-3
COV_RECON_SLICE_CAP = 5000
COV_RECON_BATCH = 64


def _cov_pool_slice(patches, L, R):
    """patches [N, D]; L,R [D, D'] -> flattened second moment [D'^2]."""
    M = (patches @ L).T @ (patches @ R) / max(len(patches), 1)
    return M.ravel().astype(np.float32)


def _iter_training_slices(token_dir, pids, cap, rng, condition, pid_to_mask):
    collected = []
    for pid in pids:
        raw = get_tokens(token_dir, pid)
        if raw is None:
            continue
        if condition in ("tumor", "lung"):
            mv = pid_to_mask.get(pid)
            if mv is None:
                del raw
                continue
            for z in range(min(raw.shape[0], mv.shape[0])):
                if mv[z].sum() == 0:
                    continue
                idx = mask_to_patch_indices(mv[z])
                patches = raw[z, 1:, :]
                collected.append(
                    (patches[idx] if len(idx) else patches).astype(np.float32)
                )
        else:
            for z in range(raw.shape[0]):
                collected.append(raw[z, 1:, :].astype(np.float32))
        del raw
        gc.collect()
    if len(collected) > cap:
        keep = rng.choice(len(collected), size=int(cap), replace=False)
        collected = [collected[i] for i in keep]
    return collected


def get_or_train_cov_bottleneck(
    token_dir, btl_dir, train_pids, fold_id, condition, pid_to_mask
):
    """Load or learn untied L, R for (condition, fold).  Cache:
    <btl_dir>/LR_<condition>_fold<fold>_d<D'>.npz."""
    os.makedirs(btl_dir, exist_ok=True)
    btl_path = os.path.join(
        btl_dir, "LR_%s_fold%d_d%d.npz" % (condition, fold_id, COV_DIM_PRIME)
    )
    if os.path.exists(btl_path):
        data = np.load(btl_path)
        return data["L"], data["R"]

    print(
        "  Training cov bottleneck (untied L,R; d'=%d; %s fold %d) ..."
        % (COV_DIM_PRIME, condition, fold_id)
    )
    rng = np.random.default_rng(GLOBAL_SEED)
    slices = _iter_training_slices(
        token_dir, train_pids, COV_RECON_SLICE_CAP, rng, condition, pid_to_mask
    )
    if not slices:
        raise RuntimeError("No patch tokens for cov bottleneck training.")

    D, Dp = slices[0].shape[1], COV_DIM_PRIME
    dev = DEVICE if NEEDS_TORCH else torch.device("cpu")
    torch.manual_seed(GLOBAL_SEED)
    L = nn.Parameter(torch.randn(D, Dp, device=dev) / (D**0.5))
    R = nn.Parameter(torch.randn(D, Dp, device=dev) / (D**0.5))
    opt = torch.optim.Adam([L, R], lr=COV_RECON_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=COV_RECON_EPOCHS, eta_min=1e-5
    )
    slice_tensors = [torch.from_numpy(s) for s in slices]
    n = len(slice_tensors)

    for epoch in range(COV_RECON_EPOCHS):
        perm = rng.permutation(n)
        for b0 in range(0, n, COV_RECON_BATCH):
            idx = perm[b0 : b0 + COV_RECON_BATCH]
            opt.zero_grad()
            batch_loss = torch.tensor(0.0, device=dev)
            for j in idx:
                X = slice_tensors[j].to(dev)
                T = max(X.shape[0], 1)
                with torch.no_grad():
                    target = (X.mT @ X) / T
                M = (X @ L).mT @ (X @ R) / T
                recons = L @ M @ R.mT
                batch_loss = batch_loss + torch.norm(recons - target, p="fro") ** 2
                del X, M, recons, target
            (batch_loss / max(len(idx), 1)).backward()
            opt.step()
        sched.step()

    L_np = L.detach().cpu().numpy().astype(np.float32)
    R_np = R.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(btl_path, L=L_np, R=R_np)
    print("  Cov bottleneck saved -> %s  (L,R shape %s)" % (btl_path, L_np.shape))
    return L_np, R_np


# foundation models: per-patient feature vectors (cached)


def compute_patient_vector(token_dir, cache_dir, pid, condition, mask_vol):
    """Mean/median/max over slice vectors (mean-pooled patches).  Cached as
    <cache_dir>/<pid>_<condition>.npz."""
    cache_path = os.path.join(cache_dir, "%s_%s.npz" % (pid, condition))
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            return {k: data[k] for k in data.files}
        except Exception:
            pass
    tokens = get_tokens(token_dir, pid)
    if tokens is None:
        return None
    if condition == "noseg":
        slice_vecs = tokens[:, 1:, :].mean(axis=1)
    else:
        if mask_vol is None or mask_vol.sum() == 0:
            return None
        patch_tokens = tokens[:, 1:, :]
        rel = [
            z
            for z in range(min(tokens.shape[0], mask_vol.shape[0]))
            if mask_vol[z].sum() > 0
        ]
        if not rel:
            return None
        vecs = []
        for z in rel:
            idx = mask_to_patch_indices(mask_vol[z])
            vecs.append(
                (patch_tokens[z][idx] if len(idx) else patch_tokens[z]).mean(axis=0)
            )
        slice_vecs = np.stack(vecs)
    del tokens
    gc.collect()
    result = {
        "mean": slice_vecs.mean(axis=0).astype(np.float32),
        "median": np.median(slice_vecs, axis=0).astype(np.float32),
        "max": slice_vecs.max(axis=0).astype(np.float32),
    }
    os.makedirs(cache_dir, exist_ok=True)
    try:
        np.savez_compressed(cache_path, **result)
    except Exception as e:
        print("    Warning: cache write failed %s: %s" % (cache_path, e))
    return result


def compute_patient_cov_vector(
    token_dir, cache_dir, pid, condition, L, R, mask_vol, fold_id
):
    """Covariance pooling per slice, then mean/median/max.  Cached as
    <cache_dir>/<pid>_<condition>_cov_fold<fold>_d<D'>.npz."""
    cache_path = os.path.join(
        cache_dir, "%s_%s_cov_fold%d_d%d.npz" % (pid, condition, fold_id, COV_DIM_PRIME)
    )
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            return {k: data[k] for k in data.files}
        except Exception:
            pass
    tokens = get_tokens(token_dir, pid)
    if tokens is None:
        return None
    patch_tokens = tokens[:, 1:, :]
    n_slices = patch_tokens.shape[0]
    cov_vecs = []
    if condition == "noseg":
        for z in range(n_slices):
            cov_vecs.append(_cov_pool_slice(patch_tokens[z], L, R))
    else:
        if mask_vol is None or mask_vol.sum() == 0:
            return None
        for z in range(min(n_slices, mask_vol.shape[0])):
            if mask_vol[z].sum() == 0:
                continue
            idx = mask_to_patch_indices(mask_vol[z])
            patches = patch_tokens[z]
            cov_vecs.append(
                _cov_pool_slice(patches[idx] if len(idx) else patches, L, R)
            )
    del tokens
    gc.collect()
    if not cov_vecs:
        return None
    cov_arr = np.stack(cov_vecs)
    result = {
        "mean": cov_arr.mean(axis=0).astype(np.float32),
        "median": np.median(cov_arr, axis=0).astype(np.float32),
        "max": cov_arr.max(axis=0).astype(np.float32),
    }
    os.makedirs(cache_dir, exist_ok=True)
    try:
        np.savez_compressed(cache_path, **result)
    except Exception as e:
        print("    Warning: cov cache write failed %s: %s" % (cache_path, e))
    return result


# radiomics: shared loaders

RADIO_META_COLS = {
    "patient_id",
    "survived_2yr",
    "Survival.time",
    "deadstatus.event",
    "mask_type",
    "mode",
    "slice",
    "histology_encoded",
    "tumor_volume_class",
    "tumor_volume_mm3",
    "t_stage_binary",
    "age",
}

CLF_TASKS = {
    "survived_2yr": "survived_2yr",
    "histology": "histology_encoded",
    "tumor_volume_class": "tumor_volume_class",
    "t_stage_binary": "t_stage_binary",
}
REG_TASKS = {"age": "age"}


def radio_feat_cols(df):
    return [
        c
        for c in df.columns
        if c not in RADIO_META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def radio_load(radio_dir, basename):
    path = os.path.join(radio_dir, "%s_labelled.csv" % basename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    return df


def radio_extract_3d(df, task_col, feat_cols, is_reg):
    sub = df[["patient_id", task_col] + feat_cols].dropna(subset=[task_col])
    sub = sub.drop_duplicates("patient_id")
    X = sub[feat_cols].values.astype(np.float32)
    y = sub[task_col].values.astype(np.float32 if is_reg else int)
    return X, y, sub["patient_id"].values


def radio_pool_2d(df, task_col, feat_cols, strategy, is_reg):
    agg = {"mean": np.mean, "median": np.median, "max": np.max}[strategy]
    recs = []
    for pid, grp in df.groupby("patient_id"):
        lbl = grp[task_col].iloc[0]
        if pd.isna(lbl):
            continue
        recs.append((pid, lbl, agg(grp[feat_cols].values.astype(np.float32), axis=0)))
    if not recs:
        return np.empty((0, len(feat_cols)), np.float32), np.array([]), np.array([])
    pids = np.array([r[0] for r in recs])
    y = np.array([r[1] for r in recs], dtype=np.float32 if is_reg else int)
    X = np.stack([r[2] for r in recs])
    return X, y, pids


# ||  CV TASK  (lung1 nested cross-validation)

INNER_VAL_FRACTION = 0.2
PATCH_POOL_METHODS = {c: list(config.PATCH_POOL_METHODS) for c in CONDITIONS}


def _cv_clf(
    X,
    y,
    pids,
    run_label,
    model_name,
    n_classes,
    cv_folds,
    extra_row,
    results_dir,
    pred_collector,
    timestamp,
):
    uses_es = model_name in ("XGBoost", "CatBoost")
    fold_metrics = {
        k: [] for k in ["auc", "accuracy", "f1_macro", "sensitivity", "specificity"]
    }
    pid_arr = np.asarray(pids)

    for fd in cv_folds:
        fold = fd["fold"]
        tr = np.where(np.isin(pid_arr, list(fd["train"])))[0]
        te = np.where(np.isin(pid_arr, list(fd["test"])))[0]
        if len(tr) == 0 or len(te) == 0:
            continue
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            print("      Fold %d skipped (single class)" % fold)
            continue
        X_tr_s, X_te_s = scale(X_tr, X_te)

        best_n = None
        if uses_es:
            inner = StratifiedShuffleSplit(
                n_splits=1, test_size=INNER_VAL_FRACTION, random_state=GLOBAL_SEED + 1
            )
            itr, iva = next(inner.split(X_tr_s, y_tr))
            if len(np.unique(y_tr[itr])) >= 2 and len(np.unique(y_tr[iva])) >= 2:
                best_n = es_clf(
                    model_name,
                    X_tr_s[itr],
                    y_tr[itr],
                    X_tr_s[iva],
                    y_tr[iva],
                    GLOBAL_SEED,
                )
        try:
            m = build_clf(model_name, GLOBAL_SEED, best_n)
            if model_name == "XGBoost" and HAS_XGB and n_classes == 2:
                neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
                m.set_params(scale_pos_weight=neg / pos if pos > 0 else 1.0)
            m.fit(X_tr_s, y_tr)
            y_pred = m.predict(X_te_s)
            y_proba = m.predict_proba(X_te_s) if hasattr(m, "predict_proba") else None
            metrics = clf_metrics(y_te, y_pred, y_proba, n_classes)
        except Exception as e:
            print("      Fold %d error: %s" % (fold, e))
            metrics, y_pred, y_proba = (
                {k: np.nan for k in fold_metrics},
                np.full(len(y_te), -1),
                None,
            )
        for k, v in metrics.items():
            if k in fold_metrics:
                fold_metrics[k].append(v)
        for i, (pid, tl, pl) in enumerate(zip(pid_arr[te], y_te, y_pred)):
            row = {
                "patient_id": pid,
                "fold": fold,
                "model": model_name,
                "true_label": tl,
                "pred_label": pl,
                **extra_row,
            }
            if y_proba is not None:
                for c in range(y_proba.shape[1]):
                    row["prob_class_%d" % c] = y_proba[i, c]
            pred_collector.append(row)
        print(
            "      Fold %d  best_n=%-5s  AUC=%.4f F1=%.4f Acc=%.4f"
            % (
                fold,
                str(best_n) if best_n else "N/A",
                metrics.get("auc", np.nan),
                metrics.get("f1_macro", np.nan),
                metrics.get("accuracy", np.nan),
            )
        )
    return aggregate(fold_metrics), fold_metrics["auc"]


def _cv_reg(
    X,
    y,
    pids,
    run_label,
    model_name,
    cv_folds,
    extra_row,
    results_dir,
    pred_collector,
    timestamp,
):
    uses_es = model_name in ("XGBoost", "CatBoost")
    fold_metrics = {k: [] for k in ["mae", "rmse", "r2"]}
    pid_arr = np.asarray(pids)

    for fd in cv_folds:
        fold = fd["fold"]
        tr = np.where(np.isin(pid_arr, list(fd["train"])))[0]
        te = np.where(np.isin(pid_arr, list(fd["test"])))[0]
        if len(tr) == 0 or len(te) == 0:
            continue
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        X_tr_s, X_te_s = scale(X_tr, X_te)
        best_n = None
        if uses_es:
            inner = ShuffleSplit(
                n_splits=1, test_size=INNER_VAL_FRACTION, random_state=GLOBAL_SEED + 1
            )
            itr, iva = next(inner.split(X_tr_s))
            best_n = es_reg(
                model_name, X_tr_s[itr], y_tr[itr], X_tr_s[iva], y_tr[iva], GLOBAL_SEED
            )
        try:
            m = build_reg(model_name, GLOBAL_SEED, best_n)
            m.fit(X_tr_s, y_tr)
            y_pred = m.predict(X_te_s)
            metrics = reg_metrics(y_te, y_pred)
        except Exception as e:
            print("      Fold %d error: %s" % (fold, e))
            metrics, y_pred = {k: np.nan for k in fold_metrics}, np.full(
                len(y_te), np.nan
            )
        for k, v in metrics.items():
            fold_metrics[k].append(v)
        for pid, tv, pv in zip(pid_arr[te], y_te, y_pred):
            pred_collector.append(
                {
                    "patient_id": pid,
                    "fold": fold,
                    "model": model_name,
                    "true_label": float(tv),
                    "pred_label": float(pv),
                    **extra_row,
                }
            )
        print(
            "      Fold %d  best_n=%-5s  MAE=%.1f RMSE=%.1f R2=%.4f"
            % (
                fold,
                str(best_n) if best_n else "N/A",
                metrics.get("mae", np.nan),
                metrics.get("rmse", np.nan),
                metrics.get("r2", np.nan),
            )
        )
    return aggregate(fold_metrics), fold_metrics["r2"]


# runs.csv

RUNS_COLUMNS = [
    "Name",
    "dataset",
    "task",
    "model",
    "condition",
    "pooling",
    "patch_pooling",
] + ["fold_%d_primary" % i for i in range(N_OUTER_FOLDS)]


def _runs_row(rec, primary):
    """rec holds the load_runs columns; primary is the per-fold metric list
    (the same values logged to MLflow as fold_N_primary)."""
    row = dict(rec)
    for i in range(N_OUTER_FOLDS):
        row["fold_%d_primary" % i] = primary[i] if i < len(primary) else ""
    return row


def _mlflow_params(rec, **extra):
    """MLflow params = the runs.csv columns (minus Name, which is the run_name)
    plus extra diagnostics. Keeps the export and the direct file aligned."""
    p = {k: v for k, v in rec.items() if k != "Name"}
    p.update(extra)
    return p


def _append_runs_csv(rows):
    """Merge rows into config.RUNS_CSV (dedup by Name) so successive cv runs of
    different backends accumulate into one runs.csv."""
    if not rows:
        return
    path = config.RUNS_CSV
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    df_new = pd.DataFrame(rows, columns=RUNS_COLUMNS)
    if os.path.exists(path):
        try:
            old = pd.read_csv(path)
            if "Name" in old.columns:
                old = old[~old["Name"].isin(df_new["Name"])]
            df_new = pd.concat([old, df_new], ignore_index=True, sort=False)
        except Exception as e:
            print("  (could not merge existing runs.csv: %s)" % e)
    df_new.to_csv(path, index=False)
    print("runs.csv -> %s  (%d rows)" % (path, len(df_new)))


def cv_radiomics():
    tag = "radiomics"
    results_dir = results_dir_for(tag)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_folds = load_cv_folds()
    print("=" * 80 + "\nCV RADIOMICS BENCHMARK\n" + "=" * 80)

    all_rows, all_preds, runs_rows = [], [], []

    def process(df, modality, dataset, pooling="NA"):
        feat_cols = radio_feat_cols(df)
        for task_name, task_col in CLF_TASKS.items():
            if task_col not in df.columns or df[task_col].isna().all():
                continue
            if modality == "3D":
                X, y, pids = radio_extract_3d(df, task_col, feat_cols, False)
            else:
                X, y, pids = radio_pool_2d(df, task_col, feat_cols, pooling, False)
            if len(y) == 0:
                continue
            n_classes = len(np.unique(y))
            print(
                "\n  CLF %-22s pooling=%-6s n=%d classes=%d"
                % (task_name, pooling, len(y), n_classes)
            )
            for model_name in clf_model_names():
                mtag = "radiomics3d" if modality == "3D" else "radiomics2d"
                slice_tok = "na" if modality == "3D" else pooling
                # bootstrap layout: {model}_{seg_mask}_{patch_pooling}_{slice_pooling}_{task}_{classifier}
                run_label = "%s_tumor_na_%s_%s_%s" % (
                    mtag,
                    slice_tok,
                    task_name,
                    model_name,
                )
                if run_already_completed(results_dir, run_label):
                    print("  [SKIP] %s" % run_label)
                    continue
                preds = []
                extra = {"task": task_name, "dataset": dataset, "pooling": pooling}
                agg, primary = _cv_clf(
                    X,
                    y,
                    pids,
                    run_label,
                    model_name,
                    n_classes,
                    cv_folds,
                    extra,
                    results_dir,
                    preds,
                    timestamp,
                )
                all_preds.extend(preds)
                save_predictions(results_dir, preds, run_label, timestamp)
                rec = {
                    "Name": run_label,
                    "dataset": (
                        "radiomics_tumor_3D"
                        if modality == "3D"
                        else "radiomics_tumor_2D"
                    ),
                    "task": task_name,
                    "model": model_name,
                    "condition": "",
                    "pooling": "" if modality == "3D" else pooling,
                    "patch_pooling": "",
                }
                log_mlflow(
                    run_label,
                    _mlflow_params(
                        rec,
                        n_samples=len(y),
                        n_features=X.shape[1],
                        n_classes=n_classes,
                    ),
                    agg,
                    primary,
                )
                runs_rows.append(_runs_row(rec, primary))
                all_rows.append(
                    {
                        "modality": modality,
                        "task_type": "classification",
                        "task": task_name,
                        "pooling": pooling,
                        "model": model_name,
                        **agg,
                    }
                )

        for task_name, task_col in REG_TASKS.items():
            if task_col not in df.columns or df[task_col].isna().all():
                continue
            if modality == "3D":
                X, y, pids = radio_extract_3d(df, task_col, feat_cols, True)
            else:
                X, y, pids = radio_pool_2d(df, task_col, feat_cols, pooling, True)
            if len(y) == 0:
                continue
            print("\n  REG %-22s pooling=%-6s n=%d" % (task_name, pooling, len(y)))
            for model_name in reg_model_names():
                mtag = "radiomics3d" if modality == "3D" else "radiomics2d"
                slice_tok = "na" if modality == "3D" else pooling
                # bootstrap layout: {model}_{seg_mask}_{patch_pooling}_{slice_pooling}_{task}_{classifier}
                run_label = "%s_tumor_na_%s_%s_%s" % (
                    mtag,
                    slice_tok,
                    task_name,
                    model_name,
                )
                if run_already_completed(results_dir, run_label):
                    print("  [SKIP] %s" % run_label)
                    continue
                preds = []
                extra = {"task": task_name, "dataset": dataset, "pooling": pooling}
                agg, primary = _cv_reg(
                    X,
                    y,
                    pids,
                    run_label,
                    model_name,
                    cv_folds,
                    extra,
                    results_dir,
                    preds,
                    timestamp,
                )
                all_preds.extend(preds)
                save_predictions(results_dir, preds, run_label, timestamp)
                rec = {
                    "Name": run_label,
                    "dataset": (
                        "radiomics_tumor_3D"
                        if modality == "3D"
                        else "radiomics_tumor_2D"
                    ),
                    "task": task_name,
                    "model": model_name,
                    "condition": "",
                    "pooling": "" if modality == "3D" else pooling,
                    "patch_pooling": "",
                }
                log_mlflow(
                    run_label,
                    _mlflow_params(rec, n_samples=len(y), n_features=X.shape[1]),
                    agg,
                    primary,
                )
                runs_rows.append(_runs_row(rec, primary))
                all_rows.append(
                    {
                        "modality": modality,
                        "task_type": "regression",
                        "task": task_name,
                        "pooling": pooling,
                        "model": model_name,
                        **agg,
                    }
                )

    # 3D
    try:
        df3d = radio_load(config.LUNG1_RADIO_DIR, "radiomics_tumor_3D")
        process(df3d, "3D", "radiomics_tumor_3D")
    except FileNotFoundError as e:
        print("  SKIP 3D: %s" % e)
    # 2D
    try:
        df2d = radio_load(config.LUNG1_RADIO_DIR, "radiomics_tumor_2D")
        for strategy in SLICE_AGG_STRATEGIES:
            print("\n  [pooling=%s]" % strategy)
            process(df2d, "2D", "radiomics_tumor_2D", pooling=strategy)
    except FileNotFoundError as e:
        print("  SKIP 2D: %s" % e)

    _save_summary(all_rows, all_preds, results_dir, timestamp)
    _append_runs_csv(runs_rows)


def cv_foundation(approach):
    tag = approach
    results_dir = results_dir_for(tag)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = config.LUNG1_OUTPUT_DIRS[approach]
    token_dir = os.path.join(out, "patch_tokens")
    cache_dir = os.path.join(out, "vector_cache")
    btl_dir = os.path.join(out, "cov_bottleneck")
    for d in (token_dir, cache_dir, btl_dir):
        os.makedirs(d, exist_ok=True)

    print(
        "=" * 80 + "\nCV %s FOUNDATION-MODEL BENCHMARK\n" % approach.upper() + "=" * 80
    )

    pids_all = sorted(
        d
        for d in os.listdir(config.LUNG1_NII_BASE)
        if os.path.isdir(os.path.join(config.LUNG1_NII_BASE, d))
    )
    if not args.skip_extraction:
        extract_tokens(approach, pids_all, config.LUNG1_NII_BASE, token_dir)
    infer_geometry_from_tokens(token_dir)

    label_df = radio_load(config.LUNG1_RADIO_DIR, "radiomics_tumor_3D")
    label_df = label_df.drop_duplicates("patient_id").set_index("patient_id")

    cv_folds = load_cv_folds()
    fold_splits = [(fd["train"], fd["test"]) for fd in cv_folds]

    all_rows, all_preds, runs_rows = [], [], []
    pid_set = set(pids_all)

    for condition in CONDITIONS:
        print("\n" + "=" * 70 + "\nCONDITION: %s\n" % condition.upper() + "=" * 70)
        available = sorted(
            f[:-3]
            for f in os.listdir(token_dir)
            if f.endswith(".pt") and f[:-3] in pid_set
        )

        pid_to_mask, pid_cohort = {}, set()
        if condition in ("tumor", "lung"):
            print("  Loading %s masks ..." % condition)
            for pid in tqdm(available, desc="Masks [%s]" % condition):
                ct = load_sitk(os.path.join(config.LUNG1_NII_BASE, pid, "image.nii.gz"))
                if ct is None:
                    continue
                tumor, lung = load_lung1_masks(pid, ct)
                m = tumor if condition == "tumor" else lung
                if m is not None and m.sum() > 0:
                    pid_cohort.add(pid)
                    pid_to_mask[pid] = m
                del ct
            gc.collect()
            print("  %d patients with valid %s mask" % (len(pid_cohort), condition))

        for patch_pool in PATCH_POOL_METHODS[condition]:
            if patch_pool == "cov":
                print("\n  --- cov bottleneck d'=%d ---" % COV_DIM_PRIME)
                vecs_by_fold = {}
                for fid, (train_set, test_set) in enumerate(fold_splits):
                    train_pids = [p for p in available if p in train_set]
                    fold_pids = [
                        p for p in available if p in train_set or p in test_set
                    ]
                    L, R = get_or_train_cov_bottleneck(
                        token_dir, btl_dir, train_pids, fid, condition, pid_to_mask
                    )
                    fv = {}
                    for pid in tqdm(
                        fold_pids, desc="Cov vectors fold %d [%s]" % (fid, condition)
                    ):
                        if condition in ("tumor", "lung") and pid not in pid_cohort:
                            continue
                        mv = (
                            pid_to_mask.get(pid)
                            if condition in ("tumor", "lung")
                            else None
                        )
                        res = compute_patient_cov_vector(
                            token_dir, cache_dir, pid, condition, L, R, mv, fid
                        )
                        if res is not None:
                            fv[pid] = res
                    vecs_by_fold[fid] = fv
                vecs_all = {}
                for fid, (train_set, test_set) in enumerate(fold_splits):
                    for pid in vecs_by_fold[fid]:
                        if pid in test_set:
                            vecs_all[pid] = vecs_by_fold[fid][pid]
            else:
                vecs_by_fold = None
                print("\n  Pre-computing vectors [%s|%s] ..." % (condition, patch_pool))
                vecs_all = {}
                for pid in tqdm(
                    available, desc="Vectors [%s|%s]" % (condition, patch_pool)
                ):
                    if condition in ("tumor", "lung") and pid not in pid_cohort:
                        continue
                    mv = (
                        pid_to_mask.get(pid) if condition in ("tumor", "lung") else None
                    )
                    res = compute_patient_vector(
                        token_dir, cache_dir, pid, condition, mv
                    )
                    if res is not None:
                        vecs_all[pid] = res
            print("  Cached vectors for %d patients" % len(vecs_all))

            for task_type, task_dict in [("reg", REG_TASKS), ("clf", CLF_TASKS)]:
                for task_name, task_col in task_dict.items():
                    for slice_agg in SLICE_AGG_STRATEGIES:
                        valid = [
                            p
                            for p in sorted(vecs_all.keys())
                            if slice_agg in vecs_all[p]
                            and p in label_df.index
                            and not pd.isna(label_df.loc[p, task_col])
                        ]
                        if len(valid) < 10:
                            continue
                        pid_arr = np.array(valid)
                        labels = label_df.loc[valid, task_col].values
                        y = (
                            labels.astype(int)
                            if task_type == "clf"
                            else labels.astype(np.float32)
                        )
                        X = np.stack([vecs_all[p][slice_agg] for p in valid])
                        n_classes = len(np.unique(y)) if task_type == "clf" else 0

                        # cov: assemble per-fold (X, pids) in each fold's own L,R frame.
                        fold_data = None
                        if patch_pool == "cov" and vecs_by_fold is not None:
                            vs = set(valid)
                            fold_data = {}
                            for fid, fv in vecs_by_fold.items():
                                pf = [
                                    p
                                    for p in sorted(fv.keys())
                                    if p in vs and slice_agg in fv[p]
                                ]
                                if pf:
                                    fold_data[fid] = (
                                        np.stack([fv[p][slice_agg] for p in pf]),
                                        np.array(pf),
                                    )

                        print(
                            "\n  %s %-22s n=%d patch_pool=%s slice_agg=%s"
                            % (
                                task_type.upper(),
                                task_name,
                                len(valid),
                                patch_pool,
                                slice_agg,
                            )
                        )
                        names = (
                            clf_model_names()
                            if task_type == "clf"
                            else reg_model_names()
                        )
                        for model_name in names:
                            run_label = "%s_%s_%s_%s_%s_%s" % (
                                approach,
                                condition,
                                patch_pool,
                                slice_agg,
                                task_name,
                                model_name,
                            )
                            if run_already_completed(results_dir, run_label):
                                print("  [SKIP] %s" % run_label)
                                continue
                            if patch_pool == "cov" and model_name == "TabPFN":
                                continue  # cov features are high-dim; TabPFN excluded
                            preds = []
                            extra = {
                                "task": task_name,
                                "condition": condition,
                                "pooling": patch_pool,
                            }
                            if task_type == "clf":
                                agg, primary = _cv_fm_clf(
                                    X,
                                    y,
                                    pid_arr,
                                    model_name,
                                    n_classes,
                                    fold_splits,
                                    fold_data,
                                    extra,
                                    preds,
                                )
                            else:
                                agg, primary = _cv_fm_reg(
                                    X,
                                    y,
                                    pid_arr,
                                    model_name,
                                    fold_splits,
                                    fold_data,
                                    extra,
                                    preds,
                                )
                            all_preds.extend(preds)
                            save_predictions(results_dir, preds, run_label, timestamp)
                            rec = {
                                "Name": run_label,
                                "dataset": approach,
                                "task": task_name,
                                "model": model_name,
                                "condition": condition,
                                "pooling": slice_agg,
                                "patch_pooling": patch_pool,
                            }
                            log_mlflow(
                                run_label,
                                _mlflow_params(
                                    rec,
                                    n_samples=len(y),
                                    n_features=X.shape[1],
                                    n_classes=n_classes,
                                ),
                                agg,
                                primary,
                            )
                            runs_rows.append(_runs_row(rec, primary))
                            all_rows.append(
                                {
                                    "condition": condition,
                                    "patch_pool": patch_pool,
                                    "slice_agg": slice_agg,
                                    "task_type": task_type,
                                    "task": task_name,
                                    "model": model_name,
                                    **agg,
                                }
                            )
                        gc.collect()
            del vecs_all
            gc.collect()
        del pid_to_mask, pid_cohort
        gc.collect()

    _save_summary(all_rows, all_preds, results_dir, timestamp)
    _append_runs_csv(runs_rows)


def _cv_fm_clf(X, y, pids, model_name, n_classes, fold_splits, fold_data, extra, preds):
    uses_es = model_name in ("XGBoost", "CatBoost")
    fold_metrics = {
        k: [] for k in ["auc", "accuracy", "f1_macro", "sensitivity", "specificity"]
    }
    pid_to_y = {str(p): yy for p, yy in zip(pids, y)} if fold_data is not None else None

    for fold, (train_set, test_set) in enumerate(fold_splits):
        if fold_data is not None:
            if fold not in fold_data:
                continue
            X_cur, pids_cur = fold_data[fold]
            y_cur = np.array([pid_to_y[str(p)] for p in pids_cur], dtype=y.dtype)
        else:
            X_cur, pids_cur, y_cur = X, pids, y
        tr = np.where(np.isin(pids_cur, list(train_set)))[0]
        te = np.where(np.isin(pids_cur, list(test_set)))[0]
        if len(tr) == 0 or len(te) == 0:
            continue
        X_tr, X_te, y_tr, y_te = X_cur[tr], X_cur[te], y_cur[tr], y_cur[te]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        inner = StratifiedKFold(
            N_INNER_FOLDS, shuffle=True, random_state=GLOBAL_SEED + 1
        )
        itr, iva = list(inner.split(X_tr, y_tr))[0]
        best_n = None
        if uses_es:
            X_itr_s, X_iva_s = scale(X_tr[itr], X_tr[iva])
            best_n = es_clf(
                model_name, X_itr_s, y_tr[itr], X_iva_s, y_tr[iva], GLOBAL_SEED
            )
        X_tr_s, X_te_s = scale(X_tr, X_te)
        try:
            m = build_clf(model_name, GLOBAL_SEED, best_n)
            if model_name == "XGBoost" and HAS_XGB and n_classes == 2:
                neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
                m.set_params(scale_pos_weight=neg / pos if pos > 0 else 1.0)
            m.fit(X_tr_s, y_tr)
            y_pred = m.predict(X_te_s)
            y_proba = m.predict_proba(X_te_s) if hasattr(m, "predict_proba") else None
            metrics = clf_metrics(y_te, y_pred, y_proba, n_classes)
        except Exception as e:
            print("      Fold %d error: %s" % (fold, e))
            metrics, y_pred, y_proba = (
                {k: np.nan for k in fold_metrics},
                np.full(len(y_te), -1),
                None,
            )
        for k, v in metrics.items():
            if k in fold_metrics:
                fold_metrics[k].append(v)
        for i, (pid, tl, pl) in enumerate(zip(pids_cur[te], y_te, y_pred)):
            row = {
                "patient_id": pid,
                "fold": fold,
                "model": model_name,
                "true_label": tl,
                "pred_label": pl,
                **extra,
            }
            if y_proba is not None:
                for c in range(y_proba.shape[1]):
                    row["prob_class_%d" % c] = y_proba[i, c]
            preds.append(row)
        print(
            "      Fold %d  best_n=%-5s  AUC=%.4f F1=%.4f"
            % (
                fold,
                str(best_n) if best_n else "N/A",
                metrics.get("auc", np.nan),
                metrics.get("f1_macro", np.nan),
            )
        )
    return aggregate(fold_metrics), fold_metrics["auc"]


def _cv_fm_reg(X, y, pids, model_name, fold_splits, fold_data, extra, preds):
    uses_es = model_name in ("XGBoost", "CatBoost")
    fold_metrics = {k: [] for k in ["mae", "rmse", "r2"]}
    pid_to_y = {str(p): yy for p, yy in zip(pids, y)} if fold_data is not None else None

    for fold, (train_set, test_set) in enumerate(fold_splits):
        if fold_data is not None:
            if fold not in fold_data:
                continue
            X_cur, pids_cur = fold_data[fold]
            y_cur = np.array([pid_to_y[str(p)] for p in pids_cur], dtype=y.dtype)
        else:
            X_cur, pids_cur, y_cur = X, pids, y
        tr = np.where(np.isin(pids_cur, list(train_set)))[0]
        te = np.where(np.isin(pids_cur, list(test_set)))[0]
        if len(tr) == 0 or len(te) == 0:
            continue
        X_tr, X_te, y_tr, y_te = X_cur[tr], X_cur[te], y_cur[tr], y_cur[te]
        inner = KFold(N_INNER_FOLDS, shuffle=True, random_state=GLOBAL_SEED + 1)
        itr, iva = list(inner.split(X_tr))[0]
        best_n = None
        if uses_es:
            X_itr_s, X_iva_s = scale(X_tr[itr], X_tr[iva])
            best_n = es_reg(
                model_name, X_itr_s, y_tr[itr], X_iva_s, y_tr[iva], GLOBAL_SEED
            )
        X_tr_s, X_te_s = scale(X_tr, X_te)
        try:
            m = build_reg(model_name, GLOBAL_SEED, best_n)
            m.fit(X_tr_s, y_tr)
            y_pred = m.predict(X_te_s)
            metrics = reg_metrics(y_te, y_pred)
        except Exception as e:
            print("      Fold %d error: %s" % (fold, e))
            metrics, y_pred = {k: np.nan for k in fold_metrics}, np.full(
                len(y_te), np.nan
            )
        for k, v in metrics.items():
            fold_metrics[k].append(v)
        for pid, tv, pv in zip(pids_cur[te], y_te, y_pred):
            preds.append(
                {
                    "patient_id": pid,
                    "fold": fold,
                    "model": model_name,
                    "true_label": float(tv),
                    "pred_label": float(pv),
                    **extra,
                }
            )
        print(
            "      Fold %d  best_n=%-5s  MAE=%.1f R2=%.4f"
            % (
                fold,
                str(best_n) if best_n else "N/A",
                metrics.get("mae", np.nan),
                metrics.get("r2", np.nan),
            )
        )
    return aggregate(fold_metrics), fold_metrics["r2"]


def _save_summary(all_rows, all_preds, results_dir, timestamp):
    results_df = pd.DataFrame(all_rows)
    path = os.path.join(results_dir, "benchmark_%s.csv" % timestamp)
    results_df.to_csv(path, index=False)
    print("\nBenchmark results -> %s" % path)
    if all_preds:
        pred_path = os.path.join(
            results_dir, "predictions", "predictions_ALL_%s.csv" % timestamp
        )
        pd.DataFrame(all_preds).to_csv(pred_path, index=False)
        print("Predictions (all) -> %s" % pred_path)
    return results_df


# ||  EXTERNAL TASK  (train lung1 -> test lung2)

TRAIN_VAL_SPLIT = 0.2

_TASK_COL_MAP = {
    "survived_2yr": ("survived_2yr", "clf"),
    "age": ("age", "reg"),
    "tumor_volume_class": ("tumor_volume_class", "clf"),
    "t_stage_binary": ("t_stage_binary", "clf"),
    "histology": ("histology_encoded", "clf"),
}
_FOLD_COLS = ["fold_%d_primary" % i for i in range(N_OUTER_FOLDS)]
_EXT_CV_FOLDS = None  # cached fold dicts


def _ext_get_fold(fold_idx):
    global _EXT_CV_FOLDS
    if _EXT_CV_FOLDS is None:
        _EXT_CV_FOLDS = {f["fold"]: f for f in load_cv_folds()}
    f = _EXT_CV_FOLDS[fold_idx]
    return frozenset(f["train"]), frozenset(f["test"])


def _ext_filter_to_fold_train(pids, X, y, fold_idx):
    train_set, _ = _ext_get_fold(fold_idx)
    mask = np.array([p in train_set for p in pids])
    if mask.sum() == 0:
        print("    WARNING: no patients matched fold %d train; using all." % fold_idx)
        return pids, X, y
    return pids[mask], X[mask], y[mask]


def _ext_approach(row):
    ds = row["dataset"]
    if ds == "radiomics_tumor_3D":
        return "radio_3d"
    if ds == "radiomics_tumor_2D":
        return "radio_2d"
    return str(row["Name"]).split("_")[0]  # curia / curia2 / dinov3


def load_runs(csv_path=config.RUNS_CSV):
    """Parse runs.csv (cv MLflow export) -> list of config tuples for dispatch."""
    df = pd.read_csv(csv_path)
    configs = []
    for _, row in df.iterrows():
        task = row["task"]
        if task not in _TASK_COL_MAP:
            continue
        task_col, task_type = _TASK_COL_MAP[task]
        model = row["model"]
        if SKIP_TABPFN and model == "TabPFN":
            continue
        vals = [float(row[c]) if not pd.isna(row[c]) else -np.inf for c in _FOLD_COLS]
        fold_idx = int(np.argmax(vals))
        approach = _ext_approach(row)
        pooling = None if pd.isna(row.get("pooling")) else row["pooling"]
        condition = None if pd.isna(row.get("condition")) else row["condition"]
        patch_pooling = (
            None if pd.isna(row.get("patch_pooling")) else row["patch_pooling"]
        )
        # Honor the config pooling switches so a stale runs.csv cannot reintroduce
        # strategies disabled in config.py. (radio_3d has no pooling -> never filtered.)
        is_fm = approach not in ("radio_2d", "radio_3d")
        if (
            (is_fm or approach == "radio_2d")
            and pooling is not None
            and str(pooling).strip().lower() not in config.SLICE_AGG_STRATEGIES
        ):
            continue
        if (
            is_fm
            and patch_pooling is not None
            and str(patch_pooling).strip().lower() not in config.PATCH_POOL_METHODS
        ):
            continue
        if approach in ("radio_2d", "radio_3d"):
            configs.append(
                (
                    task,
                    task_type,
                    task_col,
                    approach,
                    None,
                    None,
                    None,
                    pooling,
                    model,
                    fold_idx,
                )
            )
        else:
            configs.append(
                (
                    task,
                    task_type,
                    task_col,
                    approach,
                    condition,
                    patch_pooling,
                    pooling,
                    None,
                    model,
                    fold_idx,
                )
            )
    return configs


def _ext_load_label_df(radio_dir):
    df = radio_load(radio_dir, "radiomics_tumor_3D")
    return df.drop_duplicates("patient_id").set_index("patient_id")


def _ext_valid_lung2_pids():
    return frozenset(
        d
        for d in os.listdir(config.LUNG2_NII_BASE)
        if os.path.isdir(os.path.join(config.LUNG2_NII_BASE, d))
        and glob.glob(os.path.join(config.LUNG2_NII_BASE, d, "seg-*.nii.gz"))
    )


def _ext_train_eval(
    model_name,
    task_type,
    X_l1,
    y_l1,
    tr_idx,
    val_idx,
    X_l2,
    y_l2,
    valid_l2,
    run_label,
    extra,
    results_dir,
    timestamp,
    all_rows,
):
    is_reg = task_type == "reg"
    n_classes = len(np.unique(y_l1)) if not is_reg else 0
    X_l1_s, X_te_s, X_val_s = scale(X_l1, X_l2, X_l1[val_idx])
    X_tr_s, y_tr, y_val = X_l1_s[tr_idx], y_l1[tr_idx], y_l1[val_idx]

    best_n = None
    if model_name in ("XGBoost", "CatBoost"):
        if is_reg:
            best_n = es_reg(model_name, X_tr_s, y_tr, X_val_s, y_val, GLOBAL_SEED)
        elif len(np.unique(y_tr)) >= 2 and len(np.unique(y_val)) >= 2:
            best_n = es_clf(model_name, X_tr_s, y_tr, X_val_s, y_val, GLOBAL_SEED)

    y_proba = None
    if is_reg:
        m = build_reg(model_name, GLOBAL_SEED, best_n)
        m.fit(X_l1_s, y_l1)
        y_pred = m.predict(X_te_s)
        metrics = reg_metrics(y_l2, y_pred)
    else:
        m = build_clf(model_name, GLOBAL_SEED, best_n)
        m.fit(X_l1_s, y_l1)
        y_pred = m.predict(X_te_s)
        y_proba = m.predict_proba(X_te_s) if hasattr(m, "predict_proba") else None
        metrics = clf_metrics(y_l2, y_pred, y_proba, n_classes)

    preds = []
    for i, (pid, tl, pl) in enumerate(zip(valid_l2, y_l2, y_pred)):
        row = {
            "patient_id": pid,
            "model": model_name,
            "true_label": tl,
            "pred_label": pl,
            **extra,
        }
        if not is_reg and y_proba is not None:
            for c in range(y_proba.shape[1]):
                row["prob_class_%d" % c] = y_proba[i, c]
        preds.append(row)
    save_predictions(results_dir, preds, run_label, timestamp)

    if is_reg:
        print(
            "    n_train=%d n_test=%d best_n=%s  MAE=%.1f RMSE=%.1f R2=%.4f"
            % (
                len(X_l1_s),
                len(X_te_s),
                best_n or "N/A",
                metrics["mae"],
                metrics["rmse"],
                metrics["r2"],
            )
        )
    else:
        print(
            "    n_train=%d n_test=%d best_n=%s  AUC=%.4f F1=%.4f Acc=%.4f"
            % (
                len(X_l1_s),
                len(X_te_s),
                best_n or "N/A",
                metrics.get("auc", np.nan),
                metrics.get("f1_macro", np.nan),
                metrics.get("accuracy", np.nan),
            )
        )
    log_mlflow(
        run_label,
        {
            **extra,
            "model": model_name,
            "best_n": best_n or "NA",
            "n_train": len(X_l1_s),
            "n_test": len(X_te_s),
            "n_features": X_l1.shape[1],
        },
        metrics,
    )
    all_rows.append(
        {
            "run_label": run_label,
            "task_type": task_type,
            "model": model_name,
            **extra,
            **metrics,
        }
    )


def external_radiomics(configs, results_dir, valid_l2, timestamp, all_rows):
    print("\n" + "#" * 60 + "\nRADIOMICS\n" + "#" * 60)
    for (
        task,
        task_type,
        task_col,
        approach,
        _c,
        _pp,
        _sa,
        radio_pooling,
        model_name,
        fold_idx,
    ) in configs:
        if approach not in ("radio_2d", "radio_3d"):
            continue
        is_reg = task_type == "reg"
        is_3d = approach == "radio_3d"
        basename = "radiomics_tumor_3D" if is_3d else "radiomics_tumor_2D"
        mtag = "radiomics3d" if is_3d else "radiomics2d"
        slice_tok = "na" if is_3d else (radio_pooling or "na")
        # Same field order as the cv task, with a lung2_ prefix:
        # lung2_{model}_{seg_mask}_{patch_pooling}_{slice_pooling}_{task}_{classifier}
        run_label = "lung2_%s_tumor_na_%s_%s_%s" % (mtag, slice_tok, task, model_name)
        print("\n  [%s]" % run_label)
        if run_already_completed(results_dir, run_label):
            print("    SKIP (done)")
            continue
        try:
            df_l1 = radio_load(config.LUNG1_RADIO_DIR, basename)
            df_l2 = radio_load(config.LUNG2_RADIO_DIR, basename)
        except FileNotFoundError as e:
            print("    SKIP: %s" % e)
            continue

        feats = [c for c in radio_feat_cols(df_l1) if c in set(radio_feat_cols(df_l2))]
        if not feats:
            print("    SKIP: no common features")
            continue
        try:
            if is_3d:
                X_l1, y_l1, pids_l1 = radio_extract_3d(df_l1, task_col, feats, is_reg)
                X_l2, y_l2, pids_l2 = radio_extract_3d(df_l2, task_col, feats, is_reg)
            else:
                X_l1, y_l1, pids_l1 = radio_pool_2d(
                    df_l1, task_col, feats, radio_pooling, is_reg
                )
                X_l2, y_l2, pids_l2 = radio_pool_2d(
                    df_l2, task_col, feats, radio_pooling, is_reg
                )
        except Exception as e:
            print("    Feature extraction failed: %s" % e)
            continue

        keep = np.isin(pids_l2, list(valid_l2))
        X_l2, y_l2, pids_l2 = X_l2[keep], y_l2[keep], pids_l2[keep]
        if len(X_l1) == 0 or len(X_l2) == 0:
            print("    SKIP: empty")
            continue

        pids_l1, X_l1, y_l1 = _ext_filter_to_fold_train(pids_l1, X_l1, y_l1, fold_idx)
        if len(X_l1) == 0:
            print("    SKIP: no lung1 after fold filter")
            continue
        print("    Fold %d lung1 train: %d" % (fold_idx, len(pids_l1)))

        tr_idx, val_idx = _ext_split(X_l1, y_l1, is_reg)
        extra = {
            "task": task,
            "approach": approach,
            "radio_pooling": radio_pooling or "NA",
            "da_variant": "baseline",
        }
        try:
            _ext_train_eval(
                model_name,
                task_type,
                X_l1,
                y_l1,
                tr_idx,
                val_idx,
                X_l2,
                y_l2,
                pids_l2,
                run_label,
                extra,
                results_dir,
                timestamp,
                all_rows,
            )
        except Exception as e:
            print("    Train/predict failed: %s" % e)


def _ext_split(X, y, is_reg):
    if is_reg:
        sp = ShuffleSplit(
            n_splits=1, test_size=TRAIN_VAL_SPLIT, random_state=GLOBAL_SEED
        )
        return next(sp.split(X))
    try:
        sp = StratifiedShuffleSplit(
            n_splits=1, test_size=TRAIN_VAL_SPLIT, random_state=GLOBAL_SEED
        )
        return next(sp.split(X, y))
    except Exception:
        sp = ShuffleSplit(
            n_splits=1, test_size=TRAIN_VAL_SPLIT, random_state=GLOBAL_SEED
        )
        return next(sp.split(X))


def _ext_build_features(
    approach,
    pids,
    nii_base,
    token_dir,
    cache_dir,
    btl_dir,
    dataset,
    condition,
    slice_agg,
    patch_pool,
    fold_idx,
):
    """(X, valid_pids) for a list of patients under one (condition, pool, agg)."""
    is_cov = patch_pool == "cov"
    L = R = None
    if is_cov:
        btl_path = os.path.join(
            btl_dir, "LR_%s_fold%d_d%d.npz" % (condition, fold_idx, COV_DIM_PRIME)
        )
        if not os.path.exists(btl_path):
            print("    SKIP: cov bottleneck missing (run cv task first): %s" % btl_path)
            return np.empty((0, 1), np.float32), np.array([])
        data = np.load(btl_path)
        L, R = data["L"], data["R"]

    pid_to_mask = {}
    if condition in ("tumor", "lung"):
        for pid in pids:
            ct = load_sitk(os.path.join(nii_base, pid, "image.nii.gz"))
            if ct is None:
                continue
            tumor, lung = load_masks(dataset, pid, ct)
            m = tumor if condition == "tumor" else lung
            if m is not None and m.sum() > 0:
                pid_to_mask[pid] = m
            del ct
        gc.collect()

    vecs, valid = [], []
    for pid in pids:
        mv = pid_to_mask.get(pid) if condition in ("tumor", "lung") else None
        if is_cov:
            res = compute_patient_cov_vector(
                token_dir, cache_dir, pid, condition, L, R, mv, fold_idx
            )
        else:
            res = compute_patient_vector(token_dir, cache_dir, pid, condition, mv)
        if res is None or slice_agg not in res:
            continue
        vecs.append(res[slice_agg])
        valid.append(pid)
    if not vecs:
        return np.empty((0, PATCH_DIM or 1), np.float32), np.array([])
    return np.stack(vecs).astype(np.float32), np.array(valid)


def external_foundation(
    approach, configs, results_dir, valid_l2, label_l1, label_l2, timestamp, all_rows
):
    print("\n" + "#" * 60 + "\n%s\n" % approach.upper() + "#" * 60)
    out1, out2 = config.LUNG1_OUTPUT_DIRS[approach], config.LUNG2_OUTPUT_DIRS[approach]
    token1, token2 = os.path.join(out1, "patch_tokens"), os.path.join(
        out2, "patch_tokens"
    )
    cache1, cache2 = os.path.join(out1, "vector_cache"), os.path.join(
        out2, "vector_cache"
    )
    btl_dir = os.path.join(out1, "cov_bottleneck")
    for d in (cache1, cache2):
        os.makedirs(d, exist_ok=True)

    l2_pids_all = sorted(
        d
        for d in os.listdir(config.LUNG2_NII_BASE)
        if os.path.isdir(os.path.join(config.LUNG2_NII_BASE, d))
    )
    if not args.skip_extraction:
        extract_tokens(approach, l2_pids_all, config.LUNG2_NII_BASE, token2)
    infer_geometry_from_tokens(token1)

    for (
        task,
        task_type,
        task_col,
        cfg_approach,
        condition,
        patch_pool,
        slice_agg,
        _radio,
        model_name,
        fold_idx,
    ) in configs:
        if cfg_approach != approach:
            continue
        patch_pool = patch_pool or "mean"
        # Same field order as the cv task, with a lung2_ prefix:
        # lung2_{model}_{seg_mask}_{patch_pooling}_{slice_pooling}_{task}_{classifier}
        run_label = "lung2_%s_%s_%s_%s_%s_%s" % (
            approach,
            condition,
            patch_pool,
            slice_agg,
            task,
            model_name,
        )
        print("\n  [%s]" % run_label)
        if run_already_completed(results_dir, run_label):
            print("    SKIP (done)")
            continue

        l1_pids = [
            p
            for p in sorted(f[:-3] for f in os.listdir(token1) if f.endswith(".pt"))
            if p in label_l1.index and not pd.isna(label_l1.loc[p, task_col])
        ]
        l2_pids = [
            p
            for p in sorted(f[:-3] for f in os.listdir(token2) if f.endswith(".pt"))
            if p in valid_l2
            and p in label_l2.index
            and not pd.isna(label_l2.loc[p, task_col])
        ]
        if len(l1_pids) < 5 or len(l2_pids) == 0:
            print("    SKIP: lung1=%d lung2=%d" % (len(l1_pids), len(l2_pids)))
            continue

        is_reg = task_type == "reg"
        X_l1, valid_l1 = _ext_build_features(
            approach,
            l1_pids,
            config.LUNG1_NII_BASE,
            token1,
            cache1,
            btl_dir,
            "lung1",
            condition,
            slice_agg,
            patch_pool,
            fold_idx,
        )
        if len(valid_l1) == 0:
            print("    SKIP: no lung1 vectors")
            continue
        y_l1 = label_l1.loc[valid_l1, task_col].values.astype(
            np.float32 if is_reg else int
        )
        valid_l1, X_l1, y_l1 = _ext_filter_to_fold_train(valid_l1, X_l1, y_l1, fold_idx)
        if len(X_l1) == 0:
            print("    SKIP: no lung1 after fold filter")
            continue
        print("    Fold %d lung1 train: %d" % (fold_idx, len(valid_l1)))

        X_l2, valid_l2_f = _ext_build_features(
            approach,
            l2_pids,
            config.LUNG2_NII_BASE,
            token2,
            cache2,
            btl_dir,
            "lung2",
            condition,
            slice_agg,
            patch_pool,
            fold_idx,
        )
        if len(valid_l2_f) == 0:
            print("    SKIP: no lung2 vectors")
            continue
        y_l2 = label_l2.loc[valid_l2_f, task_col].values.astype(
            np.float32 if is_reg else int
        )

        tr_idx, val_idx = _ext_split(X_l1, y_l1, is_reg)
        extra = {
            "task": task,
            "approach": approach,
            "condition": condition,
            "slice_agg": slice_agg,
            "patch_pool": patch_pool,
            "da_variant": "baseline",
        }
        try:
            _ext_train_eval(
                model_name,
                task_type,
                X_l1,
                y_l1,
                tr_idx,
                val_idx,
                X_l2,
                y_l2,
                valid_l2_f,
                run_label,
                extra,
                results_dir,
                timestamp,
                all_rows,
            )
        except Exception as e:
            print("    Train/predict failed: %s" % e)


def run_external():
    results_dir = results_dir_for("all")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 80 + "\nEXTERNAL BENCHMARK  (train lung1 -> test lung2)\n" + "=" * 80)

    configs = load_runs()
    print("Loaded %d run configs from %s" % (len(configs), config.RUNS_CSV))
    valid_l2 = _ext_valid_lung2_pids()
    print("Lung2 patients with tumor seg (test set): %d" % len(valid_l2))

    all_rows = []
    if "radiomics" in BACKENDS:
        external_radiomics(configs, results_dir, valid_l2, timestamp, all_rows)

    if FM_BACKENDS:
        label_l1 = _ext_load_label_df(config.LUNG1_RADIO_DIR)
        label_l2 = _ext_load_label_df(config.LUNG2_RADIO_DIR)
        for approach in FM_BACKENDS:
            external_foundation(
                approach,
                configs,
                results_dir,
                valid_l2,
                label_l1,
                label_l2,
                timestamp,
                all_rows,
            )

    if not all_rows:
        print("\nNo results to save.")
        return
    results_df = pd.DataFrame(all_rows)
    path = os.path.join(results_dir, "external_results_%s.csv" % timestamp)
    results_df.to_csv(path, index=False)
    print("\nResults -> %s" % path)


# main


def main():
    if TASK == "cv":
        for backend in BACKENDS:
            if backend == "radiomics":
                cv_radiomics()
            else:
                cv_foundation(backend)
    else:
        run_external()


if __name__ == "__main__":
    main()
