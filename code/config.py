"""Central configuration for the benchmark.

All paths are derived from ROOT (env var, defaults to ./data).
HF_TOKEN, TABPFN_TOKEN, and MLFLOW_* credentials are also read from the environment.
"""

import os

ROOT = os.environ.get("ROOT", os.path.join(os.getcwd(), "data"))

# datasets
#   lung1 = NSCLC-Radiomics            (cross-validation / model selection)
#   lung2 = NSCLC-Radiogenomics        (external test set)

LUNG1_NII_BASE = os.path.join(ROOT, "NSCLC-Radiomics-NIFTI")
LUNG2_NII_BASE = os.path.join(ROOT, "NSCLC-Radiogenomics-NIFTI")

# lung2 has no native lung segmentation; CLIP-derived left/right lung masks live
# here as "<pid>/<pid>_Left Lung.nii.gz" / "<pid>_Right Lung.nii.gz".  Only used
# for the "lung" condition on lung2; "tumor"/"noseg" do not need it.
# https://github.com/ljwztc/clip-driven-universal-model
LUNG2_CLIP_SEG_PATH = os.path.join(ROOT, "lung2_clip_segs")

LUNG1_CLINICAL_CSV = os.path.join(
    ROOT, "lung1", "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
)
LUNG2_CLINICAL_CSV = os.path.join(
    ROOT, "lung2", "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
)

# radiomics feature CSVs (output of extract_radiomics.py)
LUNG1_RADIO_DIR = os.path.join(ROOT, "radiomics_output")
LUNG2_RADIO_DIR = os.path.join(ROOT, "radiomics_output_lung2")

# cross-validation fold definition (shared by both datasets)
FOLD_JSON = os.path.join(ROOT, "cv_folds.json")

# MLflow export of completed lung1 CV runs; used by external task to pick
# the best fold/config per task
RUNS_CSV = os.path.join(ROOT, "runs.csv")

# foundation-model output trees
#   Each model writes patch_tokens/, vector_cache/ and cov_bottleneck/ here.

LUNG1_OUTPUT_DIRS = {
    "curia": os.path.join(ROOT, "curia_output"),
    "curia2": os.path.join(ROOT, "curia2_output"),
    "dinov3": os.path.join(ROOT, "dinov3_output"),
}
LUNG2_OUTPUT_DIRS = {
    "curia": os.path.join(ROOT, "curia_output_lung2"),
    "curia2": os.path.join(ROOT, "curia2_output_lung2"),
    "dinov3": os.path.join(ROOT, "dinov3_output_lung2"),
}

# model identifiers

CURIA_HF_ID = "raidium/curia"
CURIA2_HF_ID = "raidium/curia-2"
DINOV3_TIMM_ID_DEFAULT = "vit_base_patch16_dinov3.lvd1689m"

# shared hyperparameters

GLOBAL_SEED = 42
GBDT_MAX_ROUNDS = 10_000
EARLY_STOPPING_ROUNDS = 50
TABPFN_N_ESTIMATORS = 32

CT_HU_MIN = -1000
CT_HU_MAX = 400

PATCH_OVERLAP_THRESHOLD = 0.1
CONDITIONS = ["noseg", "lung", "tumor"]
SLICE_AGG_STRATEGIES = ["mean", "median", "max"]
PATCH_POOL_METHODS = ["cov", "mean"]

COV_DIM_PRIME = 64

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5

# mlflow

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD")


def setup_mlflow(experiment_name):
    """Configure MLflow from the environment.  Returns the active mlflow module,
    or None if no tracking URI is configured (in which case logging is a no-op).
    """
    if not MLFLOW_TRACKING_URI:
        print("MLflow: MLFLOW_TRACKING_URI not set -> logging disabled.")
        return None
    import mlflow

    if MLFLOW_USERNAME:
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
    if MLFLOW_PASSWORD:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        print("MLflow -> %s  experiment='%s'" % (MLFLOW_TRACKING_URI, experiment_name))
        return mlflow
    except Exception as e:
        print("Warning: MLflow setup failed: %s" % e)
        return None
