"""Bootstrap CI evaluation for cross-validation prediction CSVs.

Out-of-fold predictions are concatenated across folds (each patient appears once)
and bootstrapped over the full set.

Filename format:
  lung1: predictions_{model}_{seg}_{ppooling}_{spooling}_{task}_{clf}_{ts}.csv
  lung2: predictions_lung2_{model}_{seg}_{ppooling}_{spooling}_{task}_{clf}_{ts}.csv
"""

import re
import numpy as np
import pandas as pd
import glob
import argparse
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.utils import resample

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


_MODELS = {"curia2", "curia", "dinov3", "radiomics2d", "radiomics3d"}
_SEG_MASKS = {"lung", "tumor", "noseg", "na"}
_PATCH_POOLING = {"mean", "cov", "na"}
_SLICE_POOLING = {"mean", "max", "median", "na"}


def _parse_filename(stem: str) -> dict:
    # Peel classifier (last token) then match known-vocab tokens left-to-right;
    # whatever remains in the middle is the task (may be multi-token).
    if stem.startswith("predictions_"):
        stem = stem[len("predictions_") :]

    parts = stem.split("_")

    if (
        len(parts) >= 2
        and re.fullmatch(r"\d{6}", parts[-1])
        and re.fullmatch(r"\d{8}", parts[-2])
    ):
        parts = parts[:-2]

    if parts and parts[-1].lower() == "baseline":
        parts = parts[:-1]

    if parts and parts[0].lower() == "lung2":
        dataset = "lung2"
        parts = parts[1:]
    else:
        dataset = "lung1"

    classifier = parts[-1] if parts else None
    parts = parts[:-1]

    matched = []
    for vocab in (_MODELS, _SEG_MASKS, _PATCH_POOLING, _SLICE_POOLING):
        if len(parts) >= 2 and f"{parts[0]}_{parts[1]}".lower() in vocab:
            matched.append(f"{parts[0]}_{parts[1]}")
            parts = parts[2:]
        elif parts and parts[0].lower() in vocab:
            matched.append(parts[0])
            parts = parts[1:]
        else:
            matched.append(None)
    m_match, seg_match, p_match, s_match = matched
    task = "_".join(parts) if parts else None

    return {
        "dataset": dataset,
        "model": m_match.lower() if m_match else None,
        "seg_mask": seg_match.lower() if seg_match else None,
        "patch_pooling": p_match.lower() if p_match else None,
        "slice_pooling": s_match.lower() if s_match else None,
        "task": task,
        "classifier": classifier,
    }


def _eval_one(ids, input_tuple, score_fn, input_tuple2=None, score_fn_kwargs={}):
    result = score_fn(*[t[ids] for t in input_tuple], **score_fn_kwargs)
    if input_tuple2 is not None:
        result = result - score_fn(*[t[ids] for t in input_tuple2], **score_fn_kwargs)
    return result


def empirical_bootstrap(
    input_tuple,
    score_fn,
    ids=None,
    n_iterations=1000,
    alpha=0.95,
    score_fn_kwargs={},
    threads=None,
    input_tuple2=None,
    ignore_nans=False,
    chunksize=50,
):
    from multiprocessing import Pool
    from functools import partial

    if not isinstance(input_tuple, tuple):
        input_tuple = (input_tuple,)
    if input_tuple2 is not None and not isinstance(input_tuple2, tuple):
        input_tuple2 = (input_tuple2,)

    _orig_fn = score_fn

    def score_fn(*args, **kw):
        return np.atleast_1d(_orig_fn(*args, **kw))

    score_point = (
        score_fn(*input_tuple, **score_fn_kwargs)
        if input_tuple2 is None
        else score_fn(*input_tuple, **score_fn_kwargs)
        - score_fn(*input_tuple2, **score_fn_kwargs)
    )

    if n_iterations == 0:
        return score_point, np.zeros_like(score_point), np.zeros_like(score_point), []

    if ids is None:
        ids = np.array(
            [
                resample(range(len(input_tuple[0])), n_samples=len(input_tuple[0]))
                for _ in range(n_iterations)
            ]
        )

    fn = partial(
        _eval_one,
        input_tuple=input_tuple,
        score_fn=score_fn,
        input_tuple2=input_tuple2,
        score_fn_kwargs=score_fn_kwargs,
    )

    if threads == 0:
        results = np.array([fn(ids[i]) for i in range(n_iterations)]).astype(np.float32)
    else:
        results = []
        for istart in tqdm(np.arange(0, n_iterations, chunksize), desc="Bootstrap"):
            iend = min(n_iterations, istart + chunksize)
            pool = Pool(threads)
            results.append(np.array(pool.map(fn, ids[istart:iend])).astype(np.float32))
            pool.close()
            pool.join()
        results = np.concatenate(results, axis=0)

    percentile_fn = np.nanpercentile if ignore_nans else np.percentile
    score_diff = results - score_point
    score_low = score_point + percentile_fn(
        score_diff, ((1.0 - alpha) / 2.0) * 100, axis=0
    )
    score_high = score_point + percentile_fn(
        score_diff, (alpha + (1.0 - alpha) / 2.0) * 100, axis=0
    )

    if ignore_nans:
        return score_point, score_low, score_high, np.sum(np.isnan(score_diff), axis=0)
    else:
        return score_point, score_low, score_high, ids


def is_classification(df: pd.DataFrame) -> bool:
    return "prob_class_1" in df.columns


def get_metrics(df: pd.DataFrame) -> dict:
    y_true = df["true_label"].values

    if is_classification(df):
        y_prob = df["prob_class_1"].values
        y_pred = df["pred_label"].values
        return {
            "AUC": (roc_auc_score, (y_true, y_prob), {}),
            "BAcc": (balanced_accuracy_score, (y_true, y_pred), {}),
        }
    else:
        y_pred = df["pred_label"].values
        return {
            "MAE": (mean_absolute_error, (y_true, y_pred), {}),
            "R2": (r2_score, (y_true, y_pred), {}),
        }


def evaluate_file(
    csv_path: str,
    n_iterations: int = 1000,
    alpha: float = 0.95,
    threads: int = 0,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    meta = _parse_filename(csv_path.stem)
    for k, v in meta.items():
        if v is None:
            meta[k] = "NA"

    n_patients = df["patient_id"].nunique()
    n_rows = len(df)
    if n_rows != n_patients:
        print(
            f"  WARNING [{csv_path.name}]: "
            f"{n_rows} rows but only {n_patients} unique patients "
            f"- some patients may appear in multiple folds."
        )

    n_folds = df["fold"].nunique() if "fold" in df.columns else 1

    records = []
    metrics = get_metrics(df)

    for metric_name, (score_fn, input_tuple, score_fn_kwargs) in metrics.items():
        input_tuple_np = tuple(np.array(a) for a in input_tuple)

        point, low, high, _ = empirical_bootstrap(
            input_tuple=input_tuple_np,
            score_fn=score_fn,
            score_fn_kwargs=score_fn_kwargs,
            n_iterations=n_iterations,
            alpha=alpha,
            threads=threads,
            ignore_nans=True,
        )

        records.append(
            {
                "dataset": meta["dataset"],
                "model": meta["model"],
                "seg_mask": meta["seg_mask"],
                "patch_pooling": meta["patch_pooling"],
                "slice_pooling": meta["slice_pooling"],
                "task": meta["task"],
                "classifier": meta["classifier"],
                "metric": metric_name,
                "n_patients": n_patients,
                "n_folds": n_folds,
                "point": float(np.squeeze(point)),
                "ci_low": float(np.squeeze(low)),
                "ci_high": float(np.squeeze(high)),
                "alpha": alpha,
                "source_file": csv_path.name,
            }
        )

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap CI evaluation for cross-validation prediction CSVs"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory containing CSV files (default: current directory)",
    )
    parser.add_argument(
        "--files", nargs="*", help="Explicit CSV file(s) or globs (overrides --dir)"
    )
    parser.add_argument("--n_iterations", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="0 = single-threaded (default), None = multiprocessing auto",
    )
    parser.add_argument("--output", type=str, default="bootstrap_results.csv")
    args = parser.parse_args()

    if args.files:
        paths = sorted(
            {p for pattern in args.files for p in (glob.glob(pattern) or [pattern])}
        )
    else:
        paths = sorted(Path(args.dir).glob("*.csv"))

    output_name = Path(args.output).name
    paths = [Path(p) for p in paths if Path(p).name != output_name]
    paths = [p for p in paths if not p.name.startswith("predictions_ALL")]

    if not paths:
        print(f"No CSV files found in {args.dir!r}.")
        return

    output_path = Path(args.output)
    already_done = set()
    write_header = True
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(output_path, usecols=["source_file"])
            already_done = set(existing["source_file"].dropna().unique())
            write_header = False
            print(
                f"Found existing {output_path.name} with {len(already_done)} processed file(s) "
                f"- will skip those and append new results."
            )
        except Exception as e:
            print(f"Could not read existing {output_path.name} ({e}); starting fresh.")

    paths_todo = [p for p in paths if p.name not in already_done]
    if len(paths_todo) < len(paths):
        print(f"Skipping {len(paths) - len(paths_todo)} already-processed file(s).")
    if not paths_todo:
        print("Nothing to do - all selected files are already in the output.")
        return

    display_cols = [
        "dataset",
        "model",
        "seg_mask",
        "patch_pooling",
        "slice_pooling",
        "task",
        "classifier",
        "metric",
        "n_patients",
        "n_folds",
        "point",
        "ci_low",
        "ci_high",
        "source_file",
    ]

    for path in paths_todo:
        print(f"\n-> {path.name}")
        try:
            result = evaluate_file(
                path,
                n_iterations=args.n_iterations,
                alpha=args.alpha,
                threads=args.threads,
            )
        except Exception as e:
            print(f"  ERROR processing {path.name}: {e}")
            continue

        result.to_csv(
            output_path,
            mode="a" if not write_header else "w",
            header=write_header,
            index=False,
        )
        write_header = False

        print(result[display_cols].to_string(index=False))
        print(f"  appended to {output_path.name}")

    print(f"\nDone -> {output_path}")


if __name__ == "__main__":
    main()
