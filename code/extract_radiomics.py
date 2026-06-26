"""PyRadiomics feature extraction for both lung datasets.

Extracts the 105 standard classical radiomic features (IBSI-compliant) from
tumor segmentation of every patient, in two modes:
  3D : one row per patient (whole tumor volume)
  2D : one row per axial slice that contains tumor

Feature set (105):
  14 shape, 18 first-order, 22 GLCM, 16 GLRLM, 16 GLSZM, 14 GLDM, 5 NGTDM
  GLCM excludes SumAverage (= 2*JointAverage) and MCC (non-standard).

Tumor mask per patient:
  seg-GTV-1.nii.gz if present (lung1), otherwise the first seg-*.nii.gz found
  (lung2 converted DICOM-SEG).
"""

import os
import sys
import glob
import logging
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

import code.config as config

try:
    from radiomics import featureextractor, setVerbosity

    setVerbosity(logging.WARNING)
except ImportError:
    sys.exit("PyRadiomics not found.  pip install pyradiomics")

# cli

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--dataset", choices=["lung1", "lung2"], required=True)
parser.add_argument(
    "--workers",
    type=int,
    default=0,
    help="Parallel worker processes (0 = all CPUs / SLURM allocation).",
)
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--only2d", action="store_true", help="Run 2D extraction only.")
mode.add_argument("--only3d", action="store_true", help="Run 3D extraction only.")
args = parser.parse_args()

DATASET = args.dataset
NII_BASE = config.LUNG1_NII_BASE if DATASET == "lung1" else config.LUNG2_NII_BASE
OUT_DIR = config.LUNG1_RADIO_DIR if DATASET == "lung1" else config.LUNG2_RADIO_DIR
RUN_3D = not args.only2d
RUN_2D = not args.only3d

# Minimum tumor voxels in a slice to attempt 2D extraction
MIN_VOXELS_2D = 10

# extractor

GLCM_STANDARD = [
    "Autocorrelation",
    "ClusterProminence",
    "ClusterShade",
    "ClusterTendency",
    "Contrast",
    "Correlation",
    "DifferenceAverage",
    "DifferenceEntropy",
    "DifferenceVariance",
    "Id",
    "Idm",
    "Idmn",
    "Idn",
    "Imc1",
    "Imc2",
    "InverseVariance",
    "JointAverage",
    "JointEnergy",
    "JointEntropy",
    "MaximumProbability",
    "SumEntropy",
    "SumSquares",
]

FEATURE_CLASSES = ["shape", "firstorder", "glrlm", "glszm", "gldm", "ngtdm"]

_BASE_SETTING = {
    "binWidth": 25,
    "resampledPixelSpacing": [1, 1, 1],
    "interpolator": "sitkBSpline",
    "normalize": False,
    "removeOutliers": None,
    "geometryTolerance": 1e-6,
    "minimumROIDimensions": 1,
    "minimumROISize": 1,
    "preCrop": True,
}


def build_extractor(force2d=False):
    setting = dict(_BASE_SETTING)
    if force2d:
        setting["force2D"] = True
        setting["force2Ddimension"] = 0
    ex = featureextractor.RadiomicsFeatureExtractor()
    ex.settings.update(setting)
    ex.disableAllImageTypes()
    ex.enableImageTypeByName("Original")
    ex.disableAllFeatures()
    for fc in FEATURE_CLASSES:
        ex.enableFeatureClassByName(fc)
    ex.enableFeaturesByName(glcm=GLCM_STANDARD)
    logging.getLogger("radiomics.glcm").setLevel(logging.ERROR)
    return ex


# image / mask helpers


def find_tumor_seg(pid):
    gtv = os.path.join(NII_BASE, pid, "seg-GTV-1.nii.gz")
    if os.path.exists(gtv):
        return gtv
    segs = sorted(glob.glob(os.path.join(NII_BASE, pid, "seg-*.nii.gz")))
    return segs[0] if segs else None


def clamp_ct(ct_img):
    arr = np.clip(
        sitk.GetArrayFromImage(ct_img), config.CT_HU_MIN, config.CT_HU_MAX
    ).astype(np.float32)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ct_img)
    return out


def align_and_binarize(seg_img, ct_img):
    if seg_img.GetSize() != ct_img.GetSize():
        r = sitk.ResampleImageFilter()
        r.SetReferenceImage(ct_img)
        r.SetInterpolator(sitk.sitkNearestNeighbor)
        r.SetDefaultPixelValue(0)
        seg_img = r.Execute(seg_img)
    arr = (sitk.GetArrayFromImage(seg_img) > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ct_img)
    return out


def _clean_row(result, extra):
    row = dict(extra)
    for k, v in result.items():
        if not k.startswith("diagnostics_"):
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                pass
    return row


# per-patient worker


def process_patient(pid):
    logging.getLogger("radiomics").setLevel(logging.WARNING)
    result = {"3D": [], "2D": []}

    ct_path = os.path.join(NII_BASE, pid, "image.nii.gz")
    seg_path = find_tumor_seg(pid)
    if not os.path.exists(ct_path):
        print("  [%s] SKIP - CT not found" % pid, flush=True)
        return result
    if seg_path is None:
        print("  [%s] SKIP - no tumor seg found" % pid, flush=True)
        return result

    ct_img = clamp_ct(sitk.ReadImage(ct_path))
    seg_img = align_and_binarize(sitk.ReadImage(seg_path), ct_img)
    seg_arr = sitk.GetArrayFromImage(seg_img)  # [Z, Y, X] uint8
    if seg_arr.sum() == 0:
        print("  [%s] SKIP - empty mask" % pid, flush=True)
        return result

    # -- 3D ----------------------------------------------------------------
    if RUN_3D:
        try:
            res = build_extractor(force2d=False).execute(ct_img, seg_img)
            result["3D"].append(
                _clean_row(res, {"patient_id": pid, "mode": "3D", "slice": None})
            )
        except Exception as e:
            print("  [%s] 3D error: %s" % (pid, e), flush=True)

    # -- 2D per slice -----------------------------------------------------
    if RUN_2D:
        ex2d = build_extractor(force2d=True)
        ct_arr = sitk.GetArrayFromImage(ct_img)
        sx, sy, sz = (float(ct_img.GetSpacing()[i]) for i in range(3))
        ok = 0
        for z in range(ct_arr.shape[0]):
            if seg_arr[z].sum() < MIN_VOXELS_2D:
                continue
            ct_sl = sitk.GetImageFromArray(ct_arr[z][np.newaxis].astype(np.float32))
            seg_sl = sitk.GetImageFromArray(seg_arr[z][np.newaxis])
            ct_sl.SetSpacing((sx, sy, sz))
            seg_sl.SetSpacing((sx, sy, sz))
            try:
                res = ex2d.execute(ct_sl, seg_sl)
                result["2D"].append(
                    _clean_row(res, {"patient_id": pid, "mode": "2D", "slice": int(z)})
                )
                ok += 1
            except Exception:
                pass
        print("  [%s] 2D: %d slices extracted" % (pid, ok), flush=True)

    return result


# main


def save(rows, suffix):
    if not rows:
        print("  [WARN] no %s rows" % suffix)
        return
    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "radiomics_tumor_%s.csv" % suffix)
    df.to_csv(out_path, index=False)
    n_feat = len([c for c in df.columns if c not in ("patient_id", "mode", "slice")])
    print(
        "  %s: %d rows x %d features  (%d patients)  ->  %s"
        % (suffix, len(df), n_feat, df["patient_id"].nunique(), out_path)
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pids = sorted(
        d for d in os.listdir(NII_BASE) if os.path.isdir(os.path.join(NII_BASE, d))
    )

    n_workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    print(
        "Dataset: %s   patients: %d   workers: %d   mode: %s"
        % (
            DATASET,
            len(pids),
            n_workers,
            "2D only" if args.only2d else "3D only" if args.only3d else "3D + 2D",
        )
    )

    rows = {"3D": [], "2D": []}
    if n_workers <= 1:
        iterator = (process_patient(p) for p in pids)
    else:
        pool = Pool(processes=n_workers)
        iterator = pool.imap_unordered(process_patient, pids)

    for i, res in enumerate(tqdm(iterator, total=len(pids), desc="Patients")):
        rows["3D"].extend(res["3D"])
        rows["2D"].extend(res["2D"])
        if (i + 1) % 10 == 0:
            if RUN_3D:
                save(rows["3D"], "3D")
            if RUN_2D:
                save(rows["2D"], "2D")

    if n_workers > 1:
        pool.close()
        pool.join()

    print("\n" + "=" * 60 + "\nEXTRACTION SUMMARY\n" + "=" * 60)
    if RUN_3D:
        save(rows["3D"], "3D")
    if RUN_2D:
        save(rows["2D"], "2D")
    print("Done.")


if __name__ == "__main__":
    main()
