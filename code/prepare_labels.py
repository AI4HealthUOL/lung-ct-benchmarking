"""Merge clinical labels into the radiomics feature CSVs, producing the
``*_labelled.csv`` files consumed by benchmark.py.

Works for both datasets via ``--dataset``:
  lung1 = NSCLC-Radiomics       (cross-validation / model selection)
  lung2 = NSCLC-Radiogenomics   (external test set)

Label derivation (identical for both datasets):
  age               numeric age at diagnosis
  histology_encoded 1 = squamous cell carcinoma, 0 = other, NaN = unknown
  t_stage_binary    T1/T2 -> 0 (early), T3/T4 -> 1 (advanced), Tis/Tx -> NaN
  survived_2yr      Dead  & time <= 730 -> 0
                    Dead  & time >  730 -> 1
                    Alive & time >  730 -> 1
                    otherwise (too short / unknown) -> NaN
  tumor_volume_mm3  original_shape_VoxelVolume from the 3D radiomics (mm3)
  tumor_volume_class 1 if tumor_volume_mm3 > threshold else 0

The volume-class threshold is the **median lung1 tumor volume** so the class
boundary is identical for both datasets.  For lung1 it is computed from the
lung1 3D volumes; for lung2 it is read back from the lung1 labelled CSV (or
overridden with ``--volume_threshold``).
"""

import os
import re
import argparse

import numpy as np
import pandas as pd

import code.config as config

# cli

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--dataset", choices=["lung1", "lung2"], required=True)
parser.add_argument(
    "--clinical_csv",
    type=str,
    default=None,
    help="Override the dataset's default clinical CSV path.",
)
parser.add_argument(
    "--volume_threshold",
    type=float,
    default=None,
    help="lung2 only: tumor-volume threshold (mm3) for "
    "tumor_volume_class.  Defaults to the lung1 median.",
)
args = parser.parse_args()

DATASET = args.dataset
IS_LUNG1 = DATASET == "lung1"

RADIO_DIR = config.LUNG1_RADIO_DIR if IS_LUNG1 else config.LUNG2_RADIO_DIR
CLINICAL_CSV = args.clinical_csv or (
    config.LUNG1_CLINICAL_CSV if IS_LUNG1 else config.LUNG2_CLINICAL_CSV
)
LUNG1_RADIO_3D_LABELLED = os.path.join(
    config.LUNG1_RADIO_DIR, "radiomics_tumor_3D_labelled.csv"
)

# shared label encoders


def encode_histology(val):
    if pd.isna(val):
        return np.nan
    v = str(val).lower().strip()
    if "squamous" in v:
        return 1
    if v in ("not collected", "n/a", ""):
        return np.nan
    return 0


def encode_t_stage(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip().upper()
    if v in ("NOT COLLECTED", "N/A", "TX", "TIS", ""):
        return np.nan
    m = re.match(r"^T?(\d)", v)
    if not m:
        return np.nan
    return 0 if int(m.group(1)) <= 2 else 1


def encode_survived_2yr(time, event):
    if pd.isna(event):
        return np.nan
    if event == 1:
        if pd.isna(time):
            return np.nan
        return 1 if time > 730 else 0
    if pd.isna(time) or time <= 730:
        return np.nan
    return 1


# load + parse clinical csv

print("Loading clinical data (%s) from:\n  %s" % (DATASET, CLINICAL_CSV))
clin = pd.read_csv(CLINICAL_CSV)
clin.columns = [c.strip() for c in clin.columns]
for col in clin.select_dtypes(include="object").columns:
    clin[col] = clin[col].str.strip()


def _first_col(predicate):
    return next((c for c in clin.columns if predicate(c.lower())), None)


if IS_LUNG1:
    for candidate in ("PatientID", "Case ID", "patient_id", "caseid"):
        if candidate in clin.columns:
            clin = clin.rename(columns={candidate: "patient_id"})
            break
    if "patient_id" not in clin.columns:
        raise ValueError("No patient-ID column found. Columns: %s" % list(clin.columns))

    age_col = _first_col(lambda c: c == "age")
    hist_col = _first_col(lambda c: "histology" in c or "hist" in c)
    t_col = _first_col(lambda c: "t.stage" in c or "tstage" in c or "t_stage" in c)
    surv_col = _first_col(
        lambda c: "survival.time" in c or ("time" in c and "death" in c)
    )
    evt_col = _first_col(lambda c: "deadstatus" in c or ("event" in c and "dead" in c))

    clin["age"] = pd.to_numeric(clin[age_col], errors="coerce") if age_col else np.nan
    clin["histology_encoded"] = (
        clin[hist_col].apply(encode_histology) if hist_col else np.nan
    )
    clin["t_stage_binary"] = clin[t_col].apply(encode_t_stage) if t_col else np.nan
    clin["Survival.time"] = (
        pd.to_numeric(clin[surv_col], errors="coerce") if surv_col else np.nan
    )
    clin["deadstatus.event"] = (
        pd.to_numeric(clin[evt_col], errors="coerce") if evt_col else np.nan
    )

else:
    clin = clin.rename(columns={"Case ID": "patient_id"})
    clin["age"] = pd.to_numeric(clin["Age at Histological Diagnosis"], errors="coerce")
    clin["histology_encoded"] = clin["Histology"].apply(encode_histology)
    clin["t_stage_binary"] = clin["Pathological T stage"].apply(encode_t_stage)
    clin["Survival.time"] = pd.to_numeric(
        clin["Time to Death (days)"].replace("N/A", np.nan), errors="coerce"
    )
    status = clin["Survival Status"].str.strip().str.lower()
    clin["deadstatus.event"] = (status == "dead").astype(float)
    clin.loc[~status.isin(["alive", "dead"]), "deadstatus.event"] = np.nan

clin["patient_id"] = clin["patient_id"].astype(str).str.strip()
clin["survived_2yr"] = clin.apply(
    lambda r: encode_survived_2yr(r["Survival.time"], r["deadstatus.event"]), axis=1
)
print("  %d patients in clinical CSV" % len(clin))

LABEL_COLS = [
    "patient_id",
    "age",
    "histology_encoded",
    "t_stage_binary",
    "survived_2yr",
    "deadstatus.event",
    "Survival.time",
]
label_df = clin[LABEL_COLS].copy()

# volume threshold (shared class boundary = lung1 median)

volume_threshold = args.volume_threshold
if volume_threshold is None and not IS_LUNG1:
    if os.path.exists(LUNG1_RADIO_3D_LABELLED):
        lung1_df = pd.read_csv(LUNG1_RADIO_3D_LABELLED)
        if "tumor_volume_mm3" in lung1_df.columns:
            volume_threshold = float(lung1_df["tumor_volume_mm3"].median())
            print("Volume threshold from lung1 median: %.1f mm3" % volume_threshold)
        else:
            print("WARNING: tumor_volume_mm3 not in lung1 CSV; using lung2 median.")
    else:
        print(
            "WARNING: lung1 labelled CSV not found at\n  %s\n"
            "  Falling back to lung2 median." % LUNG1_RADIO_3D_LABELLED
        )

# merge into radiomics CSVs

pid_to_volume_mm3 = {}

for suffix in ("3D", "2D"):
    in_csv = os.path.join(RADIO_DIR, "radiomics_tumor_%s.csv" % suffix)
    out_csv = os.path.join(RADIO_DIR, "radiomics_tumor_%s_labelled.csv" % suffix)
    if not os.path.exists(in_csv):
        print("SKIP %s - file not found: %s" % (suffix, in_csv))
        continue

    df = pd.read_csv(in_csv)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df = df.merge(label_df, on="patient_id", how="left")

    if suffix == "3D":
        vol_col = next(
            (
                c
                for c in df.columns
                if "shape" in c.lower() and "voxelvolume" in c.lower()
            ),
            None,
        )
        if vol_col is not None:
            df["tumor_volume_mm3"] = df[vol_col].astype(float)
            pid_to_volume_mm3 = (
                df.drop_duplicates("patient_id")
                .set_index("patient_id")["tumor_volume_mm3"]
                .to_dict()
            )
        else:
            print("  [3D] WARNING: VoxelVolume not found; tumor_volume_mm3 = NaN")
            df["tumor_volume_mm3"] = np.nan
        if volume_threshold is None:
            vol_vals = df["tumor_volume_mm3"].dropna()
            if len(vol_vals) > 0:
                volume_threshold = float(vol_vals.median())
                print(
                    "Volume threshold (%s 3D median): %.1f mm3"
                    % (DATASET, volume_threshold)
                )
            else:
                print("WARNING: no valid volumes; tumor_volume_class will be NaN")
    else:
        df["tumor_volume_mm3"] = (
            df["patient_id"].map(pid_to_volume_mm3) if pid_to_volume_mm3 else np.nan
        )

    if volume_threshold is not None:
        df["tumor_volume_class"] = (df["tumor_volume_mm3"] > volume_threshold).astype(
            float
        )
        df.loc[df["tumor_volume_mm3"].isna(), "tumor_volume_class"] = np.nan
    else:
        df["tumor_volume_class"] = np.nan

    df.to_csv(out_csv, index=False)
    print(
        "\n%s: %d rows, %d patients  ->  %s"
        % (suffix, len(df), df["patient_id"].nunique(), out_csv)
    )

    for col in [
        "age",
        "histology_encoded",
        "t_stage_binary",
        "survived_2yr",
        "tumor_volume_class",
    ]:
        if col in df.columns:
            vals = df.drop_duplicates("patient_id")[col]
            n_valid = vals.notna().sum()
            print(
                "  %-25s  valid=%d/%d  values=%s"
                % (
                    col,
                    n_valid,
                    len(vals),
                    (
                        dict(zip(*np.unique(vals.dropna(), return_counts=True)))
                        if n_valid > 0
                        else "all NaN"
                    ),
                )
            )

print(
    "\nVolume threshold used: %s mm3"
    % ("%.1f" % volume_threshold if volume_threshold is not None else "N/A")
)
print("Done.")
