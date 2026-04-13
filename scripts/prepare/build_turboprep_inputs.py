"""
Build turboprep input/output list files and a BrLP input CSV from an
already-converted NIfTI directory.

Expected NIfTI directory layout (produced by dicom_to_input_csv.py /
dcm2niix):
    {nifti_dir}/
    └── {SUBJECT_ID}/       e.g. 002_S_0413
        └── {IMAGE_ID}/     e.g. I240812
            └── *.nii.gz    (the NIfTI scan)
            └── *.json      (dcm2niix sidecar, ignored)

Outputs (written to --output_dir):
    tp_inputs.txt   – one NIfTI path per line   (for turboprep-multiple)
    tp_outputs.txt  – one output dir per line   (for turboprep-multiple)
    input.csv       – BrLP inference CSV pointing to turboprep results

turboprep writes two permanent files inside each output directory:
    normalized.nii.gz   → BrLP image_path
    segm.nii.gz         → BrLP segm_path

Usage:
    python scripts/prepare/build_turboprep_inputs.py \\
        --nifti_dir       /remote/nifti \\
        --metadata        ./ADNI/Random_4_10_2026.csv \\
        --preprocessed_dir /remote/preprocessed \\
        --output_dir      ./prep_inputs
"""

import argparse
import glob
import os
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_nifti(subject_dir: str, image_id: str) -> Optional[str]:
    """
    Return the first *.nii.gz file found in {subject_dir}/{image_id}/.
    Falls back to *.nii if no .nii.gz exists.
    """
    base = os.path.join(subject_dir, image_id)
    if not os.path.isdir(base):
        return None

    for pattern in (f"{image_id}*.nii.gz", "*.nii.gz", f"{image_id}*.nii", "*.nii"):
        matches = glob.glob(os.path.join(base, pattern))
        if matches:
            matches.sort(key=len)   # prefer shorter (less-suffixed) name
            return os.path.abspath(matches[0])

    return None


def sex_code(sex_str: str) -> int:
    """ADNI sex string → brlp encoding: M=1, F=2."""
    return 1 if sex_str.strip().upper() == "M" else 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build turboprep input/output files and BrLP input CSV."
    )
    parser.add_argument(
        "--nifti_dir", required=True,
        help="Root NIfTI directory ({nifti_dir}/{subject}/{image_id}/*.nii.gz).",
    )
    parser.add_argument(
        "--metadata", required=True,
        help="ADNI metadata CSV (from the ADNI portal, same file used for download).",
    )
    parser.add_argument(
        "--preprocessed_dir", required=True,
        help="Root directory where turboprep will write its outputs "
             "({preprocessed_dir}/{subject}/{image_id}/).",
    )
    parser.add_argument(
        "--output_dir", default=".",
        help="Directory to write tp_inputs.txt, tp_outputs.txt, input.csv (default: cwd).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -- load & normalise metadata -------------------------------------------
    meta = pd.read_csv(args.metadata)
    meta.columns = [c.strip().strip('"') for c in meta.columns]
    print(f"Loaded {len(meta)} rows from {args.metadata}")

    # -- deduplicate: one scan per (Subject, Acq Date) -----------------------
    meta["_acq_date"] = pd.to_datetime(meta["Acq Date"], format="%m/%d/%Y")
    meta_dedup = (
        meta.sort_values("Image Data ID")
            .drop_duplicates(subset=["Subject", "_acq_date"])
    )
    print(f"After dedup (one scan per subject×date): {len(meta_dedup)} scans")

    # -- walk nifti_dir and match against metadata ---------------------------
    # Build a lookup: image_id → metadata row
    meta_by_id = {
        str(row["Image Data ID"]).strip(): row
        for _, row in meta_dedup.iterrows()
    }

    tp_inputs  = []
    tp_outputs = []
    records    = []
    skipped    = []

    # Iterate through nifti_dir/{subject}/{image_id}/ directories
    for subject in sorted(os.listdir(args.nifti_dir)):
        subject_dir = os.path.join(args.nifti_dir, subject)
        if not os.path.isdir(subject_dir):
            continue

        for image_id in sorted(os.listdir(subject_dir)):
            nifti_path = find_nifti(subject_dir, image_id)
            if nifti_path is None:
                print(f"  [skip] No NIfTI found: {subject}/{image_id}")
                skipped.append(f"{subject}/{image_id}")
                continue

            # Look up metadata for this image_id
            row = meta_by_id.get(image_id)
            if row is None:
                print(f"  [skip] Not in metadata (may be a duplicate date): {image_id}")
                skipped.append(image_id)
                continue

            age = int(row["Age"])
            sex = sex_code(str(row["Sex"]))

            out_dir = os.path.join(
                os.path.abspath(args.preprocessed_dir), subject, image_id
            )

            tp_inputs.append(nifti_path)
            tp_outputs.append(out_dir)

            records.append({
                "subject_id": subject,
                "image_uid":  image_id,
                "age":        age,
                "sex":        sex,
                # turboprep always writes these two filenames
                "image_path": os.path.join(out_dir, "normalized.nii.gz"),
                "segm_path":  os.path.join(out_dir, "segm.nii.gz"),
            })

            print(f"  [ok] {subject}/{image_id}  age={age}  sex={'M' if sex==1 else 'F'}")

    # -- write outputs -------------------------------------------------------
    inputs_file  = os.path.join(args.output_dir, "tp_inputs.txt")
    outputs_file = os.path.join(args.output_dir, "tp_outputs.txt")
    csv_file     = os.path.join(args.output_dir, "input.csv")

    with open(inputs_file, "w") as f:
        f.write("\n".join(tp_inputs) + "\n")

    with open(outputs_file, "w") as f:
        # turboprep-multiple requires the output directories to exist beforehand
        for d in tp_outputs:
            os.makedirs(d, exist_ok=True)
        f.write("\n".join(tp_outputs) + "\n")

    pd.DataFrame(records).to_csv(csv_file, index=False)

    print(f"\nDone.")
    print(f"  Matched : {len(records)} scans")
    print(f"  Skipped : {len(skipped)}")
    print(f"  tp_inputs.txt  → {inputs_file}")
    print(f"  tp_outputs.txt → {outputs_file}")
    print(f"  input.csv      → {csv_file}")

    if skipped:
        print(f"\nSkipped image IDs: {skipped}")


if __name__ == "__main__":
    main()
