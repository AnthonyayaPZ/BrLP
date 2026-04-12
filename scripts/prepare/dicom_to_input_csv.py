"""
Convert ADNI DICOM scans to NIfTI and build an inference input CSV for brlp.

Usage:
    python scripts/prepare/dicom_to_input_csv.py \
        --adni_dir    ./ADNI \
        --metadata    ./ADNI/Random_4_10_2026.csv \
        --nifti_dir   ./nifti \
        --output_csv  ./input.csv

What it does:
  1. Reads the ADNI metadata CSV (downloaded from the ADNI portal).
  2. Locates each scan's DICOM folder inside --adni_dir using the Image Data ID.
  3. Converts each DICOM folder to NIfTI using dcm2niix.
  4. Writes input.csv with columns:
       subject_id, image_uid, age, sex, image_path
     (segm_path is omitted so brlp auto-segments via SynthSeg)

Sex encoding (required by brlp):  M=1, F=2
"""

import argparse
import os
import subprocess
import sys
import glob
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_dicom_folder(adni_dir: str, subject: str, image_id: str) -> Optional[str]:
    """
    Locate the DICOM folder for a given subject and image ID.

    Expected layout:
        {adni_dir}/{subject}/MPRAGE/{date_folder}/{image_id}/
    """
    pattern = os.path.join(adni_dir, subject, "**", image_id)
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    # Return the first directory match
    for m in matches:
        if os.path.isdir(m):
            return m
    return None


def convert_dicom_to_nifti(dicom_folder: str, output_dir: str, image_id: str) -> Optional[str]:
    """
    Run dcm2niix on a DICOM folder. Returns the path of the produced .nii.gz file,
    or None if conversion fails.
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "dcm2niix",
        "-z", "y",          # gzip output
        "-f", image_id,     # filename = image_id
        "-o", output_dir,
        dicom_folder,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [error] dcm2niix failed for {image_id}:")
        print(result.stderr[:500])
        return None

    # dcm2niix may append suffixes; find the produced .nii.gz
    candidates = glob.glob(os.path.join(output_dir, f"{image_id}*.nii.gz"))
    if not candidates:
        # sometimes dcm2niix creates .nii without gz even with -z y flag
        candidates = glob.glob(os.path.join(output_dir, f"{image_id}*.nii"))
    if not candidates:
        print(f"  [error] No NIfTI output found for {image_id} in {output_dir}")
        return None

    # Prefer the file without extra suffixes (shortest name)
    candidates.sort(key=len)
    return os.path.abspath(candidates[0])


def sex_code(sex_str: str) -> int:
    """Map ADNI sex string to brlp encoding: M=1, F=2."""
    return 1 if sex_str.strip().upper() == "M" else 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert ADNI DICOMs to NIfTI and build brlp input CSV."
    )
    parser.add_argument("--adni_dir",   required=True, help="Root ADNI data folder (contains subject subfolders).")
    parser.add_argument("--metadata",   required=True, help="ADNI metadata CSV downloaded from the portal.")
    parser.add_argument("--nifti_dir",  required=True, help="Output folder for converted NIfTI files.")
    parser.add_argument("--output_csv", required=True, help="Path for the generated input.csv.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip dcm2niix if NIfTI file already exists.")
    args = parser.parse_args()

    # -- check dcm2niix is available -----------------------------------------
    if subprocess.run(["which", "dcm2niix"], capture_output=True).returncode != 0:
        print("[error] dcm2niix not found. Install it with: brew install dcm2niix")
        sys.exit(1)

    # -- load metadata --------------------------------------------------------
    meta = pd.read_csv(args.metadata)
    # normalise column names: strip whitespace and quotes
    meta.columns = [c.strip().strip('"') for c in meta.columns]
    print(f"Loaded {len(meta)} rows from {args.metadata}")

    # -- deduplicate: keep one scan per (Subject, Acq Date) ------------------
    # Some dates have two duplicate Image IDs (e.g., I83576 / I83582 same date).
    # Keep the first occurrence so we don't feed the same timepoint twice.
    meta["_acq_date"] = pd.to_datetime(meta["Acq Date"], format="%m/%d/%Y")
    meta_dedup = (
        meta.sort_values("Image Data ID")
            .drop_duplicates(subset=["Subject", "_acq_date"])
    )
    print(f"After dedup (one scan per subject×date): {len(meta_dedup)} scans")

    # -- process each scan ----------------------------------------------------
    records = []
    skipped = []

    for _, row in meta_dedup.iterrows():
        image_id = str(row["Image Data ID"]).strip()
        subject  = str(row["Subject"]).strip()
        age      = int(row["Age"])
        sex      = sex_code(str(row["Sex"]))

        # Locate DICOM folder
        dicom_folder = find_dicom_folder(args.adni_dir, subject, image_id)
        if dicom_folder is None:
            print(f"  [skip] DICOM not found: subject={subject} image_id={image_id}")
            skipped.append(image_id)
            continue

        # Output directory per subject/scan
        out_dir = os.path.join(args.nifti_dir, subject, image_id)
        expected = os.path.join(out_dir, f"{image_id}.nii.gz")

        if args.skip_existing and os.path.exists(expected):
            print(f"  [exists] {image_id}")
            nifti_path = os.path.abspath(expected)
        else:
            print(f"  [convert] {subject} / {image_id}  age={age}  sex={'M' if sex==1 else 'F'}")
            nifti_path = convert_dicom_to_nifti(dicom_folder, out_dir, image_id)

        if nifti_path is None:
            skipped.append(image_id)
            continue

        records.append({
            "subject_id": subject,
            "image_uid":  image_id,
            "age":        age,
            "sex":        sex,
            "image_path": nifti_path,
        })

    # -- write output CSV -----------------------------------------------------
    if not records:
        print("\n[error] No scans were converted successfully.")
        sys.exit(1)

    out_df = pd.DataFrame(records).sort_values(["subject_id", "age"])
    out_df.to_csv(args.output_csv, index=False)

    print(f"\nDone.")
    print(f"  Converted : {len(records)} scans")
    print(f"  Skipped   : {len(skipped)} scans  {skipped if skipped else ''}")
    print(f"  Output CSV: {os.path.abspath(args.output_csv)}")
    print()
    print("Next steps:")
    print("  1. (Recommended) Run turboprep on each NIfTI for bias correction,")
    print("     skull stripping, MNI registration, and segmentation.")
    print("  2. Or run brlp directly — it will auto-segment via SynthSeg")
    print("     (requires FreeSurfer >= 7.4):")
    print(f"     brlp --input {args.output_csv} --confs confs.yaml \\")
    print(f"          --output ./output --target_age <AGE> --target_diagnosis <1|2|3> --steps 10")


if __name__ == "__main__":
    main()
