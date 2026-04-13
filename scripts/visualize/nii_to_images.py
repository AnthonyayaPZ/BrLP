"""
NIfTI → PNG/JPEG exporter for headless remote servers (Solution 2 — remote).

For each input file, saves:
  1. A montage PNG of evenly-spaced axial slices (default: 25 slices).
  2. Optionally, the middle axial / coronal / sagittal slice as individual files.

No display or GUI is needed — uses matplotlib's non-interactive Agg backend.

Dependencies (install once on the server):
    pip install nibabel matplotlib numpy

Usage — single file:
    python scripts/visualize/nii_to_images.py scan.nii.gz

Usage — batch (all .nii.gz under a directory):
    python scripts/visualize/nii_to_images.py /path/to/preprocessed \
        --out_dir /path/to/previews \
        --format png \
        --n_slices 25 \
        --orthogonal

Options:
    input           Path to a .nii/.nii.gz file OR a root directory to scan
                    recursively for all *.nii.gz files.
    --out_dir       Where to write images. Default: same folder as each input.
                    In batch mode a mirrored subdirectory tree is created.
    --format        Output image format: png (default) or jpg.
    --n_slices      Number of axial slices in the montage (default: 25).
    --orthogonal    Also save three individual mid-slice images
                    (axial, coronal, sagittal).
    --dpi           DPI of saved figures (default: 150).
"""

import argparse
import os
import glob
from typing import List, Optional, Tuple

# Use the non-interactive Agg backend — no display needed
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_volume(path: str) -> np.ndarray:
    img = nib.load(path)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    while data.ndim > 3:
        data = data[..., 0]
    lo, hi = np.percentile(data, 1), np.percentile(data, 99)
    data = np.clip((data - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return data


def make_montage(
    data: np.ndarray,
    n_slices: int,
    axis: int = 2,
    cmap: str = "gray",
    dpi: int = 150,
) -> plt.Figure:
    """Return a matplotlib Figure with a grid of slices along the given axis."""
    size = data.shape[axis]
    indices = np.linspace(int(size * 0.05), int(size * 0.95), n_slices, dtype=int)

    cols = min(n_slices, 5)
    rows = int(np.ceil(n_slices / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i, idx in enumerate(indices):
        if axis == 0:
            sl = np.rot90(data[idx, :, :])
        elif axis == 1:
            sl = np.rot90(data[:, idx, :])
        else:
            sl = np.rot90(data[:, :, idx])
        axes[i].imshow(sl, cmap=cmap, vmin=0, vmax=1, origin="lower")
        axes[i].set_title(f"[{idx}]", fontsize=7)
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig


def make_orthogonal(data: np.ndarray, cmap: str = "gray", dpi: int = 150) -> plt.Figure:
    """Return a Figure with the middle axial, coronal, sagittal slice."""
    nx, ny, nz = data.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    views = [
        (np.rot90(data[nx // 2, :, :]), f"Sagittal  x={nx // 2}"),
        (np.rot90(data[:, ny // 2, :]), f"Coronal   y={ny // 2}"),
        (np.rot90(data[:, :, nz // 2]), f"Axial     z={nz // 2}"),
    ]
    for ax, (sl, title) in zip(axes, views):
        ax.imshow(sl, cmap=cmap, vmin=0, vmax=1, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(
    nii_path: str,
    out_dir: Optional[str],
    fmt: str,
    n_slices: int,
    orthogonal: bool,
    dpi: int,
) -> List[str]:
    """Process one NIfTI file. Returns list of saved image paths."""
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(nii_path))
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.basename(nii_path).replace(".nii.gz", "").replace(".nii", "")
    saved = []

    print(f"  Processing {nii_path} ...")
    data = load_volume(nii_path)

    # Montage (axial)
    fig = make_montage(data, n_slices=n_slices, dpi=dpi)
    montage_path = os.path.join(out_dir, f"{stem}_montage.{fmt}")
    fig.savefig(montage_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(montage_path)

    # Orthogonal mid-slices
    if orthogonal:
        fig = make_orthogonal(data, dpi=dpi)
        orth_path = os.path.join(out_dir, f"{stem}_orthogonal.{fmt}")
        fig.savefig(orth_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(orth_path)

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert NIfTI files to PNG/JPEG images on a headless server."
    )
    parser.add_argument(
        "input",
        help="Path to a .nii/.nii.gz file, or a root directory to scan recursively.",
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory for images. Defaults to the same folder as each input.",
    )
    parser.add_argument(
        "--format", default="png", choices=["png", "jpg"],
        help="Image format (default: png).",
    )
    parser.add_argument(
        "--n_slices", type=int, default=25,
        help="Number of axial slices in the montage (default: 25).",
    )
    parser.add_argument(
        "--orthogonal", action="store_true",
        help="Also save a three-view orthogonal mid-slice image.",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="DPI of saved figures (default: 150).",
    )
    args = parser.parse_args()

    # Collect input files
    if os.path.isfile(args.input):
        nii_files = [args.input]
        use_mirror = False
    elif os.path.isdir(args.input):
        nii_files = sorted(
            glob.glob(os.path.join(args.input, "**", "*.nii.gz"), recursive=True)
            + glob.glob(os.path.join(args.input, "**", "*.nii"), recursive=True)
        )
        use_mirror = True
        print(f"Found {len(nii_files)} NIfTI files under {args.input}")
    else:
        print(f"[error] Input path not found: {args.input}")
        raise SystemExit(1)

    all_saved = []
    for nii_path in nii_files:
        # Mirror the subdirectory structure when scanning a directory
        if use_mirror and args.out_dir:
            rel = os.path.relpath(os.path.dirname(nii_path), args.input)
            out_dir = os.path.join(args.out_dir, rel)
        else:
            out_dir = args.out_dir

        saved = process_file(
            nii_path=nii_path,
            out_dir=out_dir,
            fmt=args.format,
            n_slices=args.n_slices,
            orthogonal=args.orthogonal,
            dpi=args.dpi,
        )
        all_saved.extend(saved)

    print(f"\nDone. Saved {len(all_saved)} image(s).")
    for p in all_saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
