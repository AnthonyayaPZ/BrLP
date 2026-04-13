"""
Interactive NIfTI viewer for macOS (Solution 1 — local device).

Displays three orthogonal views (axial / coronal / sagittal) of a NIfTI file
with a slider to scroll through each axis.

Supports a single .nii/.nii.gz file or a directory of files. When a directory
is given, a "File" slider lets you switch between scans without reopening the
script.

Dependencies (install once):
    pip install nibabel matplotlib numpy

Approximate disk space:
    nibabel   ~  5 MB
    matplotlib ~ 50 MB
    numpy      ~ 30 MB
    ─────────────────
    total      ~85 MB

Approximate RAM when running:
    Loading a typical ADNI T1 (256×256×170, float32) occupies ~45 MB.
    matplotlib figure adds ~20–50 MB.
    Total: ~100–150 MB per scan (only one scan is held in RAM at a time).

Usage — single file:
    python scripts/visualize/view_nii_local.py /path/to/scan.nii.gz

Usage — directory (opens each file in turn via a slider):
    python scripts/visualize/view_nii_local.py /path/to/nifti_dir
    python scripts/visualize/view_nii_local.py /path/to/nifti_dir --cmap hot
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.widgets import Slider


def collect_files(input_path: str) -> list:
    """Return a sorted list of NIfTI file paths from a file or directory."""
    if os.path.isfile(input_path):
        return [os.path.abspath(input_path)]
    if os.path.isdir(input_path):
        files = sorted(
            glob.glob(os.path.join(input_path, "**", "*.nii.gz"), recursive=True)
            + glob.glob(os.path.join(input_path, "**", "*.nii"), recursive=True)
        )
        if not files:
            raise SystemExit(f"[error] No .nii/.nii.gz files found in {input_path}")
        return files
    raise SystemExit(f"[error] Input path not found: {input_path}")


def load_volume(path: str) -> np.ndarray:
    img = nib.load(path)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    # Squeeze to 3-D (drop extra dims, e.g. time)
    while data.ndim > 3:
        data = data[..., 0]
    # Normalise to [0, 1] for display
    lo, hi = data.min(), data.max()
    if hi > lo:
        data = (data - lo) / (hi - lo)
    return data


def build_viewer(files: list, cmap: str) -> None:
    """Open a single matplotlib window that browses all files and slices."""
    # State shared across callbacks
    state = {"file_idx": 0, "data": None}

    def load_file(idx: int) -> np.ndarray:
        print(f"Loading [{idx + 1}/{len(files)}] {files[idx]} ...")
        return load_volume(files[idx])

    state["data"] = load_file(0)

    def current_shape():
        return state["data"].shape  # (nx, ny, nz)

    # ---- figure layout -------------------------------------------------
    # Extra row of sliders when multiple files are present
    n_slider_rows = 4 if len(files) > 1 else 3
    fig_height = 5 + 0.4 * n_slider_rows
    fig, axes = plt.subplots(1, 3, figsize=(14, fig_height))
    plt.subplots_adjust(bottom=0.08 * n_slider_rows, hspace=0.4)

    def title():
        return os.path.basename(files[state["file_idx"]])

    fig.suptitle(title(), fontsize=9)

    nx, ny, nz = current_shape()
    ix, iy, iz = nx // 2, ny // 2, nz // 2

    # ---- initial images ------------------------------------------------
    im_sag = axes[0].imshow(
        np.rot90(state["data"][ix, :, :]), cmap=cmap, origin="lower", vmin=0, vmax=1
    )
    axes[0].set_title(f"Sagittal  (x={ix})")
    axes[0].axis("off")

    im_cor = axes[1].imshow(
        np.rot90(state["data"][:, iy, :]), cmap=cmap, origin="lower", vmin=0, vmax=1
    )
    axes[1].set_title(f"Coronal   (y={iy})")
    axes[1].axis("off")

    im_axl = axes[2].imshow(
        np.rot90(state["data"][:, :, iz]), cmap=cmap, origin="lower", vmin=0, vmax=1
    )
    axes[2].set_title(f"Axial     (z={iz})")
    axes[2].axis("off")

    # ---- slice sliders -------------------------------------------------
    ax_sx = plt.axes([0.10, 0.13, 0.25, 0.03])
    ax_sy = plt.axes([0.40, 0.13, 0.25, 0.03])
    ax_sz = plt.axes([0.70, 0.13, 0.25, 0.03])

    slider_x = Slider(ax_sx, "Sag x", 0, nx - 1, valinit=ix, valstep=1)
    slider_y = Slider(ax_sy, "Cor y", 0, ny - 1, valinit=iy, valstep=1)
    slider_z = Slider(ax_sz, "Axl z", 0, nz - 1, valinit=iz, valstep=1)

    def refresh_slices():
        x_ = int(slider_x.val)
        y_ = int(slider_y.val)
        z_ = int(slider_z.val)
        d = state["data"]
        im_sag.set_data(np.rot90(d[x_, :, :]))
        axes[0].set_title(f"Sagittal  (x={x_})")
        im_cor.set_data(np.rot90(d[:, y_, :]))
        axes[1].set_title(f"Coronal   (y={y_})")
        im_axl.set_data(np.rot90(d[:, :, z_]))
        axes[2].set_title(f"Axial     (z={z_})")
        fig.canvas.draw_idle()

    slider_x.on_changed(lambda _: refresh_slices())
    slider_y.on_changed(lambda _: refresh_slices())
    slider_z.on_changed(lambda _: refresh_slices())

    # ---- file slider (only when multiple files) ------------------------
    if len(files) > 1:
        ax_sf = plt.axes([0.10, 0.04, 0.80, 0.03])
        slider_f = Slider(
            ax_sf, "File", 0, len(files) - 1,
            valinit=0, valstep=1,
            color="steelblue",
        )

        def on_file_change(val):
            idx = int(slider_f.val)
            if idx == state["file_idx"]:
                return
            state["file_idx"] = idx
            state["data"] = load_file(idx)
            nx_, ny_, nz_ = state["data"].shape

            # Update slider ranges to match new volume dimensions
            slider_x.valmax = nx_ - 1
            slider_x.set_val(nx_ // 2)
            slider_y.valmax = ny_ - 1
            slider_y.set_val(ny_ // 2)
            slider_z.valmax = nz_ - 1
            slider_z.set_val(nz_ // 2)

            fig.suptitle(title(), fontsize=9)
            refresh_slices()

        slider_f.on_changed(on_file_change)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive NIfTI viewer.")
    parser.add_argument(
        "input",
        help="Path to a .nii/.nii.gz file, or a directory to scan recursively.",
    )
    parser.add_argument(
        "--cmap", default="gray",
        help="Matplotlib colormap (default: gray). Try 'hot', 'viridis', etc.",
    )
    args = parser.parse_args()

    files = collect_files(args.input)
    print(f"Found {len(files)} file(s).")
    build_viewer(files, cmap=args.cmap)


if __name__ == "__main__":
    main()