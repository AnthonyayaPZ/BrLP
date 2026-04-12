"""
Validate downloaded BrLP model weight files.

Checks:
  1. File exists and is not an HTML page (i.e., not a failed download)
  2. File is a valid PyTorch checkpoint (torch.load succeeds)
  3. All expected parameter keys are present and shapes match the architecture
  4. A quick forward pass produces output of the correct shape

Usage:
    python scripts/validate_weights.py \
        --autoencoder weights/autoencoder.pth \
        --unet        weights/unet.pth

Pass only the weights you have downloaded; omit flags for files you haven't.
"""

import argparse
import os
import sys

import torch


# ---------------------------------------------------------------------------
# File-level sanity checks
# ---------------------------------------------------------------------------

def check_file(path: str, label: str) -> bool:
    if not os.path.exists(path):
        print(f"  [FAIL] {label}: file not found: {path}")
        return False
    size = os.path.getsize(path)
    if size < 1 * 1024 * 1024:
        print(f"  [FAIL] {label}: file too small ({size/1024:.1f} KB) — likely a failed download")
        return False
    with open(path, "rb") as f:
        header = f.read(16)
    if header.lstrip(b"\r\n\t ").startswith(b"<"):
        print(f"  [FAIL] {label}: file is an HTML page — SharePoint login required, re-download")
        return False
    print(f"  [ok]   {label}: file looks valid ({size/1024/1024:.1f} MB)")
    return True


def load_state_dict(path: str, label: str):
    """Load and return state dict, or None on failure."""
    try:
        sd = torch.load(path, map_location="cpu")
        # some checkpoints are wrapped in a dict
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if not isinstance(sd, dict):
            print(f"  [FAIL] {label}: unexpected checkpoint format (type={type(sd).__name__})")
            return None
        print(f"  [ok]   {label}: loaded state dict with {len(sd)} keys")
        return sd
    except Exception as e:
        print(f"  [FAIL] {label}: torch.load failed: {e}")
        return None


def check_state_dict(sd: dict, model: torch.nn.Module, label: str) -> bool:
    model_keys = set(model.state_dict().keys())
    ckpt_keys  = set(sd.keys())

    missing   = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        print(f"  [FAIL] {label}: {len(missing)} missing keys, e.g.: {list(missing)[:3]}")
        return False
    if unexpected:
        # unexpected keys are a warning, not necessarily fatal
        print(f"  [warn] {label}: {len(unexpected)} unexpected keys (may be fine): {list(unexpected)[:3]}")

    # check shapes
    bad_shapes = []
    for k in model_keys & ckpt_keys:
        expected = model.state_dict()[k].shape
        got      = sd[k].shape
        if expected != got:
            bad_shapes.append(f"{k}: expected {expected}, got {got}")
    if bad_shapes:
        print(f"  [FAIL] {label}: shape mismatch on {len(bad_shapes)} keys:")
        for s in bad_shapes[:5]:
            print(f"         {s}")
        return False

    print(f"  [ok]   {label}: all {len(model_keys)} keys present with correct shapes")
    return True


def forward_pass(model: torch.nn.Module, dummy_input, label: str, expected_shape) -> bool:
    model.eval()
    try:
        with torch.no_grad():
            out = model(*dummy_input) if isinstance(dummy_input, tuple) else model(dummy_input)
        # some models return tuples (e.g. autoencoder encode returns (z, mu, logvar))
        shape = out[0].shape if isinstance(out, (tuple, list)) else out.shape
        if shape != expected_shape:
            print(f"  [FAIL] {label}: forward pass output shape {shape}, expected {expected_shape}")
            return False
        print(f"  [ok]   {label}: forward pass succeeded, output shape {shape}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: forward pass raised: {e}")
        return False


# ---------------------------------------------------------------------------
# Model-specific validators
# ---------------------------------------------------------------------------

def validate_autoencoder(path: str) -> bool:
    label = "Autoencoder"
    print(f"\n{'='*50}")
    print(f" Validating {label}: {path}")
    print(f"{'='*50}")

    if not check_file(path, label):
        return False
    sd = load_state_dict(path, label)
    if sd is None:
        return False

    from brlp.networks import init_autoencoder
    model = init_autoencoder()          # architecture only, no weights yet

    if not check_state_dict(sd, model, label):
        return False

    # load weights then run forward pass
    model.load_state_dict(sd)

    # input: (batch=1, channels=1, D=120, H=144, W=120)
    dummy = torch.zeros(1, 1, 120, 144, 120)
    return forward_pass(model, dummy, label, torch.Size([1, 1, 120, 144, 120]))


def validate_unet(path: str) -> bool:
    label = "Diffusion UNet"
    print(f"\n{'='*50}")
    print(f" Validating {label}: {path}")
    print(f"{'='*50}")

    if not check_file(path, label):
        return False
    sd = load_state_dict(path, label)
    if sd is None:
        return False

    from brlp.networks import init_latent_diffusion
    model = init_latent_diffusion()

    if not check_state_dict(sd, model, label):
        return False

    model.load_state_dict(sd)

    # UNet forward: (latent, timestep, context)
    # latent: (1, 3, 16, 20, 16) — autoencoder output at ~1/8 resolution, padded to divisible-by-4
    # context: (1, 1, 8)         — cross_attention_dim=8
    dummy_x   = torch.zeros(1, 3, 16, 20, 16)
    dummy_t   = torch.tensor([0])
    dummy_ctx = torch.zeros(1, 1, 8)
    return forward_pass(model, (dummy_x, dummy_t, dummy_ctx), label,
                        torch.Size([1, 3, 16, 20, 16]))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate BrLP pretrained weight files.")
    parser.add_argument("--autoencoder", type=str, default=None, help="Path to autoencoder.pth")
    parser.add_argument("--unet",        type=str, default=None, help="Path to unet.pth")
    args = parser.parse_args()

    if not args.autoencoder and not args.unet:
        parser.print_help()
        sys.exit(1)

    results = {}

    if args.autoencoder:
        results["Autoencoder"] = validate_autoencoder(args.autoencoder)

    if args.unet:
        results["Diffusion UNet"] = validate_unet(args.unet)

    # --- summary ---
    print(f"\n{'='*50}")
    print(" Summary")
    print(f"{'='*50}")
    all_ok = True
    for name, ok in results.items():
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All validated weights are correct.")
    else:
        print("Some weights failed validation. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
