"""
Standalone MRI -> synthetic CT inference (plain 3D U-Net).

Pipeline: load MRI -> reorient to RAS -> min-max normalize to [0,1] -> pad to a
multiple of 16 -> MONAI sliding-window inference (fp16 autocast on CUDA) ->
clamp to [0,1] -> rescale to HU -> un-pad -> save.

Usage
-----
    python infer.py --mri brain_mr.nii.gz
    python infer.py --mri brain_mr.nii.gz --save_dir out/ --preview
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
)
from network import Unet

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "weights" / "unet_translator.pt"

# Model / inference constants (match the training run of the bundled checkpoint).
NGF = 32
NUM_DOWNS = 4
RES_MULT = 16  # pad to a multiple of 2**NUM_DOWNS
CT_LO, CT_HI = -1024.0, 3000.0  # HU range the model's [0,1] output maps to
PATCH_SIZE = 256  # sliding-window ROI (cube)
SW_OVERLAP = 0.25


def clean_state_dict(state_dict):
    """Strip the ``_orig_mod.`` prefix that torch.compile adds to keys."""
    return {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()}


def unpad(data, original_shape):
    """Crop an end-padded volume back to its pre-pad spatial shape (W, H, D)."""
    w, h, d = original_shape
    return data[..., :w, :h, :d]


def preprocess(mri_path, device):
    """Load MRI -> RAS -> min-max [0,1] -> pad to a multiple of RES_MULT. Returns (tensor, original_shape, affine)."""
    pre = Compose(
        [
            LoadImaged(keys="mri", image_only=True),
            EnsureChannelFirstd(keys="mri"),
            Orientationd(keys="mri", axcodes="RAS"),
            ScaleIntensityd(keys="mri", minv=0.0, maxv=1.0),
        ]
    )
    mri = pre({"mri": str(mri_path)})["mri"]
    original_shape = tuple(mri.shape[1:])  # (W, H, D), channel-first
    affine = np.asarray(mri.affine)

    mri = DivisiblePadd(keys="mri", k=RES_MULT, method="end", mode="constant")({"mri": mri})["mri"]
    mri = torch.as_tensor(np.asarray(mri), dtype=torch.float32).unsqueeze(0).to(device)
    return mri, original_shape, affine


def save_preview(pred_hu, png_path):
    """Save a 3-panel mid-slice PNG (axial / coronal / sagittal) for a quick look."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, h, d = pred_hu.shape
    slices = [
        ("axial", np.rot90(pred_hu[:, :, d // 2])),
        ("coronal", np.rot90(pred_hu[:, h // 2, :])),
        ("sagittal", np.rot90(pred_hu[w // 2, :, :])),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, sl) in zip(axes, slices):
        ax.imshow(np.clip(sl, -1000, 1000), cmap="gray", vmin=-1000, vmax=1000)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def mri_name(mri_path):
    """Filename without the .nii / .nii.gz extension."""
    name = Path(mri_path).name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def parse_args():
    p = argparse.ArgumentParser(description="MRI -> synthetic-CT inference.")
    p.add_argument("--mri", required=True, help="Input MRI NIfTI (.nii or .nii.gz).")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Trained checkpoint (.pt).")
    p.add_argument("--save_dir", default=None, help="Output directory (default: same dir as --mri).")
    p.add_argument("--preview", action="store_true", help="Also save a mid-slice preview PNG.")
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Pass --checkpoint.")
    print(f"[infer] checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Unet(dimension=3, input_nc=1, output_nc=1, num_downs=NUM_DOWNS, ngf=NGF, norm="batch", final_act="none").to(device)
    model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]), strict=True)
    model.eval()

    mri, original_shape, affine = preprocess(args.mri, device)
    print(f"[infer] MRI shape {original_shape}")

    # fp16 autocast (not bf16: its 7-bit mantissa posterizes soft tissue in narrow windows).
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        pred = sliding_window_inference(
            inputs=mri,
            roi_size=(PATCH_SIZE,) * 3,
            sw_batch_size=1,
            predictor=model,
            overlap=SW_OVERLAP,
            device=device,
        )

    pred = unpad(pred.float().clamp(0.0, 1.0), original_shape)
    pred_hu = (pred * (CT_HI - CT_LO) + CT_LO).cpu().numpy().squeeze()

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.mri).resolve().parent
    save_dir.mkdir(parents=True, exist_ok=True)
    name = mri_name(args.mri)
    out_path = save_dir / f"{name}_synth_ct.nii.gz"
    nib.save(nib.Nifti1Image(pred_hu, affine), str(out_path))
    print(f"[infer] saved {out_path}  (HU range {pred_hu.min():.0f}..{pred_hu.max():.0f})")

    if args.preview:
        png_path = save_dir / f"{name}_synth_ct_preview.png"
        save_preview(pred_hu, str(png_path))
        print(f"[infer] saved {png_path}")


if __name__ == "__main__":
    main()
