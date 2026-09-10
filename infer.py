"""MRI -> synthetic CT inference with a plain 3D U-Net.

python infer.py --mri mr.nii.gz
python infer.py --mri mr.nii.gz --mask mask.nii.gz --save_dir out/ --preview
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, DivisiblePadd, EnsureChannelFirstd, LoadImaged, Orientationd, ScaleIntensityd

from network import Unet

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "weights" / "pixelport.pt"
NGF = 32
NUM_DOWNS = 4
PAD_MULT = 16  # 2**NUM_DOWNS
CT_LO, CT_HI = -1024.0, 3000.0  # HU range the model's [0,1] output maps to
PATCH_SIZE = 256
SW_OVERLAP = 0.25


def load_model(ckpt_path, device):
    model = Unet(input_nc=1, output_nc=1, num_downs=NUM_DOWNS, ngf=NGF).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    return model.eval()


def load_nifti(path):
    """Load a NIfTI as a channel-first RAS tensor."""
    t = [LoadImaged(keys="x", image_only=True), EnsureChannelFirstd(keys="x"), Orientationd(keys="x", axcodes="RAS")]
    return Compose(t)({"x": str(path)})["x"]


@torch.inference_mode()
def predict(model, mri, device, patch_size=PATCH_SIZE):
    """Sliding-window inference on a (1, W, H, D) MRI tensor. Returns an (W, H, D) HU array."""
    shape = mri.shape[1:]
    mri = ScaleIntensityd(keys="x", minv=0.0, maxv=1.0)({"x": mri})["x"]
    mri = DivisiblePadd(keys="x", k=PAD_MULT, method="end")({"x": mri})["x"]
    mri = torch.as_tensor(np.asarray(mri)).unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        pred = sliding_window_inference(mri, roi_size=(patch_size,) * 3, sw_batch_size=1, predictor=model, overlap=SW_OVERLAP)
    pred = pred.float().clamp(0.0, 1.0) * (CT_HI - CT_LO) + CT_LO
    return pred[0, 0, : shape[0], : shape[1], : shape[2]].cpu().numpy()


def save_preview(ct, png_path):
    """3-panel mid-slice PNG (axial / coronal / sagittal)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, h, d = ct.shape
    views = [("axial", ct[:, :, d // 2]), ("coronal", ct[:, h // 2, :]), ("sagittal", ct[w // 2, :, :])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, sl) in zip(axes, views):
        ax.imshow(np.rot90(sl), cmap="gray", vmin=-1000, vmax=1000)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="MRI -> synthetic-CT inference.")
    p.add_argument("--mri", required=True, help="Input MRI NIfTI (.nii or .nii.gz).")
    p.add_argument("--mask", default=None, help="Optional body mask NIfTI; MRI is zeroed outside it and the sCT set to -1024 HU there.")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Trained checkpoint (.pt).")
    p.add_argument("--patch_size", type=int, default=PATCH_SIZE, help="Sliding-window size; lower it (e.g. 128) for GPUs with less memory.")
    p.add_argument("--save_dir", default=None, help="Output directory (default: same dir as --mri).")
    p.add_argument("--preview", action="store_true", help="Also save a mid-slice preview PNG.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] device {device} | checkpoint {args.checkpoint}")

    mri = load_nifti(args.mri)
    mask = None
    if args.mask is not None:
        mask = np.asarray(load_nifti(args.mask))[0] > 0
        assert mask.shape == mri.shape[1:], f"mask shape {mask.shape} != MRI shape {tuple(mri.shape[1:])}"
        mri[0][~mask] = 0  # model was trained on body-masked MRI (masked BEFORE min-max normalization)

    model = load_model(args.checkpoint, device)
    ct = predict(model, mri, device, args.patch_size)
    if mask is not None:
        ct[~mask] = CT_LO

    save_dir = Path(args.save_dir) if args.save_dir is not None else Path(args.mri).resolve().parent
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.mri).name.removesuffix(".nii.gz").removesuffix(".nii")
    out_path = save_dir / f"{stem}_sct.nii.gz"
    nib.save(nib.Nifti1Image(ct, np.asarray(mri.affine)), str(out_path))
    print(f"[infer] saved {out_path}  (HU range {ct.min():.0f}..{ct.max():.0f})")

    if args.preview:
        save_preview(ct, str(save_dir / f"{stem}_sct_preview.png"))
        print(f"[infer] saved {stem}_sct_preview.png")


if __name__ == "__main__":
    main()
