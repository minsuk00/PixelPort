# MRI → CT inference

Turn an MRI scan into a synthesized CT (NIfTI) with a plain 3D U-Net.
The network and trained weights are bundled in `weights/`.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python infer.py --mri mri.nii.gz
python infer.py --mri mri.nii.gz --mask mask.nii.gz --save_dir out/ --preview
```

Writes `<mri_name>_synth_ct.nii.gz` (CT in Hounsfield units, with the input MRI's
affine) next to the MRI.

#### Input requirements

The MRI must be **T1-weighted at 1.5 mm isotropic spacing**, **body-masked** (zero outside
the body), matching the training data. Any orientation is fine (reoriented to RAS internally).

## Model

3D U-Net (ngf 32, 4 down-samplings, batch norm, linear output head), trained on all 843
subjects of SynthRAD2025 Task 1 (brain, head & neck, thorax, abdomen, pelvis; MR and CT
body-masked with TotalSegmentator). Loss: L1 + Anatomix ViT perceptual + segmentation-teacher Dice. CT target range
−1024..3000 HU. Inference: 256³ sliding window, overlap 0.25, fp16 autocast on CUDA.
These are fixed constants at the top of `infer.py`.

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--mri` | input MRI (`.nii` or `.nii.gz`) | **required** |
| `--mask` | body mask NIfTI; zeroes the MRI outside it before inference and sets the sCT there to −1024 HU | off |
| `--checkpoint` | trained checkpoint `.pt` | `weights/unet_translator.pt` |
| `--save_dir` | output directory | the MRI's directory |
| `--preview` | also save a 3-panel mid-slice PNG | off |

Device auto-selects CUDA (else CPU, fp32).

## Acknowledgement

The U-Net architecture is from [**anatomix**](https://github.com/neel-dey/anatomix)
(Neel Dey et al., MIT License); `network.py` is vendored from that repo.
