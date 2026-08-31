# pcnn-defocus

Training and evaluation code for a convolutional neural network that detects
and localizes (x, y, z) defocused particle images, using a single-particle calibration stack as the point-spread-function model.

The network is a compact (~140k parameter) 4-level ConvNeXt-V2 U-Net that maps
a grayscale image to five output channels per pixel:

| channel | meaning |
|---|---|
| 0 | peak-probability logits |
| 1, 2 | sub-pixel offsets (dp, dq) in pixel units |
| 3 | depth z ∈ [0, 1] |
| 4 | unused |

Training targets use soft Gaussian probability labels (σ = 0.75 px) rather than
one-hot pixels, with a focal classification loss and soft-mask-weighted
regression losses.

Every generated case also stores `peaks`, the exact (N, 4) `[x, y, z, I]`
particle list that was rendered. The label maps are the training target;
`peaks` is the evaluation ground truth.

## Repository layout

```
pcnn/                         standalone module (network, metrics, interpolation)
templates/SingleParticle_cal/ 51-plane single-particle calibration stack (B00001–B00051.tif)
network/Defocus_PCNN_net00_Epoch00100.pth  final trained network (fixed-stride standard)
example_data/                 one synthetic test case per noise level (.npz)
                              with a grayscale .png preview of each image
generate_synthetic.py         synthesize the training + evaluation datasets
train.py                      train the network
evaluate.py                   detection / localization metrics on the eval set
make_figures.py               reproduce the publication figures
```

## Setup

```
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for data generation and training.
Figures render with LaTeX text when a `latex` executable is on PATH, and fall
back to matplotlib's mathtext otherwise.

## Reproducing the results

```
# 1. Synthesize data
#    train: 3 densities x 2 noise levels x 5000 tiles at 256^2  (~30 GB)
#    eval : 15 densities x 2 noise levels x 32 tiles at 512^2
python generate_synthetic.py
python generate_synthetic.py --split eval   # evaluation set only
python generate_synthetic.py --examples     # refresh example_data/ only

# 2. Train (optional — the final trained network is included)
#    Loads the full training set into RAM (~47 GB); checkpoints go to network/runs/
python train.py

# 3. Evaluate on the synthetic test grid (and the MicroSIG set, if present)
#    -> results/detection_metrics.csv, results/localization_metrics.csv,
#       results/microsig_metrics.csv
python evaluate.py                        # uses network/Defocus_PCNN_net00_Epoch00100.pth
python evaluate.py --ckpt network/runs/<run>/<checkpoint>.pth

# 4. Figures -> results/figures/
python make_figures.py
```

The MicroSIG benchmark frames (DefocusTracking JP-MST01-21), are not redistributed here; download them from
<https://defocustracking.com/Datasets/JP-MST01-21.zip> and unpack them into
`data/microsig/`, or point `evaluate.py` elsewhere with the `MICROSIG_DIR`
environment variable or `--microsig-dir`. If the directory is absent, evaluation
simply skips the MicroSIG metrics and the detection figure omits the μSIG
overlay.

`make_figures.py` produces:

| file | content |
|---|---|
| `fig_calstack.gif` | animated z-sweep through the calibration stack |
| `fig_density_range.eps` | example images at 4 seeding densities (top: low noise, bottom: high noise) |
| `fig_input_patch.eps` | example network input patch |
| `fig_labels.eps` | ground-truth label channels with a detail view |
| `fig_qualitative.eps` | detections (red ×) vs ground truth (cyan ○) at 3 densities; top half low-noise, bottom half high-noise (white divider) |
| `fig_detection.eps` | precision / recall / F1 vs seeding density (low-noise / high-noise rows), with the MicroSIG results overlaid |
| `fig_localization.eps` | lateral (xy) and depth (z) localization error vs seeding density — RMSE and median, both noise levels |

The image-only figures (`calstack`, `density_range`, `labels`) need no data or
model; `qualitative` and the metric figures require steps 1 and 3.

## Example data

`example_data/` holds one full 512² evaluation-style test case per noise level
(σₙ = 2.5e-3 and 6.25e-2) at a seeding density of 0.005 particles per pixel,
each with a grayscale preview PNG for a quick visual check. Each `.npz`
contains `img` (H × W float32), `gt` (5 × H × W float32, channels as above),
and `peaks` (N × 4 float32, `[x, y, z, I]` -- the exact particle list).
They are copied straight out of the generated evaluation set by
`python generate_synthetic.py --examples`, which also rewrites the preview PNGs.

Minimal usage:

```python
import numpy as np, torch
from pcnn import PeakCNN_UNet_4level_ConvNeXt, get_peaks_z_nms

case = np.load("example_data/case_0.005ppp_0.0025Noise_low_noise.npz")
img = torch.from_numpy(case["img"]).float()
img = (img - img.mean()) / img.std()          # per-sample z-score

model = PeakCNN_UNet_4level_ConvNeXt.load_checkpoint("network/Defocus_PCNN_net00_Epoch00100.pth").eval()
with torch.no_grad():
    out = model(img[None, None])[0]

xy, prob, z = get_peaks_z_nms(out, thr=0.5)   # [N,2] peak coords, prob, depth
```

## Notes

- Inputs are z-score normalized per image; the network is fully convolutional,
  so inference can run at any resolution. Image sizes that are not multiples of 16
  use interpolation for fallback which may degrade performance.
- Detection metrics use 3×3 non-maximum suppression on the probability map and
  greedy nearest-first matching with a 1-pixel tolerance.
- The `pcnn/` module is fully self-contained; this repository has no external
  dependency beyond the packages listed in `requirements.txt`.

## License

Released under the 3-clause BSD license -- see [LICENSE.md](LICENSE.md).
