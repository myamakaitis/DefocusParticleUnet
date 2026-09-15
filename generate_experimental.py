"""Training data for the EXPERIMENTAL case: a measured microscope calibration.

Same network, same labels, same loss as the synthetic case.  What changes is the
imaging model, and it changes in three ways:

    PSF      templates/Channel_cal -- a 49x49 z-stack measured on one aperture
             of the calibration target, at the centre of the field.  The
             synthetic case uses a 256^2 MicroSIG-rendered stack at 2.5 template
             px per image px; this one is already at image scale.

    DEPTH    real micrometres over the range the calibration covers, not a bare
             [0, 1].  The stored labels are still normalised to [0, 1] so
             train.py and evaluate.py need no special case, and every .npz
             carries `z_range_um` so a prediction can be put back in micrometres.
             A CHECKPOINT IS ONLY INTERPRETABLE ALONGSIDE THE RANGE IT WAS
             TRAINED ON: the same output value means a different depth for a
             different range, which is why train.py writes it into the
             checkpoint's metadata.

    NOISE    the sensor's own, two scales drawn log-uniformly per tile:
             SHOT, Poisson in the signal, so it acts ON the particles; and
             BACKGROUND, read plus fixed pattern, SAMPLED from a dark
             calibration run (templates/Channel_dark/dark_pool.npz) rather
             than modelled,
             because a fixed pattern is not analytic -- it is this sensor's
             static per-pixel structure.  The synthetic case instead adds white
             Gaussian noise at one of two fixed levels.

Everything else -- tile size, particle count, amplitude spread, the soft
Gaussian labels -- is shared with generate_synthetic, and the rendering is
literally its GenSampledPSFTile, so the two cases cannot drift apart on the
parts that must match.

Outputs .npz files (img + gt + peaks + z_range_um) under
data/experimental/{train,eval}/.

Usage:
    python generate_experimental.py                 # train + eval
    python generate_experimental.py --split train   # training set only
"""
import os
import csv
import glob
import argparse
from itertools import product

import numpy as np
import torch
from PIL import Image

from pcnn import GridSampleInterpolator
import generate_synthetic as gs

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PSF_DIR   = os.path.join(REPO_ROOT, "templates", "Channel_cal")
DARK_DIR  = os.path.join(REPO_ROOT, "templates", "Channel_dark")
EXP_DIR   = os.path.join(REPO_ROOT, "data", "experimental")

GEN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Depth range the network is trained over, in micrometres.
#
# THIS IS NOT THE SAME AS THE SHIPPED PLANES, and deliberately so.
# templates/Channel_cal/calibration.csv lists every plane available to
# interpolate between, and it extends PAST this range by one plane at the top:
# depth is interpolated linearly, so a particle at +5.00 um can only be
# rendered if the plane above it exists.  The range below is what z is SAMPLED
# over and what the [0, 1] labels are normalised by -- a training choice, not a
# property of the calibration, which is why it lives here.
#
# It must match the z_range recorded in the checkpoint's metadata: the same
# network output means a different depth under a different range.
Z_TRAINED_UM = (-68.08, 5.00)

# Tile geometry and label conventions are the synthetic case's, unchanged.
TRAIN_TILE_SIZE = gs.TRAIN_TILE_SIZE
EVAL_TILE_SIZE  = gs.EVAL_TILE_SIZE

# Sensor noise, both log-uniform per tile.  The two ranges are matched: at the
# median particle peak the shot std spans about the same counts as the
# background sigma, so training sees an even split of shot-limited and
# read-limited tiles rather than one regime.
BG_SIGMA_RANGE   = (1.0, 6.0)
SHOT_SCALE_RANGE = (0.5, 25.0)

# Training density grid, the same three means as the synthetic case.
TRAIN_PPPS    = gs.TRAIN_PPPS
TRAIN_SAMPLES = 10_000        # per ppp -> 30,000 tiles, matching the synthetic set

# Evaluation sweep.  Spans the densities the real clips actually reach
# (about 0.0009 ppp at the sparsest concentration to 0.011 at the densest).
# Noise is PINNED here rather than drawn, at the two corners of the sampled box,
# so the sweep is a controlled comparison the way the synthetic eval grid is.
EVAL_PPPS    = torch.linspace(0.001, 0.02, 9)
EVAL_NOISE   = {"low_noise":  dict(bg=BG_SIGMA_RANGE[0], shot=SHOT_SCALE_RANGE[1]),
                "high_noise": dict(bg=BG_SIGMA_RANGE[1], shot=SHOT_SCALE_RANGE[0])}
EVAL_SAMPLES = 32


# ----------------------------------------------------------------------------
# Calibration stack
# ----------------------------------------------------------------------------
_PSF_CACHE = {}
_DARK_CACHE = {}


def load_plane_z(path):
    """Plane depths in micrometres, ascending, from calibration.csv."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r['z_um']) for r in rows], dtype=np.float32)


def build_psf_interpolator(device):
    """Load templates/Channel_cal and return a (z, y, x) trilinear interpolator.

    NOT renormalised.  The planes are already unit-mean over the uncropped
    stamp, and the noise ranges above are calibrated in those units -- rescaling
    the stack would silently rescale the SNR the network trains at.  (The
    synthetic loader does divide by the stack max, because that stack is raw
    measured counts.)

    The z grid is taken as UNIFORM over the shipped planes.  They are not
    exactly evenly spaced -- calibration steps run 2.47 to 2.50 um -- but the
    deviation from an even grid is 0.016 um at worst, which is ~3 orders of
    magnitude below the network's own depth resolution.  calibration.csv
    carries the true plane positions if they are ever wanted.
    """
    key = str(device)
    if key in _PSF_CACHE:
        return _PSF_CACHE[key]

    z = load_plane_z(os.path.join(PSF_DIR, "calibration.csv"))
    files = sorted(glob.glob(os.path.join(PSF_DIR, "B*.tif")))
    if len(files) != len(z):
        raise FileNotFoundError(
            f"{PSF_DIR}: calibration.csv lists {len(z)} planes, found {len(files)} TIFFs")

    stack = np.stack([np.asarray(Image.open(f)) for f in files]).astype(np.float32)
    nz, Nq, Np = stack.shape

    cq, cp = Nq // 2, Np // 2
    y_grid = (np.arange(Nq, dtype=np.float32) - cq)    # 1 template px == 1 image px
    x_grid = (np.arange(Np, dtype=np.float32) - cp)
    z_grid = np.linspace(z[0], z[-1], nz, dtype=np.float32)

    interp = GridSampleInterpolator(
        (torch.tensor(z_grid), torch.tensor(y_grid), torch.tensor(x_grid)),
        torch.tensor(stack), interp_mode="bilinear").to(device)

    zlo, zhi = (float(v) for v in Z_TRAINED_UM)
    if zlo < z[0] - 1e-6 or zhi > z[-1] + 1e-6:
        raise ValueError(
            f"Z_TRAINED_UM {Z_TRAINED_UM} is not covered by the shipped "
            f"planes ({z[0]:+.2f} .. {z[-1]:+.2f} um) -- nothing can be "
            f"rendered there.")
    _PSF_CACHE[key] = (interp, zlo, zhi, dict(z_planes_um=z, n_planes=nz,
                                              patch_shape=(Nq, Np)))
    return _PSF_CACHE[key]


def load_dark_pool(device):
    """The dark frames of templates/Channel_dark, already reduced.

    Each was DC-removed and divided by the std of the FULL dark run it came
    from, not by its own, so together they sit at 0.97 rather than exactly 1.
    That is deliberate and must not be 'corrected': the divisor is what ties
    the background sigma range above to the noise the network was trained
    against, and renormalising here would silently rescale the SNR."""
    key = str(device)
    if key in _DARK_CACHE:
        return _DARK_CACHE[key]
    path = os.path.join(DARK_DIR, "dark_pool.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: the dark-noise pool is missing")
    with np.load(path) as d:
        pool = d["frames"].astype(np.float32)
    _DARK_CACHE[key] = torch.tensor(pool, device=device)
    return _DARK_CACHE[key]


def sample_dark_crop(pool, H, W, device):
    """A random H x W crop of a random dark frame, wrapping at the edges.

    Wrap-around rather than a bounded origin so every offset is reachable: the
    shipped frames are larger than a tile but not by much, and restricting the
    origin would keep re-drawing the same central region.
    """
    n = torch.randint(0, pool.shape[0], (1,)).item()
    frame = pool[n]
    Hc, Wc = frame.shape
    q = (torch.arange(H, device=device) + torch.randint(0, Hc, (1,)).item()) % Hc
    p = (torch.arange(W, device=device) + torch.randint(0, Wc, (1,)).item()) % Wc
    return frame[q][:, p]


def make_noise_fn(pool, bg_sigma, shot_scale, device):
    """Shot noise ON the signal, then the measured background pattern.

    The order is not arbitrary: shot noise is a property of the photon count, so
    it must be applied to the clean render before anything additive is mixed in.
    Gamma(lam, 1) has mean and variance lam, matching Poisson without forcing
    the signal onto integer counts.
    """
    def noise_fn(img):
        if shot_scale > 0:
            lam = (img * shot_scale).clamp(min=1e-6)
            img = torch.distributions.Gamma(lam, torch.ones_like(lam)).sample() / shot_scale
        if bg_sigma > 0:
            img = img + sample_dark_crop(pool, img.shape[0], img.shape[1], device) * bg_sigma
        return img
    return noise_fn


def draw_noise(rng):
    """One (bg_sigma, shot_scale), log-uniform in each.

    Log-uniform because both are scale parameters spanning more than a decade:
    sampling uniformly would spend most tiles near the noisy end and barely
    visit the clean one.
    """
    lo, hi = BG_SIGMA_RANGE
    bg = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    lo, hi = SHOT_SCALE_RANGE
    shot = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return bg, shot


# ----------------------------------------------------------------------------
# One case
# ----------------------------------------------------------------------------
def _case_folder(split, ppp, noise_tag=None):
    parts = [EXP_DIR, split, f"{ppp:.3f}ppp"]
    if noise_tag is not None:
        parts.append(noise_tag)
    return os.path.join(*parts)


def generate_one_case(split, ppp, i, tile_size, noise_tag=None, seed=None):
    device = torch.device(GEN_DEVICE)
    interp, zlo, zhi, _ = build_psf_interpolator(device)
    pool = load_dark_pool(device)

    rng = np.random.default_rng(seed)
    if noise_tag is None:
        bg, shot = draw_noise(rng)
    else:
        bg, shot = EVAL_NOISE[noise_tag]["bg"], EVAL_NOISE[noise_tag]["shot"]

    img, gt, peaks = gs.GenSampledPSFTile(
        interp,
        tile_size=tile_size,
        n_peaks=float(ppp) * (tile_size ** 2),
        noise_sigma=0.0,                      # unused: noise_fn replaces it
        device=device,
        z_min=zlo, z_max=zhi,                 # depth sampled in MICROMETRES
        noise_fn=make_noise_fn(pool, bg, shot, device),
    )

    # Normalise depth to [0, 1] over the trained range, exactly as the training
    # batcher did: the WHOLE channel is rescaled, background pixels included.
    # Those pixels are meaningless either way -- the depth loss is weighted by
    # the soft mask, so only pixels under a particle contribute -- but scaling
    # the whole map is what the network was trained against, so it is what gets
    # stored.
    span = max(zhi - zlo, 1e-9)
    gt[3] = (gt[3] - zlo) / span
    peaks[:, 2] = (peaks[:, 2] - zlo) / span

    folder = _case_folder(split, float(ppp), noise_tag)
    np.savez_compressed(
        os.path.join(folder, f"case{i:04d}.npz"),
        img=img.cpu().numpy(), gt=gt.cpu().numpy(), peaks=peaks.cpu().numpy(),
        z_range_um=np.array([zlo, zhi], dtype=np.float32),
        noise=np.array([bg, shot], dtype=np.float32))


# ----------------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------------
def run_train(n_samples=TRAIN_SAMPLES, tile_size=TRAIN_TILE_SIZE):
    """Training tiles: noise drawn per tile, so folders split by density only.

    The synthetic set nests a noise level under each density because its noise
    IS a two-level grid.  Here it is a continuous 2-D draw, and discretising it
    onto folders would change the training distribution -- a bigger departure
    than pre-generating the set in the first place.  The realised draw is stored
    in each file instead.
    """
    for ppp in TRAIN_PPPS:
        os.makedirs(_case_folder("train", float(ppp)), exist_ok=True)
    total = len(TRAIN_PPPS) * n_samples
    print(f"[train] {total} cases @ {tile_size}px -> {os.path.join(EXP_DIR, 'train')}")
    k = 0
    for ppp in TRAIN_PPPS:
        for i in range(n_samples):
            generate_one_case("train", float(ppp), i, tile_size, seed=k)
            k += 1
            if k % 500 == 0:
                print(f"  {k}/{total}", flush=True)


def run_eval(n_samples=EVAL_SAMPLES, tile_size=EVAL_TILE_SIZE):
    for ppp, tag in product(EVAL_PPPS, EVAL_NOISE):
        os.makedirs(_case_folder("eval", float(ppp), tag), exist_ok=True)
    total = len(EVAL_PPPS) * len(EVAL_NOISE) * n_samples
    print(f"[eval] {total} cases @ {tile_size}px -> {os.path.join(EXP_DIR, 'eval')}")
    k = 0
    for ppp, tag in product(EVAL_PPPS, EVAL_NOISE):
        for i in range(n_samples):
            generate_one_case("eval", float(ppp), i, tile_size, noise_tag=tag,
                              seed=1_000_000 + k)
            k += 1
        print(f"  ppp={float(ppp):.3f} {tag}: {n_samples} cases", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the experimental datasets.")
    ap.add_argument("--split", default="all", choices=("all", "train", "eval"))
    args = ap.parse_args()

    _, zlo, zhi, man = build_psf_interpolator(torch.device(GEN_DEVICE))
    zp = man['z_planes_um']
    print(f"calibration : {man['n_planes']} planes {man['patch_shape'][0]}x{man['patch_shape'][1]} px, "
          f"spanning {zp[0]:+.2f} .. {zp[-1]:+.2f} um")
    print(f"depth       : {zlo:+.2f} .. {zhi:+.2f} um "
          f"(labels normalised to [0, 1] over this range)")
    print(f"noise       : bg sigma ~ logU{BG_SIGMA_RANGE}, "
          f"shot ~ logU{SHOT_SCALE_RANGE}")
    print(f"device      : {GEN_DEVICE}")

    if args.split in ("all", "train"):
        run_train()
    if args.split in ("all", "eval"):
        run_eval()
