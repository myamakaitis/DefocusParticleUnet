"""Synthetic data generation for the peak-detection / localization CNN.

Renders synthetic particle images from the MicroSIG-rendered single-particle
calibration stack (templates/SingleParticle_cal, 51 z-planes) by trilinear
interpolation, and produces 5-channel soft-label ground truth:

    ch 0  mask  — 2-D Gaussian (sigma = GT_SIGMA) on the continuous sub-pixel
                  position, max-combined over a GT_WINDOW_SIZE neighbourhood
    ch 1  dp    — x offset from the pixel centre, x_c - x_pixel
    ch 2  dq    — y offset from the pixel centre, y_c - y_pixel
    ch 3  z     — defocus depth in [0, 1]
    ch 4  I     — peak amplitude (LogNormal, mean 1); vestigial, never trained

Every case also stores `peaks`, the exact (N, 4) [x, y, z, I] particle list that was
rendered.  The label maps encode at most one particle per pixel (ch 0 max-combined,
chs 1-3 nearest particle only); `peaks` does not.  train.py uses the label maps,
evaluate.py uses `peaks`.

Outputs .npz files (img + gt + peaks) under data/synthetic/{train,eval}/.

Usage:
    python generate_synthetic.py                 # generate train + eval sets
    python generate_synthetic.py --split eval    # evaluation set only
    python generate_synthetic.py --examples      # refresh example_data/ only
"""
import os
import argparse
from itertools import product

import numpy as np
import torch
from torch.distributions import Uniform, Normal, LogNormal
from joblib import Parallel, delayed
from PIL import Image

from pcnn import GridSampleInterpolator

# ----------------------------------------------------------------------------
# Paths (relative to this file)
# ----------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PSF_DIR   = os.path.join(REPO_ROOT, "templates", "SingleParticle_cal")
SYNTH_DIR = os.path.join(REPO_ROOT, "data", "synthetic")

# Generation device.  GPU is much faster (rendering is one big grid_sample per
# chunk of peaks) and runs sequentially; on CPU-only machines the loky-parallel
# driver is used across cores instead.
GEN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_JOBS     = 1 if GEN_DEVICE == "cuda" else -1

# PSF calibration stack geometry
PSF_TILE_SIZE = 256
NZ            = 51
PSF_XY_SCALE  = 2.5     # template px per image px
PSF_CENTER    = 127     # centre pixel of the 256² calibration tiles

# Image / label geometry.  Training tiles are 256²; evaluation tiles are 512².
TRAIN_TILE_SIZE = 256
EVAL_TILE_SIZE  = 512
GT_SIGMA        = 0.75  # soft-label Gaussian sigma (pixels)
GT_WINDOW_SIZE  = 5     # neighbourhood (odd) over which each peak paints its Gaussian

# Peak / intensity sampling
I_STD   = 1.0 / np.e    # LogNormal amplitude spread (mean 1.0)
PPP_STD = 0.2           # std of particle count as a fraction of the mean
Z_MIN   = 0.0
Z_MAX   = 1.0

# Gaussian dark-noise standard deviations (low / high)
NOISE_LOW  = 2.5e-3
NOISE_HIGH = 6.25e-2

# Dataset grids
TRAIN_PPPS    = torch.tensor([0.001, 0.005, 0.025])
TRAIN_NOISE   = (NOISE_LOW, NOISE_HIGH)
TRAIN_SAMPLES = 5000                          # samples per (ppp, noise) folder

EVAL_PPPS     = torch.linspace(0.001, 0.05, 15)
EVAL_NOISE    = (NOISE_LOW, NOISE_HIGH)
EVAL_SAMPLES  = 32

# example_data/: one evaluation case per noise level, copied out of the generated
# eval set so the shipped examples always match the current .npz format.
EXAMPLE_DIR   = os.path.join(REPO_ROOT, "example_data")
EXAMPLE_PPP   = 0.005
EXAMPLE_CASE  = 0
EXAMPLE_TAGS  = {NOISE_LOW: "low_noise", NOISE_HIGH: "high_noise"}


# ----------------------------------------------------------------------------
# PSF interpolator (cached per device)
# ----------------------------------------------------------------------------
_PSF_CACHE = {}


def build_psf_interpolator(device):
    """Load the calibration stack and return a (z, y, x) trilinear interpolator."""
    key = str(device)
    if key in _PSF_CACHE:
        return _PSF_CACHE[key]

    psf_stack = np.zeros((NZ, PSF_TILE_SIZE, PSF_TILE_SIZE), dtype=np.float32)
    for i in range(NZ):
        psf_stack[i] = np.asarray(Image.open(os.path.join(PSF_DIR, f"B{i + 1:05d}.tif")))
    psf_stack /= psf_stack.max()

    x_grid = (np.arange(0, PSF_TILE_SIZE, dtype=np.float32) - PSF_CENTER) / PSF_XY_SCALE
    y_grid = (np.arange(0, PSF_TILE_SIZE, dtype=np.float32) - PSF_CENTER) / PSF_XY_SCALE
    z_grid = np.linspace(0, 1, NZ, dtype=np.float32)

    grid = (torch.tensor(z_grid), torch.tensor(y_grid), torch.tensor(x_grid))
    values = torch.tensor(psf_stack, dtype=torch.float32)
    interp = GridSampleInterpolator(grid, values, interp_mode='bilinear').to(device)

    _PSF_CACHE[key] = interp
    return interp


# ----------------------------------------------------------------------------
# Soft Gaussian ground-truth maps
# ----------------------------------------------------------------------------
def _make_gt_maps(p_in, q_in, ip_in, iq_in, z_in, amp_in,
                  H, W, gt_sigma, gt_window_size, device):
    R = (gt_window_size - 1) // 2

    dq_off = torch.arange(-R, R + 1, device=device)
    dp_off = torch.arange(-R, R + 1, device=device)
    DQ, DP = torch.meshgrid(dq_off, dp_off, indexing='ij')
    DQ = DQ.reshape(-1)
    DP = DP.reshape(-1)

    q_nb = iq_in[:, None] + DQ[None, :]
    p_nb = ip_in[:, None] + DP[None, :]
    valid = (q_nb >= 0) & (q_nb < H) & (p_nb >= 0) & (p_nb < W)

    dp_sub = p_in[:, None] - (ip_in[:, None] + DP[None, :]).float()
    dq_sub = q_in[:, None] - (iq_in[:, None] + DQ[None, :]).float()

    gauss = torch.exp(-(dp_sub ** 2 + dq_sub ** 2) / (2 * gt_sigma ** 2)) * valid.float()

    flat_idx   = (q_nb.clamp(0, H - 1) * W + p_nb.clamp(0, W - 1)).reshape(-1)
    gauss_flat = gauss.reshape(-1)

    mask_flat = torch.zeros(H * W, device=device)
    mask_flat.scatter_reduce_(0, flat_idx, gauss_flat, reduce='amax', include_self=True)

    winner = (gauss_flat == mask_flat[flat_idx]) & (gauss_flat > 0)

    dp_out = torch.zeros(H * W, device=device)
    dq_out = torch.zeros(H * W, device=device)
    z_out  = torch.zeros(H * W, device=device)
    I_out  = torch.zeros(H * W, device=device)

    w_idx = flat_idx[winner]
    dp_out.scatter_(0, w_idx, dp_sub.reshape(-1)[winner])
    dq_out.scatter_(0, w_idx, dq_sub.reshape(-1)[winner])
    z_out.scatter_(0,  w_idx, z_in[:, None].expand_as(gauss).reshape(-1)[winner])
    I_out.scatter_(0,  w_idx, amp_in[:, None].expand_as(gauss).reshape(-1)[winner])

    return (mask_flat.view(H, W),
            dp_out.view(H, W), dq_out.view(H, W),
            z_out.view(H, W),  I_out.view(H, W))


# ----------------------------------------------------------------------------
# Single-tile render + soft labels
# ----------------------------------------------------------------------------
def GenSampledPSFTile(discretePSF, tile_size, n_peaks, noise_sigma, device,
                      n_peak_std=PPP_STD, I_std=I_STD, z_min=Z_MIN, z_max=Z_MAX,
                      gt_sigma=GT_SIGMA, gt_window_size=GT_WINDOW_SIZE):
    """Render one synthetic image, its 5-channel soft GT map, and the exact
    particle list.

    Returns
    -------
    img   : (H, W) float32    — synthetic grayscale image (raw, un-normalized)
    gt    : (5, H, W) float32 — [mask_soft, dp, dq, z, I], the training target
    peaks : (N, 4) float32    — exact [x, y, z, I] per rendered particle
    """
    H = W = int(tile_size)

    U = Uniform(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))

    MeanILN  = torch.tensor(1.0, device=device)
    muILN    = torch.log(MeanILN ** 2 / torch.sqrt(MeanILN ** 2 + I_std ** 2))
    sigmaILN = torch.sqrt(torch.log(1 + (I_std / MeanILN) ** 2))
    ILN      = LogNormal(muILN, sigmaILN)

    N = Normal(0.0, noise_sigma)

    # --- sample peaks --------------------------------------------------------
    mean_np = float(n_peaks)
    if n_peak_std > 0:
        n_peaks = torch.normal(torch.tensor(mean_np), torch.tensor(n_peak_std * mean_np)).item()
    n_peaks = max(int(abs(n_peaks)), 1)

    cx = U.sample((n_peaks,)) * (W - 1)
    cy = U.sample((n_peaks,)) * (H - 1)
    cz = U.sample((n_peaks,)) * (z_max - z_min) + z_min
    Ip = ILN.sample((n_peaks,)).flatten()

    # --- render image by accumulating interpolated PSFs ----------------------
    # One grid_sample over a chunk of peaks at a time: query[i] = PSF at
    # (z_i, y_pixel - y_i, x_pixel - x_i).  Chunking bounds memory regardless
    # of density / tile size.
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ys = yy.reshape(-1)
    xs = xx.reshape(-1)
    HW = H * W
    chunk = max(1, 4_000_000 // HW)   # ~4M query points per grid_sample call

    img = torch.zeros(HW, dtype=torch.float32, device=device)
    with torch.no_grad():
        for s in range(0, n_peaks, chunk):
            zc, yc, xc = cz[s:s + chunk], cy[s:s + chunk], cx[s:s + chunk]
            ic = Ip[s:s + chunk]
            P = zc.shape[0]
            zq = zc[:, None].expand(P, HW)
            yq = ys[None, :] - yc[:, None]
            xq = xs[None, :] - xc[:, None]
            g = discretePSF(torch.stack([zq, yq, xq], dim=-1))   # (P, HW)
            img += (ic[:, None] * g).sum(0)

    img = img.reshape((H, W))
    img += N.sample(img.shape).to(device)

    # --- soft Gaussian ground-truth labels ------------------------------------
    ip, iq = cx.round().long(), cy.round().long()
    inbounds = (ip >= 0) & (ip < W) & (iq >= 0) & (iq < H)
    p_in,  q_in   = cx[inbounds], cy[inbounds]
    ip_in, iq_in  = ip[inbounds], iq[inbounds]
    z_in,  amp_in = cz[inbounds], Ip[inbounds]

    mask, dp_gt, dq_gt, z_gt, I_gt = _make_gt_maps(
        p_in, q_in, ip_in, iq_in, z_in, amp_in,
        H, W, gt_sigma, gt_window_size, device,
    )

    gt = torch.stack([mask, dp_gt, dq_gt, z_gt, I_gt])

    # Exact particle list.  Every sampled particle is rendered into the image and
    # lies inside the frame by construction (cx in [0, W-1], cy in [0, H-1]).
    peaks = torch.stack([cx, cy, cz, Ip], dim=1)
    return img, gt, peaks


# ----------------------------------------------------------------------------
# Job driver
# ----------------------------------------------------------------------------
def _case_folder(split, ppp, noise_sigma):
    return os.path.join(SYNTH_DIR, split, f"{ppp:.3f}ppp", f"{noise_sigma:.4f}Noise")


def generate_one_case(split, ppp, noise_sigma, i, tile_size):
    device = torch.device(GEN_DEVICE)
    discretePSF = build_psf_interpolator(device)   # cached per device

    img, gt, peaks = GenSampledPSFTile(
        discretePSF,
        tile_size=tile_size,
        n_peaks=float(ppp) * (tile_size ** 2),
        noise_sigma=float(noise_sigma),
        device=device,
    )

    folder = _case_folder(split, float(ppp), float(noise_sigma))
    np.savez_compressed(os.path.join(folder, f"case{i:04d}.npz"),
                        img=img.cpu().numpy(), gt=gt.cpu().numpy(),
                        peaks=peaks.cpu().numpy())


def export_examples():
    """Copy one eval case per noise level into example_data/, with a preview PNG.

    Sources the cases from data/synthetic/eval so the shipped examples carry
    whatever the current generator writes (img, gt, peaks).  The preview is a
    pixel-exact grayscale render of `img` clipped at its 99.9th percentile —
    the same display convention make_figures.py uses.
    """
    os.makedirs(EXAMPLE_DIR, exist_ok=True)
    for noise_sigma, tag in EXAMPLE_TAGS.items():
        src = os.path.join(_case_folder("eval", EXAMPLE_PPP, noise_sigma),
                           f"case{EXAMPLE_CASE:04d}.npz")
        if not os.path.exists(src):
            print(f"  [examples] {src} missing - run --split eval first.")
            continue
        stem = f"case_{EXAMPLE_PPP:.3f}ppp_{noise_sigma:.4f}Noise_{tag}"
        case = np.load(src)
        np.savez_compressed(os.path.join(EXAMPLE_DIR, stem + ".npz"), **case)

        img = case["img"]
        vmax = float(np.quantile(img, 0.999))
        g = np.clip(img / vmax, 0.0, 1.0) if vmax > 0 else np.zeros_like(img)
        Image.fromarray(np.round(g * 255).astype(np.uint8), mode="L").save(
            os.path.join(EXAMPLE_DIR, stem + ".png"))
        print(f"  [examples] {stem}  ({case['peaks'].shape[0]} particles)"
              if "peaks" in case.files else f"  [examples] {stem}")


def build_jobs(split, ppps, noises, n_samples, tile_size):
    jobs = []
    for ppp, noise_sigma in product(ppps, noises):
        os.makedirs(_case_folder(split, float(ppp), float(noise_sigma)), exist_ok=True)
        for i in range(n_samples):
            jobs.append((split, float(ppp), float(noise_sigma), i, tile_size))
    return jobs


def run(split, ppps, noises, n_samples, tile_size, n_jobs=N_JOBS):
    jobs = build_jobs(split, ppps, noises, n_samples, tile_size)
    print(f"[{split}] {len(jobs)} cases @ {tile_size}px -> {os.path.join(SYNTH_DIR, split)}")
    if n_jobs == 1:
        for job in jobs:
            generate_one_case(*job)
    else:
        Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(generate_one_case)(*job) for job in jobs
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the synthetic datasets.")
    ap.add_argument("--split", default=None, choices=("all", "train", "eval"),
                    help="which dataset to (re)generate (default: all, unless "
                         "--examples is given alone). The eval set is ~960 tiles; "
                         "the train set is ~30 GB.")
    ap.add_argument("--examples", action="store_true",
                    help="refresh example_data/ from the existing eval set "
                         "(implied by --split eval/all). On its own it generates "
                         "no datasets.")
    args = ap.parse_args()

    # --examples on its own must not trigger a 30 GB training run: only fall back
    # to "all" when no split was requested and no other work was asked for.
    split = args.split if args.split is not None else (None if args.examples else "all")

    if split in ("all", "train"):
        run("train", TRAIN_PPPS, TRAIN_NOISE, TRAIN_SAMPLES, TRAIN_TILE_SIZE)
    if split in ("all", "eval"):
        run("eval",  EVAL_PPPS,  EVAL_NOISE,  EVAL_SAMPLES,  EVAL_TILE_SIZE)
    if args.examples or split in ("all", "eval"):
        export_examples()
