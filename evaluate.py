"""Evaluation of the trained network on the synthetic test grid.

For every (density, noise) folder under data/synthetic/eval, sweeps the
detection threshold for the best F1 (with 3×3-NMS peak extraction and greedy
1-px matching) and computes localization RMSE in x, y, z over the matched
peaks at that threshold.

Ground truth is the exact `peaks` particle list stored in each case.

Also evaluates the MicroSIG benchmark frames (if MICROSIG_DIR is present): per-case
best F1 / precision / recall, with a per-case bias correction (see microsig_eval).

Outputs:
    results/detection_metrics.csv     — TP/FP/FN, precision, recall, F1, threshold
    results/localization_metrics.csv  — per-axis RMSE (x, y, z), lateral-xy and z
                                        RMSE + median error, and matched count
    results/microsig_metrics.csv      — MicroSIG detection metrics per case
                                        (case, ppi, ppp, P/R/F1, threshold)

Usage:
    python evaluate.py                       # uses network/Defocus_PCNN_net00_Epoch00100.pth
    python evaluate.py --ckpt path/to.pth
"""
import os
import glob
import re
import argparse

import numpy as np
import pandas as pd
import torch

from pcnn import PeakCNN_UNet_4level_ConvNeXt, get_peaks_z_nms, match_peaks, best_f1_row

# ----------------------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR    = os.path.join(REPO_ROOT, "data", "synthetic", "eval")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DEFAULT_CKPT = os.path.join(REPO_ROOT, "network", "Defocus_PCNN_net00_Epoch00100.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Z_SCALE      = 1.0                            # must match train.Z_SCALE
THRESHOLDS   = np.linspace(0.05, 0.95, 37)
DIST_THRESH  = 1.0
EVAL_NOISE   = (2.5e-3, 6.25e-2)              # low, high (matches folder labels)
EVAL_SAMPLES = 32

# Independent MicroSIG benchmark set (DefocusTracking JP-MST01-21) -- a separate
# simulator, NOT experimental data.  Not redistributed with this repository;
# download and unpack it from
#   https://defocustracking.com/Datasets/JP-MST01-21.zip
# Each case is a `{case}.txt` (frame, X, Y, Z, ...) + a `{case}/B#####.tif` stack;
# the case name encodes the seeding as "<n>ppi" (particles per image).
#
# Defaults to data/microsig/ inside the repository; override with the
# MICROSIG_DIR environment variable or --microsig-dir.  If the directory is
# absent the MicroSIG metrics are simply skipped.
MICROSIG_DIR = os.environ.get("MICROSIG_DIR",
                              os.path.join(REPO_ROOT, "data", "microsig"))

# MicroSIG reports positions in MATLAB's 1-based pixel-centre convention
# (centres at 1..N); this network reports 0-based centres (0..N-1).  The two
# differ by exactly one pixel in both axes, so MicroSIG coordinates must be
# shifted by -1 before matching.
#
# Measured over 24,935 matched pairs the offset is (-0.990, -1.006) px, stable
# across all six cases (x: -0.987..-0.998, y: -1.001..-1.010).  Correcting by
# 1.0 px leaves a residual of (+0.010, -0.006) px -- the same ~0.005 px floor
# the network shows on synthetic data, where the convention matches by
# construction.  This replaced a per-case running estimate of mean(pred - gt),
# which recovered the same constant but fitted it to the evaluation data; every
# case's F1 is unchanged to 5e-5.  It is a coordinate convention, not a model
# bias, so it is applied as a constant and not re-estimated.
MICROSIG_XY_OFFSET = 1.0


# ----------------------------------------------------------------------------
@torch.no_grad()
def model_predict(model, img2d):
    """Per-sample z-score normalize, forward, return (C, H, W) on `device`."""
    x = (img2d - img2d.mean()) / img2d.std()
    return model(x[None, None].to(device))[0]


def pred_peaks_xyz(modelOut, thr):
    xy, p, z = get_peaks_z_nms(modelOut, thr=thr, apply_sigmoid=True)
    return xy, p, z * Z_SCALE


def gt_peaks_xyz(peaks):
    """Ground truth (xy [N, 2], z [N]) from a case's exact particle list.

    `peaks` is the (N, 4) [x, y, z, I] array written by generate_synthetic.py;
    z is the raw depth in [0, 1].
    """
    return peaks[:, :2], peaks[:, 2]


def RMSE(dx):
    return torch.sqrt(torch.sum(dx * dx) / dx.numel()) if dx.numel() else torch.tensor(float("nan"))


def MEDIAN(dx):
    """Median of |dx| — a robust central error, less sensitive to outliers than RMSE."""
    return torch.median(dx.abs()) if dx.numel() else torch.tensor(float("nan"))


def sweep_from_peaks(gt_xy, pr_xy, pr_prob, thresholds, dist_thresh=DIST_THRESH):
    """Threshold sweep from a SINGLE match per frame.

    3x3-NMS local maxima are threshold-independent, so the peaks above a higher
    threshold are exactly the probability-filtered subset of the candidates found at
    the lowest one.  Matching once and then filtering by probability therefore gives
    the same TP/FP/FN as matching per threshold, with one distance matrix per frame.

    Returns (tp, fp, fn, matches[(pr, gt)]); ``matches`` indexes the candidate
    arrays so the regression pass can filter by probability afterwards.
    """
    thr = np.asarray(thresholds, dtype=float)
    tp = np.zeros(len(thr)); fp = np.zeros(len(thr)); fn = np.full(len(thr), float(gt_xy.shape[0]))
    if pr_xy.shape[0] == 0 or gt_xy.shape[0] == 0:
        return tp, fp, fn, torch.empty((0, 2), dtype=torch.long)

    _, _, _, _, m = match_peaks(gt_xy, pr_xy, dist_thresh)
    matched = np.zeros(pr_xy.shape[0], dtype=bool)
    if m.shape[0]:
        matched[m[:, 0].cpu().numpy()] = True
    p = pr_prob.detach().cpu().numpy()
    for j, t in enumerate(thr):
        keep = p >= t
        tp[j] = np.count_nonzero(keep & matched)
        fp[j] = np.count_nonzero(keep & ~matched)
        fn[j] = gt_xy.shape[0] - tp[j]
    return tp, fp, fn, m


def evaluate(model):
    det_rows, reg_rows = [], []

    ppp_dirs = sorted(glob.glob(os.path.join(EVAL_DIR, "*ppp")))
    if not ppp_dirs:
        raise FileNotFoundError(f"No evaluation data under {EVAL_DIR} - run generate_synthetic.py first.")

    for ppp_dir in ppp_dirs:
        ppp = float(os.path.basename(ppp_dir)[:-3])
        for noise_sigma in EVAL_NOISE:
            folder = os.path.join(ppp_dir, f"{noise_sigma:.4f}Noise")
            if not os.path.isdir(folder):
                print(f"  [skip missing] {folder}")
                continue

            tp = np.zeros(len(THRESHOLDS)); fp = np.zeros_like(tp); fn = np.zeros_like(tp)
            thr0 = float(THRESHOLDS.min())

            # one match per frame; cache candidate peaks for the regression pass
            cache = []
            for n in range(EVAL_SAMPLES):
                f = os.path.join(folder, f"case{n:04d}.npz")
                if not os.path.exists(f):
                    continue
                data = np.load(f)
                img = torch.from_numpy(data['img']).float()
                gt_xy, gt_z = gt_peaks_xyz(torch.from_numpy(data["peaks"]).float().to(device))

                pred = model_predict(model, img)
                pr_xy, pr_prob, pr_z = pred_peaks_xyz(pred, thr0)
                tpj, fpj, fnj, m = sweep_from_peaks(gt_xy, pr_xy, pr_prob, THRESHOLDS)
                tp += tpj; fp += fpj; fn += fnj
                cache.append((pr_xy, pr_prob, pr_z, gt_xy, gt_z, m))

            if not cache:
                continue

            row = best_f1_row(tp, fp, fn, THRESHOLDS)
            row.update(noise=noise_sigma, ppp=ppp)
            det_rows.append(row)

            # localization: matched pairs whose pred prob >= best-F1 threshold
            best_thr = float(row["Thr"])
            dx_l, dy_l, dz_l = [], [], []
            for pr_xy, pr_prob, pr_z, gt_xy, gt_z, m in cache:
                if m.shape[0] == 0:
                    continue
                md = m.to(pr_prob.device)
                mm = md[pr_prob[md[:, 0]] >= best_thr]
                if mm.shape[0] == 0:
                    continue
                dx_l.append(pr_xy[mm[:, 0], 0] - gt_xy[mm[:, 1], 0])
                dy_l.append(pr_xy[mm[:, 0], 1] - gt_xy[mm[:, 1], 1])
                dz_l.append(pr_z[mm[:, 0]] - gt_z[mm[:, 1]])

            dx = torch.cat(dx_l) if dx_l else torch.empty(0)
            dy = torch.cat(dy_l) if dy_l else torch.empty(0)
            dz = torch.cat(dz_l) if dz_l else torch.empty(0)
            dxy = torch.sqrt(dx ** 2 + dy ** 2) if dx.numel() else torch.empty(0)  # lateral residual
            reg_rows.append(dict(noise=noise_sigma, ppp=ppp,
                                 # per-axis RMSE (kept for completeness)
                                 x=RMSE(dx).item(), y=RMSE(dy).item(), z=RMSE(dz).item(),
                                 # lateral (xy) and depth (z): RMSE + robust median error
                                 xy_rmse=RMSE(dxy).item(), xy_median=(torch.median(dxy).item() if dxy.numel() else float("nan")),
                                 z_rmse=RMSE(dz).item(), z_median=MEDIAN(dz).item(),
                                 n_matched=int(dx.numel())))
            print(f"  ppp={ppp:.3f} noise={noise_sigma:.4f} "
                  f"F1={row['F1']:.3f}@{best_thr:.2f}  "
                  f"xy rmse/med={reg_rows[-1]['xy_rmse']:.3f}/{reg_rows[-1]['xy_median']:.3f}  "
                  f"z rmse/med={reg_rows[-1]['z_rmse']:.4f}/{reg_rows[-1]['z_median']:.4f}")
            del cache
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return pd.DataFrame(det_rows), pd.DataFrame(reg_rows)


def load_tif(path):
    """Read a (possibly 16-bit) TIFF to a float32 ndarray. PIL's libtiff decoder
    can crash on the 16-bit MicroSIG frames, so prefer tifffile, then cv2, then PIL."""
    try:
        import tifffile
        return tifffile.imread(path).astype(np.float32)
    except Exception:
        pass
    try:
        import cv2
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if a is not None:
            return a.astype(np.float32)
    except Exception:
        pass
    from PIL import Image
    return np.array(Image.open(path)).astype(np.float32)


def microsig_eval(model, microsig_dir=MICROSIG_DIR):
    """Detection metrics on the MicroSIG frames (full 1024² frame).

    Same NMS + greedy matching as the synthetic eval.  MicroSIG's 1-based pixel
    coordinates are shifted onto this network's 0-based grid by the constant
    MICROSIG_XY_OFFSET before matching.  ``ppp = ppi / (H*W)`` puts each case on
    the synthetic sweep's density axis.  Returns an empty frame if the data
    directory is absent.
    """
    if not os.path.isdir(microsig_dir):
        print(f"  [microsig] {microsig_dir} not found - skipping MicroSIG eval.")
        return pd.DataFrame()

    rows = []
    for gt_txt in sorted(glob.glob(os.path.join(microsig_dir, "*.txt"))):
        case = os.path.splitext(os.path.basename(gt_txt))[0]
        case_dir = os.path.join(microsig_dir, case)
        if not os.path.isdir(case_dir):
            continue
        m = re.search(r"(\d+)\s*ppi", case)
        ppi = int(m.group(1)) if m else float("nan")

        caseGT = np.loadtxt(gt_txt, skiprows=1, delimiter=",")
        gt_frames = caseGT[:, 0].astype(np.int32)
        frames = sorted(set(gt_frames.tolist()))

        tp = np.zeros(len(THRESHOLDS)); fp = np.zeros_like(tp); fn = np.zeros_like(tp)
        area = None

        for frameN in frames:
            img_path = os.path.join(case_dir, f"B{frameN:05d}.tif")
            if not os.path.exists(img_path):
                continue
            IMG = load_tif(img_path)
            area = IMG.shape[0] * IMG.shape[1]
            img = torch.from_numpy(IMG).float()
            pred = model_predict(model, img)

            # MATLAB 1-based -> 0-based pixel centres
            gt_xy = torch.tensor(caseGT[gt_frames == frameN, 1:3],
                                 dtype=torch.float32, device=device) - MICROSIG_XY_OFFSET

            # 3x3-NMS maxima are threshold-independent, so match once and read
            # off the sweep by probability filtering (as in sweep_from_peaks).
            pr, prob, _ = pred_peaks_xyz(pred, float(THRESHOLDS.min()))
            tpj, fpj, fnj, _ = sweep_from_peaks(gt_xy, pr, prob, THRESHOLDS)
            tp += tpj; fp += fpj; fn += fnj

        if area is None:
            continue
        row = best_f1_row(tp, fp, fn, THRESHOLDS)
        row.update(case=case, ppi=ppi, ppp=(ppi / area if ppi == ppi else float("nan")))
        rows.append(row)
        print(f"  [microsig] {case}: F1={row['F1']:.3f}@{row['Thr']:.2f} "
              f"P={row['Precision']:.3f} R={row['Recall']:.3f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="model checkpoint path")
    ap.add_argument("--microsig-dir", default=MICROSIG_DIR,
                    help="MicroSIG benchmark directory (default: $MICROSIG_DIR "
                         "or data/microsig); skipped if absent")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = PeakCNN_UNet_4level_ConvNeXt.load_checkpoint(args.ckpt, device=device).eval()
    print(f"Model: {args.ckpt}")
    det_df, reg_df = evaluate(model)
    det_df.to_csv(os.path.join(RESULTS_DIR, "detection_metrics.csv"), index=False)
    reg_df.to_csv(os.path.join(RESULTS_DIR, "localization_metrics.csv"), index=False)

    print("\n=== MicroSIG eval ===")
    sig_df = microsig_eval(model, args.microsig_dir)
    if not sig_df.empty:
        sig_df.to_csv(os.path.join(RESULTS_DIR, "microsig_metrics.csv"), index=False)

    print(f"\nWrote metrics CSVs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
