"""Publication figures.

Reproduces every figure from the repository contents:

    fig_calstack.gif      — animated z-sweep through the calibration stack
    fig_density_range.eps — 4 densities, low-noise top / high-noise bottom
    fig_input_patch.eps  — example network input (grayscale, pixel-crisp)
    fig_labels.eps       — ground-truth label channels with a detail view
    fig_qualitative.eps  — detections vs ground truth at 3 densities
    fig_detection.eps    — P / R / F1 vs density, low-/high-noise rows   (needs evaluate.py)
    fig_localization.eps — lateral (xy) & depth (z) error vs density,
                            RMSE + median, both noise levels              (needs evaluate.py)

Usage:
    python make_figures.py               # all figures
    python make_figures.py --fig density_range,labels
"""
import os
import glob
import shutil
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

import generate_synthetic as gen
import evaluate as ev
from pcnn import PeakCNN_UNet_4level_ConvNeXt

# ----------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR  = os.path.join(REPO_ROOT, "results", "figures")

# Use LaTeX text rendering when available; otherwise fall back to mathtext.
_USETEX = shutil.which("latex") is not None
# mathtext.fontset = "stix" -> proper serif math-italic (Times-like) for the few
# labels rendered without usetex (the multi-line fig_labels titles); avoids both
# the sans-serif mathtext default and the stiff Computer-Modern look.
plt.rcParams.update({"text.usetex": _USETEX, "font.family": "serif", "font.size": 12,
                     "mathtext.fontset": "stix"})
PCT = r"\%" if _USETEX else r"%"

DENS       = [0.002, 0.005, 0.01, 0.02]        # densities for the image panels
LN, HN     = gen.NOISE_LOW, gen.NOISE_HIGH
SIGMA_ADD  = float(np.sqrt(HN ** 2 - LN ** 2)) # extra noise: LN image → HN image
SEED       = 1234
CPU        = torch.device("cpu")
TRAIN_DENS = list(gen.TRAIN_PPPS.numpy())      # vlines marking training densities

XLABEL_FS = 17
TICK_FS   = 14


def render_patch(tile, n_peaks, device, seed):
    """Matched low-/high-noise patch with shared particle layout + 5-ch GT.

    Returns (img_ln, img_hn, gt, peaks); `peaks` is the exact (N, 4)
    [x, y, z, I] particle list, which — unlike `gt` — keeps close pairs.
    """
    torch.manual_seed(seed)
    psf = gen.build_psf_interpolator(device)
    img_ln, gt, peaks = gen.GenSampledPSFTile(psf, tile_size=tile, n_peaks=int(n_peaks),
                                              noise_sigma=LN, device=device, n_peak_std=0.0)
    img_hn = img_ln + SIGMA_ADD * torch.randn_like(img_ln)
    return img_ln, img_hn, gt, peaks


def _split_show(ax, img_ln, img_hn, vmax):
    """Top half = low noise, bottom half = high noise."""
    ax.imshow(img_ln, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    hn = np.ma.masked_array(img_hn, mask=np.zeros(img_hn.shape, bool))
    hn.mask[: img_hn.shape[0] // 2, :] = True
    ax.imshow(hn, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")


# ----------------------------------------------------------------------------
# Calibration z-stack GIF
# ----------------------------------------------------------------------------
def fig_calstack():
    stack = np.zeros((gen.NZ, gen.PSF_TILE_SIZE, gen.PSF_TILE_SIZE), np.float32)
    for i in range(gen.NZ):
        stack[i] = np.asarray(Image.open(os.path.join(gen.PSF_DIR, f"B{i + 1:05d}.tif")))
    stack /= stack.max()
    c, half = gen.PSF_CENTER, 64
    crop = stack[:, c - half:c + half, c - half:c + half]
    z = np.linspace(0, 1, gen.NZ)

    frames = []
    for i in range(gen.NZ):
        fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
        ax.imshow(crop[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(rf"$z = {z[i]:.2f}$")
        ax.set_xticks(()); ax.set_yticks(())
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)

    out = os.path.join(FIGS_DIR, "fig_calstack.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print("wrote", out)


# ----------------------------------------------------------------------------
# Density-range panel (1 × 4)
# ----------------------------------------------------------------------------
def fig_density_range():
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), dpi=600)
    for i, (ax, ppp) in enumerate(zip(axes, DENS)):
        n = round(ppp * 128 * 128)
        img_ln, img_hn, _, _ = render_patch(128, n, CPU, SEED + i)
        ln, hn = img_ln.cpu().numpy(), img_hn.cpu().numpy()
        _split_show(ax, ln, hn, float(np.quantile(ln, 0.999)))
        ax.set_title(rf"$N_i = {ppp:g}$")
        ax.set_xticks(()); ax.set_yticks(())
    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "fig_density_range.eps")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------------
# Example input patch + label channels (with a detail-view zoom)
# ----------------------------------------------------------------------------
def fig_labels():
    # Slightly denser patch so a small closeup window can hold ~5 particles.
    N_PEAKS = 64
    img_ln, _, gt, peaks = render_patch(128, N_PEAKS, CPU, SEED)
    img = img_ln.cpu().numpy()
    g = gt.cpu().numpy()
    prob, dp, dq, z = g[0], g[1], g[2], g[3]
    H, W = prob.shape
    vmax_img = float(np.quantile(img, 0.999))

    # standalone input patch (pixel-crisp) — also used as row 0 below
    fig, ax = plt.subplots(figsize=(4, 4), dpi=256)
    ax.imshow(img, cmap="gray", vmin=0, vmax=vmax_img, interpolation="nearest")
    ax.set_xticks(()); ax.set_yticks(()); ax.set_frame_on(False)
    out1 = os.path.join(FIGS_DIR, "fig_input_patch.eps")
    fig.savefig(out1, bbox_inches="tight", pad_inches=0); plt.close(fig)
    print("wrote", out1)

    # --- pick a BS×BS closeup window holding ~TARGET particles, kept a margin
    #     (PAD) inside the frame for visual padding and biased toward the
    #     center-right of the patch. ---
    BS, TARGET, PAD = 20, 5, 14
    cx_pref, cy_pref = 0.60 * W, 0.50 * H          # preferred window centre (center-right)
    pk_xy, _ = ev.gt_peaks_xyz(peaks)
    px, py = pk_xy[:, 0].cpu().numpy(), pk_xy[:, 1].cpu().numpy()
    cands = []
    for by0 in range(PAD, H - BS - PAD + 1):
        for bx0 in range(PAD, W - BS - PAD + 1):
            c = int(np.count_nonzero((px >= bx0) & (px < bx0 + BS) & (py >= by0) & (py < by0 + BS)))
            cands.append((c, bx0, by0))
    pool = [t for t in cands if abs(t[0] - TARGET) <= 1] or cands
    ncap, bx0, by0 = min(pool, key=lambda t: (abs(t[0] - TARGET),
                                              (t[1] + BS / 2 - cx_pref) ** 2 + (t[2] + BS / 2 - cy_pref) ** 2))
    bx1, by1 = bx0 + BS, by0 + BS
    print(f"closeup {BS}px window at ({bx0},{by0}) holds {ncap} particles")

    THR, ACCENT, ACCENT2 = 0.15, "red", "deepskyblue"
    mask = prob > THR
    ys, xs = np.nonzero(mask)
    zcmap = plt.get_cmap("viridis").copy(); zcmap.set_bad("white")
    zm = np.ma.masked_array(z, mask=~mask)

    # The offsets row (ZOOM_ROW) zooms its closeup further
    from itertools import combinations
    ZOOM_ROW = 2


    cx3, cy3, SS = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0, 4
    cy3 -= 5
    cx3 += 0.5
    sx0, sy0 = cx3 - SS / 5.0, cy3 - SS / 5.0
    print(f"row {ZOOM_ROW} closeup zoomed to {SS}px on 3-cluster at ({cx3:.1f}, {cy3:.1f})")

    def _row_window(r):
        return (sx0, sy0, SS, ACCENT2) if r == ZOOM_ROW else (bx0, by0, BS, ACCENT)

    row_titles = ("Grayscale Image\n[Input]",
                  "$p$\n[Channel 0]",
                  r"$(\Delta x, \Delta y)$" + "\n[Channels 1, 2]",   # raw math + real newline
                  "$z$\n[Channel 3]")
    letters = ("a)", "b)", "c)", "d)")

    # single column of a two-column paper: narrow, tall (4 rows x full/detail).
    fig, ax = plt.subplots(4, 2, figsize=(3.4, 6.4), dpi=300)

    def _cbar(a, mappable, show):
        cax = make_axes_locatable(a).append_axes("right", size="6%", pad=0.04)
        if show:
            cb = fig.colorbar(mappable, cax=cax); cb.ax.tick_params(labelsize=6)
        else:
            cax.axis("off")          # keep detail axes the same width across rows

    for r in range(4):
        full, det = ax[r, 0], ax[r, 1]
        wx0, wy0, ws, wc = _row_window(r)
        m = None
        for a in (full, det):
            if r == 0:
                a.imshow(img, cmap="gray", vmin=0, vmax=vmax_img, interpolation="nearest", extent=[0, W, H, 0])
            elif r == 1:
                m = a.imshow(prob, cmap="magma", vmin=0, vmax=1, interpolation="nearest", extent=[0, W, H, 0])
            elif r == 2:
                a.set_facecolor("white")
                a.quiver(xs, ys, dp[mask], dq[mask], angles="xy", scale_units="xy", scale=1,
                         color="k", width=0.01)
            else:
                m = a.imshow(zm, cmap=zcmap, vmin=0, vmax=1, interpolation="nearest", extent=[0, W, H, 0])
            a.set_aspect("equal"); a.set_xticks(()); a.set_yticks(())
        full.set_xlim(0, W); full.set_ylim(H, 0)
        full.add_patch(Rectangle((wx0, wy0), ws, ws, fill=False, edgecolor=wc, linewidth=1.2))
        # mathtext (usetex=False) for the two-line titles — MiKTeX here chokes on
        # multi-line usetex strings; mathtext renders $...$ + newlines reliably.
        full.set_ylabel(row_titles[r], fontsize=9, rotation=90, labelpad=14, va="center", usetex=False)
        full.text(0.05, 0.95, letters[r], transform=full.transAxes, va="top", ha="left",
                  fontsize=14, fontweight="bold",
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))
        det.set_xlim(wx0, wx0 + ws); det.set_ylim(wy0 + ws, wy0)
        for s in det.spines.values():
            s.set_edgecolor(wc); s.set_linewidth(1.4); s.set_visible(True)
        _cbar(det, m, show=(r in (1, 3)))

    fig.tight_layout(h_pad=0.15)
    for r in range(4):
        wx0, wy0, ws, wc = _row_window(r)
        for yc in (wy0, wy0 + ws):
            fig.add_artist(ConnectionPatch(xyA=(wx0 + ws, yc), coordsA=ax[r, 0].transData,
                                           xyB=(wx0, yc), coordsB=ax[r, 1].transData,
                                           color=wc, linewidth=0.7, zorder=5))

    out2 = os.path.join(FIGS_DIR, "fig_labels.eps")
    fig.savefig(out2, bbox_inches="tight"); plt.close(fig)
    print("wrote", out2)


# ----------------------------------------------------------------------------
# Qualitative results (256² inference, central 128² display — edge-effect free)
# ----------------------------------------------------------------------------
def _nearest_thr(det, ppp, ns):
    s = det[np.isclose(det.noise, ns)]
    i = (s.ppp - ppp).abs().idxmin()
    return float(s.loc[i, "Thr"])


def fig_qualitative():
    det = pd.read_csv(os.path.join(ev.RESULTS_DIR, "detection_metrics.csv"))
    model = PeakCNN_UNet_4level_ConvNeXt.load_checkpoint(ev.DEFAULT_CKPT, device=ev.device).to(ev.device).eval()
    lo, hi = 64, 192
    dens_q = [0.002, 0.008, 0.02]                       # 3 patches: low / medium / high density

    def central(xy):
        m = (xy[:, 0] >= lo) & (xy[:, 0] < hi) & (xy[:, 1] >= lo) & (xy[:, 1] < hi)
        return xy[m] - lo

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), dpi=600)
    for i, (ax, ppp) in enumerate(zip(axes, dens_q)):
        n = round(ppp * 256 * 256)
        img_ln, img_hn, _, peaks = render_patch(256, n, ev.device, SEED + 100 + i)

        gt_xy, _ = ev.gt_peaks_xyz(peaks.to(ev.device))
        pr_ln, _, _ = ev.pred_peaks_xyz(ev.model_predict(model, img_ln), _nearest_thr(det, ppp, LN))
        pr_hn, _, _ = ev.pred_peaks_xyz(ev.model_predict(model, img_hn), _nearest_thr(det, ppp, HN))

        gtc = central(gt_xy.cpu()); plc = central(pr_ln.cpu()); phc = central(pr_hn.cpu())
        ln = img_ln.cpu().numpy()[lo:hi, lo:hi]
        hn = img_hn.cpu().numpy()[lo:hi, lo:hi]
        _split_show(ax, ln, hn, float(np.quantile(ln, 0.999)))

        # ground truth (cyan circles) and predictions (red x); top half = low
        # noise, bottom half = high noise.
        ax.scatter(gtc[:, 0], gtc[:, 1], s=33, marker="o", facecolors="cyan",
                   edgecolors="black", linewidths=0.3, zorder=3)
        top = plc[plc[:, 1] < 64]; bot = phc[phc[:, 1] >= 64]
        ax.scatter(top[:, 0], top[:, 1], s=44, color="orangered", marker="x", linewidth=1.4, zorder=4)
        ax.scatter(bot[:, 0], bot[:, 1], s=44, color="orangered", marker="x", linewidth=1.4, zorder=4)

        ax.axhline(64, color="white", linewidth=1.5, zorder=5)    # low- / high-noise divider

        ax.set_title(rf"$N_i = {ppp:g}$")
        ax.set_xticks(()); ax.set_yticks(()); ax.set_frame_on(False)
        ax.set_xlim(0, 128); ax.set_ylim(128, 0)
    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "fig_qualitative.eps")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------------
# Detection + localization metrics vs density
# ----------------------------------------------------------------------------
# Compact single-column journal figures: 2 rows x 1 col.
NOISE_TITLES = {
    "LN": r"low noise ($\sigma_w = 2.5\mathrm{e}{-}3$)",
    "HN": r"high noise ($\sigma_w = 6.25\mathrm{e}{-}2$)",
}
LABEL_FS, TK_FS, LEG_FS = 11, 9, 8


# density-axis ticks: decades + their 3× multiples (labelled).  Detection spans
# down to the mu-SIG range (3e-4); localization only has synthetic data (>= 1e-3),
# so it drops the 3e-4 tick.
XTICKS  = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
XLABELS = [r"$3{\times}10^{-4}$", r"$10^{-3}$", r"$3{\times}10^{-3}$",
           r"$10^{-2}$", r"$3{\times}10^{-2}$"]
XTICKS_LOC, XLABELS_LOC = XTICKS[1:], XLABELS[1:]


def _density_xaxis(a, xlim, ticks=XTICKS, labels=XLABELS):
    a.set_xscale("log"); a.set_xlim(*xlim)
    a.set_xticks(ticks); a.set_xticklabels(labels)
    a.set_xticks([], minor=True)
    a.tick_params(labelsize=TK_FS)


def fig_metrics():
    det = pd.read_csv(os.path.join(ev.RESULTS_DIR, "detection_metrics.csv"))
    reg = pd.read_csv(os.path.join(ev.RESULTS_DIR, "localization_metrics.csv"))
    sig_path = os.path.join(ev.RESULTS_DIR, "microsig_metrics.csv")
    sig = pd.read_csv(sig_path).sort_values("ppp") if os.path.exists(sig_path) else None

    # ---- detection: rows = noise level; P / R / F1 together on one axis ------
    #      metric -> linestyle; dataset -> colour (extended = navy, mu-SIG = orange).
    #      "extended" = single-particle template repeated; mu-SIG = ray-traced.
    #      The MicroSIG reference is overlaid on the high-noise panel only.
    #      Panels share the x-axis, so the low-noise panel simply has no data
    #      left of 1e-3 (where only the mu-SIG points sit).
    metric_styles = [("F1", "-"), ("Precision", "--"), ("Recall", ":")]
    
    noise_colors = {"LN": "navy", "HN":"darkorange"}
    noise_marker = {"LN": "o", "HN":"s"}
    
    fig, ax = plt.subplots(2, 1, figsize=(3.6, 5.0), dpi=600, sharex=True)
    for a, (tag, ns) in zip(ax, (("HN", HN), ("LN", LN))):
        a.vlines(TRAIN_DENS, 0, 2, color="#52C9E8", linewidth=2.2, zorder=-1)
        s = det[np.isclose(det.noise, ns)].sort_values("ppp")
        for metric, ls in metric_styles:
            a.plot(s.ppp, s[metric], ms=4, ls=ls, color=noise_colors[tag], marker=noise_marker[tag])
            if sig is not None and tag == "HN":
                a.plot(sig.ppp, sig[metric], marker="^", ms=4, ls=ls, color="limegreen")
        _density_xaxis(a, (1.5e-4, 0.06)); a.set_ylim(0.4, 1.01); a.grid(True)
        a.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        a.set_ylabel("score", fontsize=LABEL_FS)
        #a.set_title(NOISE_TITLES[tag], fontsize=LABEL_FS)
    ax[1].set_xlabel(r"$N_i$ [ppp]", fontsize=LABEL_FS)
    handles = [
        Line2D([0], [0], color="k", ls="--", label=r"$P$"),
		Line2D([0], [0], color="k", ls="-",  label=r"$F_1$"),
        Line2D([0], [0], color="k", ls=":",  label=r"$R$"),
        Line2D([0], [0], color="limegreen", marker='^', ls="-", label=r"$\mu$SIG"),
        Line2D([0], [0], color="darkorange", marker='s', ls="-", label=r"$\sigma_w{=}6.25\mathrm{e}{-}2$"),
        Line2D([0], [0], color="navy", marker='o', ls="-", label=r"$\sigma_w{=}2.5\mathrm{e}{-}3$"),
    ]
    ax[1].legend(handles=handles, fontsize=LEG_FS, ncol=2, loc="lower left",
                 columnspacing=1.0, handlelength=1.6)
    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "fig_detection.eps")
    fig.savefig(out); plt.close(fig); print("wrote", out)

    # ---- localization: row 0 = lateral (xy), row 1 = depth (z) --------------
    #      both noise levels; RMSE (solid) and median (dashed).
    noises = [(LN, "LN"), (HN, "HN")]

    fig, ax = plt.subplots(2, 1, figsize=(3.6, 5.0), dpi=600, sharex=True)
    for ns, c in noises:
        s = reg[np.isclose(reg.noise, ns)].sort_values("ppp")
        ax[0].plot(s.ppp, s.xy_rmse,       "-",  ms=3, color=noise_colors[c], marker=noise_marker[c])
        ax[0].plot(s.ppp, s.xy_median,     "--", ms=3, color=noise_colors[c], marker=noise_marker[c])
        ax[1].plot(s.ppp, 100.0 * s.z_rmse,   "-",  ms=3, color=noise_colors[c], marker=noise_marker[c])
        ax[1].plot(s.ppp, 100.0 * s.z_median, "--", ms=3, color=noise_colors[c], marker=noise_marker[c])
    ax[0].set_ylabel(r"$\varepsilon_{xy}$ [px]", fontsize=LABEL_FS)
    ax[0].set_ylim(0, 0.5); ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax[1].set_ylabel(r"$\varepsilon_z$ [" + PCT + "]", fontsize=LABEL_FS)
    ax[1].set_ylim(0, 12.5); ax[1].set_yticks([0, 2.5, 5, 7.5, 10, 12.5])
    for a in ax:
        a.vlines(TRAIN_DENS, 0, 50, color="#52C9E8", linewidth=2.2, zorder=0)
        _density_xaxis(a, (0.0009, 0.055), XTICKS_LOC, XLABELS_LOC); a.grid(True)
    ax[1].set_xlabel(r"$N_i$ [ppp]", fontsize=LABEL_FS)
    ax[1].legend(handles=[
        Line2D([0], [0], marker='s', ms=3, color="darkorange",     ls="-", label=r"$\sigma_w{=}6.25\mathrm{e}{-}2$"),
        Line2D([0], [0], marker='o', ms=3, color="navy", ls="-", label=r"$\sigma_w{=}2.5\mathrm{e}{-}3$"),
        
        Line2D([0], [0], color="k",         ls="-", label="RMSE"),
        Line2D([0], [0], color="k",         ls="--", label="median"),
    ], fontsize=LEG_FS, ncol=2, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "fig_localization.eps")
    fig.savefig(out); plt.close(fig); print("wrote", out)


FIGS = {
    "calstack": fig_calstack,
    "density_range": fig_density_range,
    "labels": fig_labels,
    "qualitative": fig_qualitative,
    "metrics": fig_metrics,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default="all", help=f"comma list of {list(FIGS)} or 'all'")
    args = ap.parse_args()
    os.makedirs(FIGS_DIR, exist_ok=True)
    which = list(FIGS) if args.fig == "all" else args.fig.split(",")
    for name in which:
        print(f"\n=== {name} ===")
        FIGS[name]()
    print("\nDone ->", FIGS_DIR)
