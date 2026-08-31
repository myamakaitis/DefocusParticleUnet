"""Peak extraction, matching, and detection metrics.

Model output / ground-truth convention (C, H, W):
    ch 0    : peak probability (logits for model output, probability for GT)
    ch 1, 2 : sub-pixel offsets (dp, dq) added to the integer pixel location
    ch 3    : depth z
Peak coordinates are returned as [N, 2] arrays of (p, q) = (x, y).
"""
import numpy as np
import torch
import torch.nn.functional as F


# ── peak extraction ──────────────────────────────────────────────────────────

def get_peaks(modelOut: torch.Tensor, thr: float = 0.5, apply_sigmoid: bool = True) -> torch.Tensor:
    """Extract sub-pixel peak coordinates by simple thresholding of ch 0."""
    prob = torch.sigmoid(modelOut[0]) if apply_sigmoid else modelOut[0]
    detected = prob > thr

    q_int, p_int = torch.nonzero(detected, as_tuple=True)
    q, p = q_int.float(), p_int.float()
    p += modelOut[1, q_int, p_int]
    q += modelOut[2, q_int, p_int]
    return torch.stack((p, q), 1)


def get_peaks_nms(modelOut: torch.Tensor, thr: float = 0.7, apply_sigmoid: bool = True) -> torch.Tensor:
    """Extract sub-pixel peak coordinates with 3×3 non-maximum suppression."""
    prob = torch.sigmoid(modelOut[0]) if apply_sigmoid else modelOut[0]
    local_max = F.max_pool2d(prob[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    detected = (prob > thr) & (prob == local_max)

    q_int, p_int = torch.nonzero(detected, as_tuple=True)
    q, p = q_int.float(), p_int.float()
    p += modelOut[1, q_int, p_int]
    q += modelOut[2, q_int, p_int]
    return torch.stack((p, q), 1)


def get_peaks_z_nms(modelOut: torch.Tensor, thr: float = 0.7, apply_sigmoid: bool = True):
    """As get_peaks_nms, additionally returning the peak probability and depth.

    Returns (coords [N, 2], probability [N], z [N]).
    """
    prob = torch.sigmoid(modelOut[0]) if apply_sigmoid else modelOut[0]
    local_max = F.max_pool2d(prob[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    detected = (prob > thr) & (prob == local_max)

    q_int, p_int = torch.nonzero(detected, as_tuple=True)
    q, p = q_int.float(), p_int.float()
    p += modelOut[1, q_int, p_int]
    q += modelOut[2, q_int, p_int]

    Pr = prob[q_int, p_int]
    z = modelOut[3, q_int, p_int]
    return torch.stack((p, q), 1), Pr, z


# ── matching ─────────────────────────────────────────────────────────────────

def match_peaks(peaks_gt: torch.Tensor, peaks_pred: torch.Tensor, dist_thresh: float = 1.0):
    """Greedy nearest-first matching of predicted peaks to ground truth.

    The distance matrix stays on the inputs' device; only within-threshold candidate
    pairs move to CPU for the greedy dedup.

    Returns (tp, fp, fn, bias, matches):
        tp, fp, fn : detection counts
        bias       : [2] mean (pred - gt) offset over matched pairs
        matches    : [tp, 2] long tensor of (pred_idx, gt_idx) pairs
    """
    npred, ngt = peaks_pred.shape[0], peaks_gt.shape[0]
    if npred == 0 or ngt == 0:
        return 0, npred, ngt, torch.zeros(2), torch.empty((0, 2), dtype=torch.long)

    dxdy = peaks_pred[:, None, :] - peaks_gt[None, :, :]
    d = torch.linalg.vector_norm(dxdy, dim=2)

    row, col = torch.nonzero(d <= dist_thresh, as_tuple=True)
    order = torch.argsort(d[row, col])            # closest pairs first
    row, col = row[order], col[order]
    dxdy_c = dxdy[row, col].cpu()
    row = row.cpu().numpy(); col = col.cpu().numpy()

    matched_gt, matched_pr, matches = set(), set(), []
    bias = torch.zeros(2)
    tp = 0
    for k in range(row.shape[0]):
        pr, g = int(row[k]), int(col[k])
        if g not in matched_gt and pr not in matched_pr:
            matched_gt.add(g)
            matched_pr.add(pr)
            matches.append((pr, g))
            bias += dxdy_c[k]
            tp += 1

    fp = npred - tp
    fn = ngt - tp
    bias = bias / tp if tp else torch.zeros(2)
    return tp, fp, fn, bias, torch.tensor(matches, dtype=torch.long).reshape(-1, 2)


# ── threshold-sweep evaluation ───────────────────────────────────────────────

def best_f1_row(tp, fp, fn, thresholds):
    """Select the threshold with the best F1 from accumulated count arrays."""
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    k = int(np.argmax(f1))
    return dict(TP=tp[k], FP=fp[k], FN=fn[k], Precision=prec[k], Recall=rec[k],
                F1=f1[k], Thr=thresholds[k])


def eval_classification_nms(gt: torch.Tensor, pred: torch.Tensor,
                            thresholds=np.linspace(0.1, 0.9, 17),
                            dist_thresh: float = 1.0):
    """Batch threshold-sweep of detection metrics using NMS peak extraction.

    gt, pred: [B, C, H, W].  GT ch 0 is a probability map (no sigmoid applied);
    prediction ch 0 is logits (sigmoid applied).

    Returns (best_f1, best_thr, best_precision, best_recall).
    """
    tp = np.zeros(len(thresholds)); fp = np.zeros_like(tp); fn = np.zeros_like(tp)

    with torch.no_grad():
        for n in range(gt.shape[0]):
            gt_xy = get_peaks_nms(gt[n], apply_sigmoid=False)
            for j, th in enumerate(thresholds):
                pr_xy = get_peaks_nms(pred[n], float(th))
                t, f_, n_, _, _ = match_peaks(gt_xy, pr_xy, dist_thresh)
                tp[j] += t; fp[j] += f_; fn[j] += n_

    row = best_f1_row(tp, fp, fn, thresholds)
    print(f"Best F1: {row['F1']:.3f} | w/ precision:{row['Precision']:.3f} "
          f"| w/ recall: {row['Recall']:.3f} | @ threshold: {row['Thr']:.2f}")
    return row['F1'], row['Thr'], row['Precision'], row['Recall']
