"""Training script for the peak-detection / localization CNN.

Loads the full training set into RAM (per-sample z-score normalization), then
trains N_NETWORKS independently initialized PeakCNN_UNet_4level_ConvNeXt models
with a focal classification loss plus soft-mask-weighted regression losses,
AdamW, cosine-annealed learning rate, and gradient clipping.  Checkpoints
(with init_args for later reconstruction) are written to
network/runs/<timestamp>/.

Usage:
    python train.py
"""
import os
from itertools import product
from time import perf_counter
from datetime import datetime as dt
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed

from pcnn import PeakCNN_UNet_4level_ConvNeXt, eval_classification_nms

# ----------------------------------------------------------------------------
# Paths / configuration
# ----------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(REPO_ROOT, "data", "synthetic", "train")
SAVE_ROOT = os.path.join(REPO_ROOT, "network", "runs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# number of independent networks to train (fresh init each; dataset loaded once)
N_NETWORKS = 1

# image / model geometry (must match generate_synthetic.TRAIN_TILE_SIZE).
# The network is fully convolutional, so inference may run at any size
# divisible by 16 (evaluation uses 512² tiles).
IMG_H = 256
IMG_W = 256
NCH   = 32     # matches the published network net00 (N_mid=32, ~140k parameters)
N_OUT = 5

# Depth is generated in [0, 1]; no rescaling needed.
Z_SCALE = 1.0

# dataset grid (must match generate_synthetic TRAIN_* settings).
# 3 ppp × 2 noise × 5000 = 30,000 images at 256², all held in RAM
# (~47 GB with the 5-channel GT).
TRAIN_PPPS  = np.array([0.001, 0.005, 0.025])
TRAIN_NOISE = (2.5e-3, 6.25e-2)
N_CASES     = 5000            # samples per (ppp, noise) folder
CASE_OFFSET = 0

# training schedule
N_EPOCH    = 100
BATCH_SIZE = 16
CheckPoint = 10

LR_MAX = 3e-3
LR_MIN = 1e-5

# loss
FOCAL_ALPHA = 0.5
FOCAL_GAMMA = 2.0
lambda_cls  = 1_000.0
lambda_spx  = 10.0
lambda_z    = 50.0
lambda_I    = 0.0        # amplitude (ch 4) is UNUSED: zero weight -> head not learned


# ----------------------------------------------------------------------------
# In-memory dataset
# ----------------------------------------------------------------------------
class CPUMemDataset(Dataset):
    """Loads every case####.npz into RAM once, with per-sample z-score norm."""

    def __init__(self, folders, n_cases, img_h, img_w, gt_depth, offset):
        self.folders  = folders
        self.n_cases  = n_cases
        self.offset   = offset
        self.size     = len(folders) * n_cases
        self.img_h    = img_h
        self.img_w    = img_w
        self.gt_depth = gt_depth

        self.imgs = torch.empty((self.size, 1,        img_h, img_w), dtype=torch.float32)
        self.gts  = torch.empty((self.size, gt_depth, img_h, img_w), dtype=torch.float32)

        files = [(i, folder, n + self.offset)
                 for i, (folder, n) in enumerate(product(folders, range(n_cases)))]

        Parallel(n_jobs=-1, require="sharedmem", backend="threading", verbose=0)(
            delayed(self.LoadOneNPZ)(i, folder, n) for i, folder, n in files)

    def LoadOneNPZ(self, i, folder, n):
        case = np.load(os.path.join(folder, f"case{n:04d}.npz"))

        img = torch.from_numpy(case['img'].reshape(1, self.img_h, self.img_w)).float()
        img = (img - img.mean()) / img.std()
        self.imgs[i] = img

        gt = torch.from_numpy(case['gt']).float()
        gt[3] = gt[3] / Z_SCALE
        self.gts[i] = gt

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.imgs[index], self.gts[index]

    def getDataLoader(self, batch_size, num_workers=None, **kwargs):
        if num_workers is None:
            num_workers = cpu_count()
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, **kwargs)


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------
def focal_loss(logits, targets, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    p       = torch.sigmoid(logits)
    ce      = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t     = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()


def peak_cnn_loss(gt, pred, lambda_cls, lambda_spx, lambda_z, lambda_I):
    p_hat,  du_hat,  dv_hat,  z_hat,  I_hat  = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]
    p_true, du_true, dv_true, z_true, I_true = gt[:, 0],   gt[:, 1],   gt[:, 2],   gt[:, 3],   gt[:, 4]

    loss_cls = focal_loss(p_hat, p_true)
    w        = p_true                     # weight by the soft Gaussian mask
    w_sum    = w.sum().clamp(min=1)

    loss_spx = ((w * (du_hat - du_true) ** 2).sum() + (w * (dv_hat - dv_true) ** 2).sum()) / w_sum
    loss_z   = (w * (z_hat - z_true) ** 2).sum() / w_sum
    loss_I   = (w * (I_hat - I_true) ** 2).sum() / w_sum   # ch 4 unused: lambda_I = 0

    loss = lambda_cls * loss_cls + lambda_spx * loss_spx + lambda_z * loss_z + lambda_I * loss_I
    return loss, (loss_cls, loss_spx, loss_z, loss_I)


def write_params(folder, now):
    with open(os.path.join(folder, f"params_{now}.txt"), 'w') as f:
        f.write(f"N_NETWORKS    = {N_NETWORKS}\n")
        f.write(f"IMG_H/W       = {IMG_H}x{IMG_W}\n")
        f.write(f"NCH           = {NCH}\n")
        f.write(f"N_OUT         = {N_OUT}\n")
        f.write(f"Z_SCALE       = {Z_SCALE}\n")
        f.write(f"N_EPOCH       = {N_EPOCH}\n")
        f.write(f"BATCH_SIZE    = {BATCH_SIZE}\n")
        f.write(f"TRAIN_PPPS    = {list(TRAIN_PPPS)}\n")
        f.write(f"TRAIN_NOISE   = {TRAIN_NOISE}\n")
        f.write(f"N_CASES       = {N_CASES}\n")
        f.write(f"LR_MAX/MIN    = {LR_MAX}/{LR_MIN}\n")
        f.write(f"scheduler     = CosineAnnealingLR (no restarts)\n")
        f.write(f"FOCAL a/g     = {FOCAL_ALPHA}/{FOCAL_GAMMA}\n")
        f.write(f"lambda c/s/z/I= {lambda_cls}/{lambda_spx}/{lambda_z}/{lambda_I}\n")


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
def train_one_network(net_idx, data_loader, save_folder, t0):
    model     = PeakCNN_UNet_4level_ConvNeXt(N_in=1, N_mid=NCH, N_out=N_OUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCH, eta_min=LR_MIN
    )

    for i in range(0, N_EPOCH + 1):
        print(f"\n[net {net_idx:02d}] Epoch: {i:06d}")

        c_total = 0.0
        tally = dict(total=0.0, cls=0.0, spx=0.0, z=0.0, I=0.0)
        grad_norm = 0.0

        for img, gt in data_loader:
            img, gt = img.to(device), gt.to(device)

            pred = model(img)
            loss, loss_ind = peak_cnn_loss(gt, pred,
                                           lambda_cls=lambda_cls, lambda_spx=lambda_spx,
                                           lambda_z=lambda_z, lambda_I=lambda_I)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            c_total += 1.0
            tally['total'] += loss.item()
            tally['cls']   += loss_ind[0].item()
            tally['spx']   += loss_ind[1].item()
            tally['z']     += loss_ind[2].item()
            tally['I']     += loss_ind[3].item()

        print("Epoch Mean")
        print(f"Loss: {tally['total']/c_total:.4f} | Cls: {tally['cls']/c_total:.5f} | "
              f"SubPx: {tally['spx']/c_total:.3f} | Depth: {tally['z']/c_total:.3f} | "
              f"Intensity: {tally['I']/c_total:.3f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"grad norm: {grad_norm:.3f}")
        print(f"{(perf_counter() - t0)/60:.2f} min")

        f1, _, _, _ = eval_classification_nms(gt, pred)

        scheduler.step()

        if (i % CheckPoint == 0 and i != 0) or i == N_EPOCH:
            model.save_checkpoint(
                os.path.join(save_folder, f"PCNN_net{net_idx:02d}_Epoch{i:05d}_f1{f1:.3f}.pth")
            )

    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)

    data_folders = [
        os.path.join(TRAIN_DIR, f"{ppp:.3f}ppp", f"{ns:.4f}Noise")
        for ppp, ns in product(TRAIN_PPPS, TRAIN_NOISE)
    ]

    # dataset loaded ONCE, shared by every network
    dataset     = CPUMemDataset(data_folders, N_CASES, IMG_H, IMG_W, N_OUT, CASE_OFFSET)
    data_loader = dataset.getDataLoader(BATCH_SIZE, drop_last=True, pin_memory=True,
                                        num_workers=0, shuffle=True)

    now         = dt.now().strftime("%Y_%m_%d_%H-%M")
    save_folder = os.path.join(SAVE_ROOT, f"PCNN_{now}")
    os.makedirs(save_folder, exist_ok=True)
    write_params(save_folder, now)

    t0 = perf_counter()
    for net_idx in range(N_NETWORKS):
        print(f"\n{'='*60}\nTraining network {net_idx + 1} / {N_NETWORKS}\n{'='*60}")
        train_one_network(net_idx, data_loader, save_folder, t0)


if __name__ == "__main__":
    main()
