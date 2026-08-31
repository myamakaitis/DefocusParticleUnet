"""Peak-detection CNN architecture (ConvNeXt-V2 U-Net).

Standalone copy of the network definition used for the publication results.
The checkpoint format stores ``init_args`` alongside the ``state_dict`` so a
model can be reconstructed with ``PeakCNN_UNet_4level_ConvNeXt.load_checkpoint``.

Down/up-sampling uses stride-2 conv / transpose-conv with padding=0, so H and W
must be divisible by 16.  The transpose path then exactly inverts the downsampling
and the interpolation fallback in ``UpNextV2`` never fires.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for (B, C, H, W) tensors: nn.LayerNorm(C) applied
    independently at every spatial position."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlockV2(nn.Module):
    """Depthwise k×k conv → LayerNorm → channel MLP with GRN after GELU → residual.

    Follows facebookresearch/ConvNeXt-V2: the GRN acts on the expanded
    (mlp_ratio x dim) features between the two pointwise convs.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 7, mlp_ratio: float = 4.0):
        super().__init__()
        pad = kernel_size // 2
        n_h = int(out_ch * mlp_ratio)

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, groups=in_ch, kernel_size=kernel_size, padding=pad, bias=False),
            LayerNorm2d(in_ch),
            nn.Conv2d(in_ch, n_h, kernel_size=1, padding=0, bias=False),   # reverse bottleneck
            nn.GELU(),
            GRN(n_h),
            nn.Conv2d(n_h, out_ch, kernel_size=1, padding=0, bias=False),  # bottleneck
        )
        self.resample = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.resample(x) + self.block(x)


class DownNextV2(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, mlp_ratio: float = 4.0):
        super().__init__()
        self.down = nn.Sequential(
            LayerNorm2d(in_ch),
            GRN(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=False),  # aligned 2x downsample
        )
        self.conv = ConvNeXtBlockV2(in_ch, out_ch, kernel_size, mlp_ratio)

    def forward(self, x):
        return self.conv(self.down(x))


class UpNextV2(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel_size: int = 7, mlp_ratio: float = 4.0):
        super().__init__()
        self.up = nn.Sequential(
            LayerNorm2d(in_ch),
            GRN(in_ch),
            nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2, padding=0, bias=False),  # aligned 2x upsample
        )
        self.combine = nn.Conv2d(in_ch + skip_ch, in_ch, kernel_size=1, padding=0, bias=False)
        self.conv = ConvNeXtBlockV2(in_ch, out_ch, kernel_size=kernel_size, mlp_ratio=mlp_ratio)

    def forward(self, x_in, x_skip):
        x_in_us = self.up(x_in)
        if x_in_us.size()[-2:] != x_skip.size()[-2:]:
            x_in_us = F.interpolate(x_in_us, size=x_skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(self.combine(torch.cat([x_skip, x_in_us], dim=1)))


class PeakCNN_UNet_4level_ConvNeXt(nn.Module):
    """4-level U-Net built from ConvNeXt-V2 blocks.

    Input : B × N_in  × H × W   (H, W divisible by 16)
    Output: B × N_out × H × W
        ch 0     : peak-probability logits (apply sigmoid)
        ch 1, 2  : sub-pixel offsets (dp, dq) in pixel units
        ch 3     : depth z
        ch 4     : peak amplitude - vestigial, never trained (lambda_I = 0)
    """
    def __init__(self, N_in: int, N_mid: int, N_out: int,
                 kernel_size: int = 7, mlp_ratio: float = 4.0):
        super().__init__()
        self.init_args = {'N_in': N_in, 'N_mid': N_mid, 'N_out': N_out,
                          'kernel_size': kernel_size, 'mlp_ratio': mlp_ratio}

        self.stem = nn.Sequential(
            nn.Conv2d(N_in, N_mid, kernel_size=1),
            ConvNeXtBlockV2(N_mid, N_mid, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        )
        self.down1 = DownNextV2(N_mid, N_mid, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.down2 = DownNextV2(N_mid, N_mid, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.down3 = DownNextV2(N_mid, N_mid, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.down4 = DownNextV2(N_mid, N_mid, kernel_size=kernel_size, mlp_ratio=mlp_ratio)

        self.up1 = UpNextV2(N_mid, N_mid, N_mid)
        self.up2 = UpNextV2(N_mid, N_mid, N_mid)
        self.up3 = UpNextV2(N_mid, N_mid, N_mid)
        self.up4 = UpNextV2(N_mid, N_mid, N_mid)
        self.final = nn.Conv2d(N_mid, N_out, kernel_size=1, bias=True)

    def __repr__(self):
        return "4-Level ConvNeXt U-net"

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.final(x)

    def save_checkpoint(self, filepath: str):
        torch.save({'arch': 'convnext', 'init_args': self.init_args,
                    'state_dict': self.state_dict()}, filepath)

    @classmethod
    def load_checkpoint(cls, filepath: str, device='cpu'):
        ck = torch.load(filepath, map_location=device)
        model = cls(**ck['init_args'])
        model.load_state_dict(ck['state_dict'])
        return model.to(device)
