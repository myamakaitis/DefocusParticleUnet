"""Trilinear grid interpolation backed by F.grid_sample (GPU-native)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GridSampleInterpolator(nn.Module):
    """Linear or cubic interpolation on a uniform 2-D or 3-D regular grid,
    backed by F.grid_sample."""

    def __init__(self, grid, values: torch.Tensor, interp_mode: str = 'bilinear'):
        super().__init__()
        ndim = len(grid)
        if ndim not in (2, 3):
            raise NotImplementedError(
                "GridSampleInterpolator supports 2-D or 3-D tables.")
        expected = tuple(len(g) for g in grid)
        if tuple(values.shape) != expected:
            raise ValueError(f"values.shape={tuple(values.shape)} != grid sizes {expected}")

        self.ndim = ndim
        self.interp_mode = interp_mode

        coord_min = torch.tensor([float(g[0]) for g in grid], dtype=torch.float32)
        coord_max = torch.tensor([float(g[-1]) for g in grid], dtype=torch.float32)
        self.register_buffer('coord_min', coord_min, persistent=False)
        self.register_buffer('coord_max', coord_max, persistent=False)

        # grid_sample's last-dim order is (x, y, z) → reverse of user-space
        # (axis0, axis1, ...). Permute once at construction.
        perm = list(reversed(range(ndim)))
        values_perm = values.permute(*perm).contiguous()
        stored_shape = [1, 1] + list(values_perm.shape)
        self.values = nn.Parameter(values_perm.view(*stored_shape))

    def _normalize(self, query: torch.Tensor) -> torch.Tensor:
        span = self.coord_max - self.coord_min
        return 2.0 * (query - self.coord_min) / span - 1.0

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """query: [N, G, ndim] in user-space coords. Returns [N, G]."""
        N, G, _ = query.shape
        q = self._normalize(query)
        if self.ndim == 2:
            grid = q.view(1, 1, N * G, 2)
            out = F.grid_sample(self.values, grid, mode=self.interp_mode,
                                padding_mode='zeros', align_corners=True)
        else:
            grid = q.view(1, 1, 1, N * G, 3)
            out = F.grid_sample(self.values, grid, mode=self.interp_mode,
                                padding_mode='zeros', align_corners=True)
        return out.view(N, G)
