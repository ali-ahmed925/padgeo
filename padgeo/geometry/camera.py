"""Depth unprojection with VGGT's own utility, shared by the VGGT and fake backends.

Conventions (VGGT's, which the fake backend reproduces so that this code path is tested without VGGT):

- `extrinsics[s] = [R | t]` maps world to camera: X_cam = R X_world + t (camera-from-world).
- Camera axes follow OpenCV: +x right, +y down, +z forward; depth is the camera-frame z coordinate.
- Pixel (row v, column u) has image coordinates (x, y) = (u, v) with integer u, v (no +0.5 offset), and
  `intrinsics[s] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` with the principal point at (W/2, H/2).
- Therefore X_cam = depth * K^-1 [u, v, 1]^T and X_world = R^T (X_cam - t).
"""

from __future__ import annotations

import torch
from vggt.utils.geometry import unproject_depth_map_to_point_map

from padgeo.utils.checks import check_tensor


def unproject_depth(depth: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """World points from depth maps and cameras, via `vggt.utils.geometry.unproject_depth_map_to_point_map`.

    Args:
        depth: [S, H, W] float32 camera-frame depth (> 0).
        extrinsics: [S, 3, 4] float32 camera-from-world matrices.
        intrinsics: [S, 3, 3] float32 pinhole intrinsics in pixels.

    Returns:
        [S, H, W, 3] float32 points in the world frame of `extrinsics`.
    """
    check_tensor("depth", depth, (None, None, None), torch.float32, finite=True)
    num = depth.shape[0]
    check_tensor("extrinsics", extrinsics, (num, 3, 4), torch.float32, finite=True)
    check_tensor("intrinsics", intrinsics, (num, 3, 3), torch.float32, finite=True)
    # VGGT's function takes depth as (S, H, W, 1): it squeezes the last axis of each frame.
    points = unproject_depth_map_to_point_map(
        depth.detach().cpu().numpy()[..., None],
        extrinsics.detach().cpu().numpy(),
        intrinsics.detach().cpu().numpy(),
    )
    return torch.from_numpy(points).to(torch.float32)
