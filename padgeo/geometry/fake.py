"""Model-free geometry backend: the exact geometry of a known synthetic scene.

It exists so that everything downstream of the geometry model (joining to the patch grid, caching,
anchored scoring, the profiling and kill-test scripts) can be developed and tested without VGGT.

Scene: the unit sphere centred at the origin of a synthetic world. Each image gets a pinhole camera that
looks at the origin from distance `CAMERA_DISTANCE`. Azimuth and elevation are derived from a hash of the
image's pixels within a cone of viewpoints, so the output for an image set is a deterministic function of
the images, as it is for VGGT, and different images see different, overlapping caps of the sphere. The field
of view is narrow enough that every pixel's ray hits the sphere.

The outputs follow VGGT's conventions exactly (see `padgeo.geometry.camera`):

- the world frame is the camera frame of image 0;
- extrinsics are camera-from-world with OpenCV axes;
- the principal point is (W/2, H/2), and pixel coordinates are integers;
- confidence is >= 1: 1 + 9 cos(angle between ray and surface normal), so it is low at grazing angles.

Consistency across views therefore holds exactly: a surface point seen in two images gets the same 3D
point in both, up to floating-point error.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from padgeo.config import FakeGeometryConfig
from padgeo.geometry.base import GeometryBackend, PixelGeometry
from padgeo.geometry.camera import unproject_depth
from padgeo.utils.checks import check_tensor
from padgeo.utils.precision import torch_dtype

CAMERA_DISTANCE = 3.0
# Focal length in units of image width. The sphere's silhouette subtends asin(1/3) = 19.5 degrees from the
# camera; with f = 2.5 W the image corners are at atan(sqrt(2) / 5) = 15.8 degrees, so every ray hits it.
FOCAL_FACTOR = 2.5
# Viewpoints are drawn in a cone (azimuth within +-45 degrees, elevation within +-30 degrees) so that images
# of one set overlap, as references and query do in practice; this also keeps cameras away from the poles,
# where the look-at frame degenerates.
MAX_AZIMUTH = math.pi / 4
MAX_ELEVATION = math.pi / 6


@dataclass(frozen=True)
class SyntheticCamera:
    """A pinhole camera in the synthetic world frame (float32 CPU tensors).

    Attributes:
        rotation: [3, 3] R of the camera-from-world transform X_cam = R X_world + t.
        translation: [3] t of the same transform.
        intrinsics: [3, 3] K in pixels, principal point (W/2, H/2).
    """

    rotation: torch.Tensor
    translation: torch.Tensor
    intrinsics: torch.Tensor


def viewpoint_for_image(image: torch.Tensor) -> tuple[float, float]:
    """Deterministic (azimuth, elevation) in radians derived from the SHA-256 of the image's pixels."""
    digest = hashlib.sha256(image.detach().cpu().contiguous().numpy().tobytes()).digest()
    u1 = int.from_bytes(digest[:8], "big") / 2**64
    u2 = int.from_bytes(digest[8:16], "big") / 2**64
    return MAX_AZIMUTH * (2.0 * u1 - 1.0), MAX_ELEVATION * (2.0 * u2 - 1.0)


def look_at_camera(azimuth: float, elevation: float, height: int, width: int) -> SyntheticCamera:
    """Camera on a sphere of radius `CAMERA_DISTANCE` looking at the origin, world +z up.

    Args:
        azimuth: Angle around world z, radians.
        elevation: Angle above the xy-plane, radians; |sin(elevation)| must be < 1.
        height: Image height H in pixels.
        width: Image width W in pixels.
    """
    centre = CAMERA_DISTANCE * torch.tensor(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=torch.float64,
    )
    forward = -centre / centre.norm()
    right = torch.linalg.cross(forward, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    right = right / right.norm()
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack([right, down, forward])  # rows: camera axes x, y, z in world coordinates
    focal = FOCAL_FACTOR * width
    intrinsics = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    return SyntheticCamera(
        rotation=rotation.to(torch.float32),
        translation=(-rotation @ centre).to(torch.float32),
        intrinsics=intrinsics.to(torch.float32),
    )


def raycast_unit_sphere(
    camera: SyntheticCamera, height: int, width: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact depth, world points and confidence of the unit sphere seen by `camera`.

    Rays use VGGT's pixel convention: pixel (v, u) has image coordinates (u, v), and its camera-frame ray
    direction is K^-1 [u, v, 1]^T, whose z component is 1, so the ray parameter equals the depth.

    Returns:
        depth [H, W], points [H, W, 3] (synthetic world frame) and confidence [H, W], all float32 on CPU.

    Raises:
        RuntimeError: If some ray misses the sphere (would violate the backend's design invariant).
    """
    rotation = camera.rotation.to(device=device, dtype=dtype)
    centre = -(camera.rotation.T @ camera.translation).to(device=device, dtype=dtype)
    k = camera.intrinsics
    rows, cols = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    directions_cam = torch.stack(
        [
            (cols - float(k[0, 2])) / float(k[0, 0]),
            (rows - float(k[1, 2])) / float(k[1, 1]),
            torch.ones_like(rows),
        ],
        dim=-1,
    )
    directions = directions_cam @ rotation  # row-vector form of R^T d: camera to world directions
    a = (directions * directions).sum(-1)
    b = 2.0 * (directions @ centre)
    c = (centre * centre).sum() - 1.0
    discriminant = b * b - 4.0 * a * c
    if not bool((discriminant > 0).all()):
        raise RuntimeError("a ray misses the synthetic sphere; field of view too wide")
    depth = (-b - discriminant.sqrt()) / (2.0 * a)
    points = centre + depth[..., None] * directions
    cos_incidence = -(directions * points).sum(-1) / a.sqrt()  # unit sphere: the normal is the point
    confidence = 1.0 + 9.0 * cos_incidence.clamp(0.0, 1.0)
    return depth.float().cpu(), points.float().cpu(), confidence.float().cpu()


def relative_extrinsic(camera: SyntheticCamera, reference: SyntheticCamera) -> torch.Tensor:
    """[3, 4] camera-from-world matrix of `camera` with the world frame moved to `reference`'s camera frame.

    With X_ref = R_ref X + t_ref, X_cam = R X + t gives X_cam = (R R_ref^T) X_ref + (t - R R_ref^T t_ref).
    """
    rotation = camera.rotation @ reference.rotation.T
    translation = camera.translation - rotation @ reference.translation
    return torch.cat([rotation, translation[:, None]], dim=1)


class FakeGeometryBackend(GeometryBackend):
    """Exact sphere geometry under hashed viewpoints, in VGGT's output conventions."""

    def __init__(self, config: FakeGeometryConfig) -> None:
        self.config = config
        self.dtype = torch_dtype(config.precision)
        self.device = torch.device(config.device)

    @property
    def identity(self) -> dict[str, Any]:
        """Backend settings that determine the output (part of geometry cache keys)."""
        return {
            "backend": "fake",
            "point_source": self.config.point_source,
            "precision": self.config.precision,
            "scene": (
                f"unit_sphere_d{CAMERA_DISTANCE}_f{FOCAL_FACTOR}_cone{MAX_AZIMUTH:.4f}_{MAX_ELEVATION:.4f}"
            ),
        }

    def predict(self, images: torch.Tensor) -> PixelGeometry:
        """Geometry for `images` [S, 3, H, W] with one hashed viewpoint per image."""
        check_tensor("images", images, (None, 3, None, None), torch.float32, value_range=(0.0, 1.0))
        _, _, height, width = images.shape
        cameras = [look_at_camera(*viewpoint_for_image(image), height, width) for image in images]
        return self.render(cameras, height, width)

    def render(self, cameras: Sequence[SyntheticCamera], height: int, width: int) -> PixelGeometry:
        """Geometry for explicit cameras, expressed in the camera frame of `cameras[0]`.

        The "depth" point source unprojects the rendered depth through `unproject_depth` (VGGT's
        utility, the same call as the VGGT backend's default branch); "pointmap" transforms the
        ray-cast points directly.
        """
        rendered = [raycast_unit_sphere(camera, height, width, self.dtype, self.device) for camera in cameras]
        depth = torch.stack([item[0] for item in rendered])
        confidence = torch.stack([item[2] for item in rendered])
        first = cameras[0]
        if self.config.point_source == "pointmap":
            points = torch.stack([item[1] @ first.rotation.T + first.translation for item in rendered])
        else:
            extrinsics = torch.stack([relative_extrinsic(camera, first) for camera in cameras])
            intrinsics = torch.stack([camera.intrinsics for camera in cameras])
            points = unproject_depth(depth, extrinsics, intrinsics)
        return PixelGeometry(points=points, confidence=confidence)
