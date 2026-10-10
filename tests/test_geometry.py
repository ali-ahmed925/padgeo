"""Geometry conventions, the fake backend, unprojection, patch pooling and the VGGT backend's construction.

Convention errors are the most likely silent bugs, so each convention gets its own test:
camera-from-world vs world-from-camera, OpenCV axes, the pixel-centre convention, row/column order and the
patch-grid mapping.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from padgeo.config import FakeGeometryConfig, VGGTGeometryConfig
from padgeo.geometry import build_geometry_backend
from padgeo.geometry.base import PixelGeometry
from padgeo.geometry.camera import unproject_depth
from padgeo.geometry.fake import (
    FakeGeometryBackend,
    SyntheticCamera,
    look_at_camera,
    raycast_unit_sphere,
    relative_extrinsic,
    viewpoint_for_image,
)
from padgeo.geometry.patches import pool_to_patch_grid
from padgeo.geometry.vggt_backend import VGGTBackend, keep_needed_heads

H = W = 70  # 5 x 5 patches of 14 px: small but on the real patch size


def fake(point_source: str = "depth", precision: str = "fp32") -> FakeGeometryBackend:
    return FakeGeometryBackend(
        FakeGeometryConfig(backend="fake", point_source=point_source, precision=precision)
    )


# --- unprojection conventions (VGGT's utility, used by both backends) --------------------------------


def test_unproject_depth_camera_from_world_and_opencv_axes():
    # Camera rotated 90 degrees about world y and translated: X_cam = R X_world + t.
    angle = math.pi / 2
    rotation = torch.tensor(
        [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]],
        dtype=torch.float32,
    )
    translation = torch.tensor([0.5, -0.2, 3.0])
    extrinsic = torch.cat([rotation, translation[:, None]], dim=1)[None]
    focal, cx, cy = 50.0, W / 2, H / 2
    intrinsic = torch.tensor([[[focal, 0, cx], [0, focal, cy], [0, 0, 1]]])
    depth = torch.full((1, H, W), 2.0)
    points = unproject_depth(depth, extrinsic, intrinsic)
    # Pixel (row v=40, col u=10): camera point ((u - cx) z / f, (v - cy) z / f, z) with integer u, v.
    u, v = 10, 40
    x_cam = torch.tensor([(u - cx) * 2.0 / focal, (v - cy) * 2.0 / focal, 2.0])
    expected_world = rotation.T @ (x_cam - translation)
    assert torch.allclose(points[0, v, u], expected_world, atol=1e-5)
    # The opposite convention (treating the matrix as world-from-camera) would give a different point.
    assert not torch.allclose(points[0, v, u], rotation @ x_cam + translation, atol=1e-2)


def test_unproject_depth_pixel_centre_and_axis_directions():
    identity = torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=1)[None]
    focal = 40.0
    intrinsic = torch.tensor([[[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]]])
    points = unproject_depth(torch.ones(1, H, W), identity, intrinsic)
    # Integer pixel coordinates with the principal point at (W/2, H/2): pixel (H/2, W/2) is on the axis.
    assert torch.allclose(points[0, H // 2, W // 2], torch.tensor([0.0, 0.0, 1.0]))
    # Column index u grows to the right (+x), row index v grows downwards (+y).
    assert points[0, 0, W - 1, 0] > 0 > points[0, 0, 0, 0]
    assert points[0, H - 1, 0, 1] > 0 > points[0, 0, 0, 1]


# --- fake backend ------------------------------------------------------------------------------------


def two_cameras() -> list[SyntheticCamera]:
    return [look_at_camera(0.3, 0.1, H, W), look_at_camera(0.6, 0.2, H, W)]


def test_fake_camera_is_a_proper_rotation_looking_at_origin():
    camera = look_at_camera(1.0, 0.4, H, W)
    assert torch.allclose(camera.rotation @ camera.rotation.T, torch.eye(3), atol=1e-6)
    assert torch.linalg.det(camera.rotation) == pytest.approx(1.0, abs=1e-6)
    origin_in_camera = camera.translation  # X_cam of the world origin is R 0 + t
    assert origin_in_camera[0] == pytest.approx(0.0, abs=1e-6)
    assert origin_in_camera[1] == pytest.approx(0.0, abs=1e-6)
    assert origin_in_camera[2] > 0  # in front of the camera (+z forward)


def test_fake_raycast_points_lie_on_unit_sphere_and_match_depth():
    camera = look_at_camera(0.2, -0.3, H, W)
    depth, points, confidence = raycast_unit_sphere(camera, H, W, torch.float32, torch.device("cpu"))
    assert torch.allclose(points.norm(dim=-1), torch.ones(H, W), atol=1e-5)
    camera_points = points @ camera.rotation.T + camera.translation
    assert torch.allclose(camera_points[..., 2], depth, atol=1e-5)  # depth is camera-frame z
    assert bool((confidence >= 1.0).all())
    assert bool((confidence <= 10.0).all())
    # Grazing rays (image border) are less confident than the central ray.
    assert confidence[H // 2, W // 2] > confidence[0, 0]


def test_fake_depth_and_pointmap_sources_agree():
    cameras = two_cameras()
    depth_points = fake("depth").render(cameras, H, W).points
    map_points = fake("pointmap").render(cameras, H, W).points
    assert torch.allclose(depth_points, map_points, atol=1e-4)


def test_fake_world_frame_is_first_camera_frame():
    cameras = two_cameras()
    geometry = fake("pointmap").render(cameras, H, W)
    depth, _, _ = raycast_unit_sphere(cameras[0], H, W, torch.float32, torch.device("cpu"))
    # In image 0's own camera frame, the z coordinate of its points is its depth.
    assert torch.allclose(geometry.points[0, ..., 2], depth, atol=1e-5)


def test_fake_cross_view_consistency():
    """A surface point seen in image 1 projects into image 0 where image 0 sees the same point."""
    cameras = two_cameras()
    geometry = fake("depth").render(cameras, H, W)
    k = cameras[0].intrinsics
    matched = 0
    for v in range(0, H, 7):
        for u in range(0, W, 7):
            point = geometry.points[1, v, u]  # frame 0 = camera 0 frame
            if point[2] <= 0:
                continue
            u0 = float(k[0, 0] * point[0] / point[2] + k[0, 2])
            v0 = float(k[1, 1] * point[1] / point[2] + k[1, 2])
            iu, iv = round(u0), round(v0)
            if not (0 <= iu < W and 0 <= iv < H):
                continue
            seen = geometry.points[0, iv, iu]
            if abs(float(seen[2]) - float(point[2])) > 0.02:  # occluded in image 0, or a steep silhouette
                continue
            # Same surface point up to rounding to the nearest pixel (footprint ~ depth / f ~ 0.01).
            assert float((seen - point).norm()) < 0.03
            matched += 1
    assert matched > 10


def test_fake_relative_extrinsic_round_trip():
    a, b = two_cameras()
    world_point = torch.tensor([0.1, -0.2, 0.3])
    in_a = a.rotation @ world_point + a.translation
    in_b = b.rotation @ world_point + b.translation
    relative = relative_extrinsic(b, a)
    assert torch.allclose(relative[:, :3] @ in_a + relative[:, 3], in_b, atol=1e-6)


def test_fake_predict_is_deterministic_and_order_dependent_only_through_the_frame():
    gen = torch.Generator().manual_seed(0)
    images = torch.rand(3, 3, H, W, generator=gen)
    backend = fake()
    first = backend.predict(images)
    again = backend.predict(images)
    assert torch.equal(first.points, again.points)
    assert viewpoint_for_image(images[0]) != viewpoint_for_image(images[1])
    # Reordering images 1 and 2 (image 0 fixed) permutes the outputs exactly: same world frame.
    swapped = backend.predict(images[[0, 2, 1]])
    assert torch.allclose(swapped.points[1], first.points[2], atol=1e-5)
    assert torch.allclose(swapped.points[2], first.points[1], atol=1e-5)


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_fake_half_precision_is_close_to_full(precision):
    cameras = two_cameras()
    full = fake("pointmap", "fp32").render(cameras, H, W).points
    half = fake("pointmap", precision).render(cameras, H, W).points
    shift = (half - full).norm(dim=-1)
    # bf16 keeps 8 mantissa bits; the ray-sphere discriminant cancels, so shifts of a few 1e-2 are expected.
    assert float(shift.max()) < 0.15
    assert float(shift.max()) > 0  # the precision setting really changes the computation


# --- join: pooling onto the patch grid ---------------------------------------------------------------


def test_pool_to_patch_grid_row_column_order_and_alignment():
    rows, cols = torch.meshgrid(
        torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij"
    )
    points = torch.stack([cols, rows, torch.zeros_like(rows)], dim=-1)[None]  # x = column, y = row
    geometry = PixelGeometry(points=points, confidence=torch.ones(1, H, W))
    pooled = pool_to_patch_grid(geometry, torch.ones(1, H, W, dtype=torch.bool), patch_size=14)
    assert pooled.xyz.shape == (1, 5, 5, 3)
    # Patch (i, j) covers rows 14i..14i+13 and columns 14j..14j+13: mean column 14j + 6.5, mean row 14i + 6.5.
    for i in range(5):
        for j in range(5):
            assert pooled.xyz[0, i, j, 0] == pytest.approx(14 * j + 6.5)
            assert pooled.xyz[0, i, j, 1] == pytest.approx(14 * i + 6.5)


def test_pool_to_patch_grid_uses_only_foreground_pixels():
    points = torch.zeros(1, H, W, 3)
    points[0, :, :7, 2] = 1.0  # left half of the first patch column is at depth 1, the rest at 0
    confidence = torch.ones(1, H, W)
    confidence[0, :, :7] = 9.0
    foreground = torch.zeros(1, H, W, dtype=torch.bool)
    foreground[0, :14, :7] = True  # only the left half of patch (0, 0) is foreground
    pooled = pool_to_patch_grid(PixelGeometry(points, confidence), foreground, patch_size=14)
    assert pooled.xyz[0, 0, 0, 2] == pytest.approx(1.0)  # not 0.5: background pixels excluded
    assert pooled.confidence[0, 0, 0] == pytest.approx(9.0)
    assert torch.isnan(pooled.xyz[0, 1, 1]).all()  # no foreground pixel -> NaN
    assert torch.isnan(pooled.confidence[0, 1, 1])


def test_pool_to_patch_grid_rejects_non_multiple_sizes():
    geometry = PixelGeometry(points=torch.zeros(1, 15, 14, 3), confidence=torch.ones(1, 15, 14))
    with pytest.raises(ValueError, match="multiple of the patch size"):
        pool_to_patch_grid(geometry, torch.ones(1, 15, 14, dtype=torch.bool), patch_size=14)


# --- VGGT backend -------------------------------------------------------------------------------------

VGGT_CONFIG = VGGTGeometryConfig(backend="vggt", revision="860abec7937da0a4c03c41d3c269c366e82abdf9")


def test_vggt_architecture_builds_on_cpu_and_keeps_only_depth_heads():
    """The real VGGT-1B architecture (random weights, ~5 GB RAM, ~12 s; meta tensors are not supported by
    its constructor) builds, and the depth point source keeps exactly the camera and depth heads."""
    from vggt.models.vggt import VGGT

    model = keep_needed_heads(VGGT(), "depth")
    assert model.camera_head is not None
    assert model.depth_head is not None
    assert model.point_head is None
    assert model.track_head is None
    del model


class _Heads(torch.nn.Module):
    """Stand-in with VGGT's four head attributes, to check head selection without building VGGT."""

    def __init__(self) -> None:
        super().__init__()
        for name in ("camera_head", "depth_head", "point_head", "track_head"):
            setattr(self, name, torch.nn.Linear(1, 1))


@pytest.mark.parametrize(
    ("point_source", "kept"),
    [("depth", {"camera_head", "depth_head"}), ("pointmap", {"point_head"})],
)
def test_keep_needed_heads(point_source, kept):
    model = keep_needed_heads(_Heads(), point_source)  # type: ignore[arg-type]
    present = {
        name
        for name in ("camera_head", "depth_head", "point_head", "track_head")
        if getattr(model, name) is not None
    }
    assert present == kept


def test_vggt_backend_builds_without_loading_weights():
    backend = build_geometry_backend(VGGT_CONFIG)
    assert isinstance(backend, VGGTBackend)
    assert backend._model is None  # weights are loaded lazily, on the first predict
    assert backend.identity["revision"] == VGGT_CONFIG.revision
    assert backend.identity["point_source"] == "depth"


@pytest.mark.vggt_weights
@pytest.mark.skipif(
    os.environ.get("PADGEO_RUN_VGGT") != "1", reason="set PADGEO_RUN_VGGT=1 (downloads ~5 GB)"
)
def test_vggt_forward_smoke():
    """Experiment machine only: one real VGGT-1B forward pass; checks output shapes and confidence range."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = VGGTBackend(VGGT_CONFIG.model_copy(update={"device": device}))
    images = torch.rand(3, 3, 518, 518, generator=torch.Generator().manual_seed(0))
    geometry = backend.predict(images)
    assert geometry.points.shape == (3, 518, 518, 3)
    assert bool((geometry.confidence >= 1.0).all())
