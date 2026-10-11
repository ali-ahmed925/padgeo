"""VGGTBackend.predict without the 5 GB checkpoint: a stub stands in for the network.

The stub's heads return the exact geometry of the fake backend's synthetic scene, with the cameras passed
through VGGT's own camera encoding (`extri_intri_to_pose_encoding`). The backend therefore has to decode
the cameras with `pose_encoding_to_extri_intri` (including the (H, W) order), index VGGT's output shapes
correctly and unproject depth in the right convention; only then does it reproduce the scene's points.
This checks every line of the real backend's forward path except the network weights themselves.
"""

from __future__ import annotations

import pytest
import torch
from vggt.utils.pose_enc import extri_intri_to_pose_encoding

from padgeo.config import FakeGeometryConfig, VGGTGeometryConfig
from padgeo.geometry import vggt_backend
from padgeo.geometry.fake import FakeGeometryBackend, look_at_camera, raycast_unit_sphere, relative_extrinsic
from padgeo.geometry.vggt_backend import VGGTBackend

H, W = 42, 56  # non-square on purpose: catches (H, W) vs (W, H) mix-ups
VIEWPOINTS = [(0.1, 0.2), (0.5, 0.1), (-0.3, 0.3)]


class FixedOutput(torch.nn.Module):
    """A stand-in head: ignores its inputs (tokens, images, patch_start_idx) and returns a fixed output."""

    def __init__(self, output: object) -> None:
        super().__init__()
        self.output = output

    def forward(self, *_inputs: object, **_named_inputs: object) -> object:
        return self.output


class RecordingOutput(FixedOutput):
    """A FixedOutput that records whether CPU autocast was active, and its dtype, when called."""

    def __init__(self, output: object) -> None:
        super().__init__(output)
        self.autocast: tuple[bool, torch.dtype | None] | None = None

    def forward(self, *_inputs: object, **_named_inputs: object) -> object:
        enabled = torch.is_autocast_cpu_enabled()
        self.autocast = (enabled, torch.get_autocast_cpu_dtype() if enabled else None)
        return self.output


class StubVGGT(torch.nn.Module):
    """Mimics VGGT's module interface and output shapes; outputs the fake backend's exact scene."""

    def __init__(self) -> None:
        super().__init__()
        cameras = [look_at_camera(az, el, H, W) for az, el in VIEWPOINTS]
        rendered = [raycast_unit_sphere(c, H, W, torch.float32, torch.device("cpu")) for c in cameras]
        extrinsics = torch.stack([relative_extrinsic(c, cameras[0]) for c in cameras])[None]
        intrinsics = torch.stack([c.intrinsics for c in cameras])[None]
        self.pose_enc = extri_intri_to_pose_encoding(extrinsics, intrinsics, (H, W))  # [1, S, 9]
        self.depth = torch.stack([r[0] for r in rendered])[None, ..., None]  # [1, S, H, W, 1]
        self.conf = torch.stack([r[2] for r in rendered])[None]  # [1, S, H, W]
        fake = FakeGeometryBackend(FakeGeometryConfig(backend="fake"))
        self.points = fake.render(cameras, H, W).points[None]  # [1, S, H, W, 3]
        self.aggregator = RecordingOutput(([torch.zeros(1)], 5))  # (token list, patch start index)
        self.camera_head = FixedOutput([self.pose_enc])  # one pose encoding per refinement iteration
        self.depth_head = RecordingOutput((self.depth, self.conf))
        self.point_head = FixedOutput((self.points, self.conf))
        self.track_head = FixedOutput(None)
        self.calls: list[tuple[str, str, bool]] = []  # arguments of each from_pretrained call


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubVGGT:
    model = StubVGGT()

    def from_pretrained(name: str, revision: str, strict: bool) -> StubVGGT:
        model.calls.append((name, revision, strict))
        return model

    monkeypatch.setattr(vggt_backend.VGGT, "from_pretrained", staticmethod(from_pretrained))
    return model


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_depth_branch_decodes_cameras_and_unprojects_like_the_scene(stub, precision):
    config = VGGTGeometryConfig(
        backend="vggt", revision="r1", point_source="depth", precision=precision, device="cpu"
    )
    backend = VGGTBackend(config)
    geometry = backend.predict(torch.rand(len(VIEWPOINTS), 3, H, W))
    assert stub.calls == [("facebook/VGGT-1B", "r1", True)]  # pinned revision, strict loading
    assert stub.point_head is None  # the depth source drops the point head
    assert stub.track_head is None
    assert geometry.points.shape == (len(VIEWPOINTS), H, W, 3)
    assert torch.allclose(geometry.points, stub.points[0], atol=2e-4)  # = the scene, via VGGT's decoding
    assert torch.equal(geometry.confidence, stub.conf[0])


def test_pointmap_branch_returns_the_point_head_output(stub):
    config = VGGTGeometryConfig(
        backend="vggt", revision="r1", point_source="pointmap", precision="fp32", device="cpu"
    )
    geometry = VGGTBackend(config).predict(torch.rand(len(VIEWPOINTS), 3, H, W))
    assert stub.camera_head is None  # the point-map source drops the camera and depth heads
    assert stub.depth_head is None
    assert torch.equal(geometry.points, stub.points[0])
    assert torch.equal(geometry.confidence, stub.conf[0])


def test_model_is_loaded_once_and_lazily(stub):
    backend = VGGTBackend(VGGTGeometryConfig(backend="vggt", revision="r1", device="cpu", precision="fp32"))
    assert stub.calls == []
    images = torch.rand(len(VIEWPOINTS), 3, H, W)
    backend.predict(images)
    backend.predict(images)
    assert len(stub.calls) == 1


@pytest.mark.parametrize(
    ("precision", "aggregator"), [("fp32", (False, None)), ("bf16", (True, torch.bfloat16))]
)
def test_precision_sets_autocast_for_the_aggregator_and_never_for_the_heads(stub, precision, aggregator):
    config = VGGTGeometryConfig(backend="vggt", revision="r1", precision=precision, device="cpu")
    VGGTBackend(config).predict(torch.rand(len(VIEWPOINTS), 3, H, W))
    assert stub.aggregator.autocast == aggregator
    assert stub.depth_head.autocast == (False, None)  # heads run in fp32, as in VGGT.forward
