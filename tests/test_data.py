"""MAD-Sim loader on the fixture dataset (same on-disk format as MAD-Sim), and model-input preprocessing."""

from __future__ import annotations

import shutil

import pytest
import torch
from PIL import Image
from vggt.utils.load_fn import load_and_preprocess_images

from padgeo.data.base import hash_file
from padgeo.data.images import MODEL_INPUT_WIDTH, load_model_input, model_input_height
from padgeo.data.mad_sim import MADSim


def test_split_lists_pool_and_queries_without_labels(dataset_root):
    split = MADSim(dataset_root).load_split("01Alpha")
    assert len(split.reference_pool) == 8
    assert [r.content_hash for r in split.reference_pool] == sorted(
        r.content_hash for r in split.reference_pool
    )
    names = [q.name for q in split.queries]
    assert names == sorted(names)
    assert len(names) == 4 + 2 + 2  # good (incl. the blank image), Missing, Stains
    assert all(name.startswith("test/") for name in names)
    assert all(ref.name.startswith("train/good/") for ref in split.reference_pool)
    for ref in (*split.reference_pool, *split.queries):
        assert ref.content_hash == hash_file(ref.path)


def test_ground_truth_labels_types_and_masks(dataset_root):
    targets = MADSim(dataset_root).ground_truth("01Alpha")
    assert targets["test/good/0.png"].label == 0
    assert targets["test/good/0.png"].mask_path is None
    missing = targets["test/Missing/1.png"]
    assert (missing.label, missing.defect_type) == (1, "Missing")
    assert missing.mask_path == dataset_root / "01Alpha" / "ground_truth" / "Missing" / "1_mask.png"
    assert set(targets) == {q.name for q in MADSim(dataset_root).load_split("01Alpha").queries}


def test_unknown_class_and_missing_mask_fail_loudly(dataset_root, tmp_path):
    with pytest.raises(FileNotFoundError, match="available"):
        MADSim(dataset_root).load_split("99Nothing")
    copy = tmp_path / "copy"
    shutil.copytree(dataset_root / "01Alpha", copy / "01Alpha")
    (copy / "01Alpha" / "ground_truth" / "Stains" / "0_mask.png").unlink()
    with pytest.raises(FileNotFoundError, match="missing mask"):
        MADSim(copy).ground_truth("01Alpha")


def test_query_identical_to_training_image_is_rejected(dataset_root, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(dataset_root / "01Alpha", copy / "01Alpha")
    shutil.copy(copy / "01Alpha" / "train" / "good" / "3.png", copy / "01Alpha" / "test" / "good" / "9.png")
    with pytest.raises(ValueError, match="identical to training images"):
        MADSim(copy).load_split("01Alpha")


@pytest.mark.usefixtures("forbid_transforms_json")
def test_transforms_json_is_never_opened(dataset_root):
    dataset = MADSim(dataset_root)
    for class_name in ("01Alpha", "02Beta"):
        dataset.load_split(class_name)
        dataset.ground_truth(class_name)
    # The fixture's transforms.json is not even valid JSON, so any parse attempt would also fail loudly.


def test_model_input_is_vggt_preprocessing(dataset_root):
    path = dataset_root / "01Alpha" / "train" / "good" / "0.png"
    model_input = load_model_input(path)
    assert model_input.pixels.shape == (3, 518, 518)
    assert model_input.original_size == (112, 112)
    assert torch.equal(model_input.pixels, load_and_preprocess_images([str(path)], mode="crop")[0])


def test_model_input_landscape_and_portrait(tmp_path):
    landscape = tmp_path / "landscape.png"
    Image.new("RGB", (800, 400), (255, 255, 255)).save(landscape)
    assert model_input_height((400, 800)) == 252  # round(18.5) * 14 with banker's rounding = 18 * 14
    assert load_model_input(landscape).pixels.shape == (3, 252, MODEL_INPUT_WIDTH)
    portrait = tmp_path / "portrait.png"
    Image.new("RGB", (400, 800), (255, 255, 255)).save(portrait)
    with pytest.raises(ValueError, match="centre-cropped"):
        load_model_input(portrait)
