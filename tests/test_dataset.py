"""Unit tests for Market-1501 dataset parsing and transforms."""

from pathlib import Path
from PIL import Image
import torch
from reid.data import Market1501, build_transforms


def test_filename_parsing() -> None:
    # Pattern: [pid]_c[camid]s[seq]_[frame]_[idx].jpg
    parsed = Market1501.FILENAME_PATTERN.match("0002_c1s1_000451_03.jpg")
    assert parsed is not None
    assert int(parsed.group(1)) == 2
    assert int(parsed.group(2)) == 1

    parsed_junk = Market1501.FILENAME_PATTERN.match("-1_c2s1_001051_00.jpg")
    assert parsed_junk is not None
    assert int(parsed_junk.group(1)) == -1
    assert int(parsed_junk.group(2)) == 2


def test_transforms_output_shape() -> None:
    transform = build_transforms(image_size=(256, 128), is_train=True)
    img = Image.new("RGB", (100, 200), color=(128, 128, 128))
    tensor = transform(img)
    assert tensor.shape == (3, 256, 128)
    assert isinstance(tensor, torch.Tensor)


def test_market1501_real_data_if_present() -> None:
    import os
    env_dir = os.environ.get("REID_DATA_ROOT")
    data_dir = Path(env_dir) if env_dir else Path("data")
    if (data_dir / "Market-1501-v15.09.15").is_dir() or (data_dir / "bounding_box_train").is_dir():
        ds_train = Market1501(root_dir=data_dir, split="train")
        assert len(ds_train) > 0
        img, pid, cam = ds_train[0]
        assert img.shape == (3, 256, 128)
        assert 0 <= pid < ds_train.num_classes
        assert cam in {1, 2, 3, 4, 5, 6}

        ds_query = Market1501(root_dir=data_dir, split="query")
        assert len(ds_query) > 0

        ds_gallery = Market1501(root_dir=data_dir, split="gallery")
        assert len(ds_gallery) > 0
