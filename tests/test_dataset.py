"""Unit tests for Market-1501 dataset parsing and transforms."""

from pathlib import Path

import torch
from PIL import Image

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
    if (data_dir / "Market-1501-v15.09.15").is_dir() or (
        data_dir / "bounding_box_train"
    ).is_dir():
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


def test_val_split_disjointness_and_exclusion(tmp_path: Path) -> None:
    """Verify that val and train identities are disjoint and val-query never appears in val-gallery."""
    train_dir = tmp_path / "bounding_box_train"
    train_dir.mkdir(parents=True, exist_ok=True)

    dummy_img = Image.new("RGB", (64, 128), color=(100, 150, 200))

    # Create 10 identities (pids 1 to 10), 2 cameras each
    # pid 1..5: 3 images in c1, 1 image in c2
    # pid 6..10: 2 images in c1, 2 images in c2
    for pid in range(1, 11):
        for cam in (1, 2):
            num_imgs = (
                3 if (pid <= 5 and cam == 1) else (1 if (pid <= 5 and cam == 2) else 2)
            )
            for idx in range(num_imgs):
                fname = f"{pid:04d}_c{cam}s1_{pid * 100 + idx:06d}_{idx:02d}.jpg"
                dummy_img.save(train_dir / fname)

    val_identities = 4
    val_seed = 42

    ds_train = Market1501(
        root_dir=tmp_path,
        split="train",
        val_identities=val_identities,
        val_seed=val_seed,
    )
    ds_val_q = Market1501(
        root_dir=tmp_path,
        split="val_query",
        val_identities=val_identities,
        val_seed=val_seed,
    )
    ds_val_g = Market1501(
        root_dir=tmp_path,
        split="val_gallery",
        val_identities=val_identities,
        val_seed=val_seed,
    )

    # 1. Check identity counts
    assert len(ds_train.heldout_val_pids) == val_identities
    assert len(ds_train.train_pids) == 10 - val_identities
    assert set(ds_train.heldout_val_pids).isdisjoint(set(ds_train.train_pids))

    # 2. Check that val-query and val-gallery identities belong strictly to held-out set
    val_q_pids = {s[1] for s in ds_val_q.samples}
    val_g_pids = {s[1] for s in ds_val_g.samples}
    heldout_set = set(ds_train.heldout_val_pids)

    assert val_q_pids.issubset(heldout_set)
    assert val_g_pids.issubset(heldout_set)
    assert val_q_pids.isdisjoint(set(ds_train.train_pids))
    assert val_g_pids.isdisjoint(set(ds_train.train_pids))

    # 3. Check that val-query and val-gallery images are strictly disjoint
    val_q_paths = {s[0] for s in ds_val_q.samples}
    val_g_paths = {s[0] for s in ds_val_g.samples}

    assert len(val_q_paths) > 0
    assert len(val_g_paths) > 0
    assert val_q_paths.isdisjoint(val_g_paths), (
        "Val-query images must NEVER appear in val-gallery!"
    )
