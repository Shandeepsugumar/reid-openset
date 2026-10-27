"""Unit tests for model-independent look-alike protocol, grouping, and failure analysis."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from reid.eval.lookalike import (
    bhattacharyya_distance,
    cluster_lookalike_groups,
    compute_clothing_descriptor,
    partition_lookalike_split,
    partition_matched_control_split,
)
from reid.eval.metrics import compute_metrics_from_scores_v2
from scripts.failure_analysis import verify_file_lists


def _create_synthetic_pedestrian(
    upper_color: tuple, lower_color: tuple, size: tuple = (256, 128)
) -> Image.Image:
    """Create a synthetic pedestrian image with distinct upper and lower colors."""
    h, w = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Background: gray (128, 128, 128)
    arr[:, :] = [128, 128, 128]
    # Upper body: rows 20% to 55%, cols 15% to 85%
    u_r0, u_r1 = int(h * 0.20), int(h * 0.55)
    c0, c1 = int(w * 0.15), int(w * 0.85)
    arr[u_r0:u_r1, c0:c1] = upper_color
    # Lower body: rows 55% to 95%, cols 15% to 85%
    l_r0, l_r1 = int(h * 0.55), int(h * 0.95)
    arr[l_r0:l_r1, c0:c1] = lower_color
    return Image.fromarray(arr)


def test_synthetic_images_grouping() -> None:
    """Two identities with identical clothing colors are grouped; a different-colored identity is not."""
    # Identity 1 & 2: Navy upper, khaki lower
    id1_img = _create_synthetic_pedestrian(
        upper_color=(20, 30, 80), lower_color=(190, 170, 120)
    )
    id2_img = _create_synthetic_pedestrian(
        upper_color=(20, 30, 80), lower_color=(190, 170, 120)
    )
    # Identity 3: Bright red upper, black lower
    id3_img = _create_synthetic_pedestrian(
        upper_color=(220, 20, 20), lower_color=(10, 10, 10)
    )

    u1, l1 = compute_clothing_descriptor(id1_img)
    u2, l2 = compute_clothing_descriptor(id2_img)
    u3, l3 = compute_clothing_descriptor(id3_img)

    descs_u = np.array([u1, u2, u3])
    descs_l = np.array([l1, l2, l3])
    np.array([101, 102, 103], dtype=np.int32)

    dist_matrix = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(i + 1, 3):
            d = 0.5 * (
                bhattacharyya_distance(descs_u[i], descs_u[j])
                + bhattacharyya_distance(descs_l[i], descs_l[j])
            )
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Distance between ID 101 and 102 should be 0.0 (identical)
    assert dist_matrix[0, 1] < 1e-4
    # Distance to ID 103 should be substantial
    assert dist_matrix[0, 2] > 0.40
    assert dist_matrix[1, 2] > 0.40

    # Cluster with threshold = 0.10
    from sklearn.cluster import AgglomerativeClustering

    agg = AgglomerativeClustering(
        distance_threshold=0.10,
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
    )
    labels = agg.fit_predict(dist_matrix)
    # ID 101 and 102 share cluster; 103 is separate
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_grouping_determinism() -> None:
    """Complete-linkage agglomerative clustering produces identical results across runs."""
    rng = np.random.RandomState(42)
    n = 20
    mat = rng.uniform(0.1, 0.9, size=(n, n))
    mat = 0.5 * (mat + mat.T)
    np.fill_diagonal(mat, 0.0)
    pids = np.arange(1, n + 1)

    res1 = cluster_lookalike_groups(mat, pids, percentile=10.0, max_group_size=4)
    res2 = cluster_lookalike_groups(mat, pids, percentile=10.0, max_group_size=4)

    assert res1["num_groups"] == res2["num_groups"]
    assert res1["groups"] == res2["groups"]
    assert res1["distance_threshold"] == res2["distance_threshold"]
    assert res1["group_size_histogram"] == res2["group_size_histogram"]


def test_paired_stratified_split() -> None:
    """Paired look-alike split guarantees look-alikes on both enrolled and unknown sides for every group."""
    groups = [[1, 2], [3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16]]
    all_pids = list(range(1, 25))

    for seed in (0, 1, 42):
        enrolled, unknown, extra = partition_lookalike_split(
            groups, all_pids, seed=seed
        )

        assert enrolled.isdisjoint(unknown)
        assert enrolled.isdisjoint(extra)
        assert unknown.isdisjoint(extra)

        for g in groups:
            g_set = set(g)
            g_enrolled = g_set & enrolled
            g_unknown = g_set & unknown
            # Every group must place >=1 in enrolled and >=1 in unknown
            assert len(g_enrolled) >= 1
            assert len(g_unknown) >= 1
            assert g_enrolled | g_unknown == g_set


def test_matched_control_split_counts() -> None:
    """Matched random control partition preserves exact enrolled and unknown identity counts."""
    all_pids = list(range(1, 101))
    enrolled, unknown, extra = partition_matched_control_split(
        num_enrolled=25, num_unknown=15, all_test_pids=all_pids, seed=123
    )
    assert len(enrolled) == 25
    assert len(unknown) == 15
    assert len(extra) == 60
    assert enrolled.isdisjoint(unknown)
    assert enrolled.isdisjoint(extra)


def test_unreachable_tar_returns_none() -> None:
    """When target accuracy cannot reach a TAR target, metric returns None rather than an artificial value."""
    # Target probes only achieve 60% top-1 accuracy (3 correct out of 5)
    tgt_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    tgt_correct = np.array([True, True, True, False, False])  # max TAR = 0.60
    imp_scores = np.array([0.4, 0.3, 0.2])

    metrics = compute_metrics_from_scores_v2(
        target_scores=tgt_scores,
        target_correct_rank1=tgt_correct,
        impostor_scores=imp_scores,
        tar_targets=(0.50, 0.80, 0.90),
    )

    far_dict = metrics["far_at_tar"]
    # 0.50 is reachable
    assert far_dict[0.50] is not None
    # 0.80 and 0.90 exceed max_tar (0.60) and MUST return None
    assert far_dict[0.80] is None
    assert far_dict[0.90] is None
    assert metrics["max_tar"] == 0.60


def test_file_list_verification_fails_on_mismatch(tmp_path: Path) -> None:
    """verify_file_lists raises ValueError on any length, pid, or camid mismatch."""
    # Create mock dataset directory
    train_dir = tmp_path / "bounding_box_train"
    q_dir = tmp_path / "query"
    g_dir = tmp_path / "bounding_box_test"
    train_dir.mkdir(parents=True)
    q_dir.mkdir(parents=True)
    g_dir.mkdir(parents=True)

    # 1 query, 1 gallery
    (q_dir / "0001_c1s1_0001_01.jpg").touch()
    (g_dir / "0001_c2s1_0002_02.jpg").touch()

    # Mismatched query count
    with pytest.raises(ValueError, match="Query file count mismatch"):
        verify_file_lists(
            data_root=tmp_path,
            q_pids_saved=np.array([1, 2], dtype=np.int32),
            q_cams_saved=np.array([1, 1], dtype=np.int32),
            g_pids_saved=np.array([1], dtype=np.int32),
            g_cams_saved=np.array([2], dtype=np.int32),
        )

    # Mismatched query PID
    with pytest.raises(ValueError, match="Query identity IDs in dataset do not match"):
        verify_file_lists(
            data_root=tmp_path,
            q_pids_saved=np.array([999], dtype=np.int32),
            q_cams_saved=np.array([1], dtype=np.int32),
            g_pids_saved=np.array([1], dtype=np.int32),
            g_cams_saved=np.array([2], dtype=np.int32),
        )

    # Mismatched gallery camera ID
    with pytest.raises(ValueError, match="Gallery camera IDs in dataset do not match"):
        verify_file_lists(
            data_root=tmp_path,
            q_pids_saved=np.array([1], dtype=np.int32),
            q_cams_saved=np.array([1], dtype=np.int32),
            g_pids_saved=np.array([1], dtype=np.int32),
            g_cams_saved=np.array([5], dtype=np.int32),  # actual is 2
        )
