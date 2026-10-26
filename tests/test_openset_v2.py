"""Unit tests for Evaluator v2 open-set ReID metrics."""

import numpy as np
import pytest

from reid.eval.metrics import (
    compute_metrics_from_scores_v2,
    compute_openset_metrics_v2,
)


def test_perfect_separation():
    """When target and impostor score distributions are disjoint with 100% Rank-1."""
    tgt_scores = np.array([0.85, 0.90, 0.92, 0.88, 0.95])
    tgt_correct = np.array([True, True, True, True, True])
    imp_scores = np.array([0.10, 0.15, 0.20, 0.05, 0.12])

    res = compute_metrics_from_scores_v2(
        target_scores=tgt_scores,
        target_correct_rank1=tgt_correct,
        impostor_scores=imp_scores,
        tar_targets=(0.70, 0.80, 0.85, 0.90),
        far_targets=(0.001, 0.01, 0.05, 0.10),
    )

    assert res["max_tar"] == pytest.approx(1.0)
    for t in (0.70, 0.80, 0.85, 0.90):
        assert res["far_at_tar"][t] == pytest.approx(0.0)
    for f in (0.001, 0.01, 0.05, 0.10):
        assert res["dir_at_far"][f] == pytest.approx(1.0)
    assert res["auc"] == pytest.approx(1.0)
    assert res["eer"] == pytest.approx(0.0)


def test_fully_overlapping():
    """When target and impostor score distributions are identical."""
    tgt_scores = np.array([0.5, 0.5, 0.5, 0.5])
    tgt_correct = np.array([True, True, True, True])
    imp_scores = np.array([0.5, 0.5, 0.5, 0.5])

    res = compute_metrics_from_scores_v2(
        target_scores=tgt_scores,
        target_correct_rank1=tgt_correct,
        impostor_scores=imp_scores,
    )

    assert res["auc"] == pytest.approx(0.5, abs=1e-3)
    assert res["eer"] == pytest.approx(0.5, abs=1e-3)


def test_unreachable_tar_returns_none_and_na():
    """When target TAR cannot be achieved because Rank-1 identification is capped."""
    # 10 enrolled targets, but only 7 are correctly identified at Rank-1 -> max_tar = 0.70
    tgt_scores = np.array([0.9] * 7 + [0.85] * 3)
    tgt_correct = np.array([True] * 7 + [False] * 3)
    imp_scores = np.array([0.2] * 10)

    res = compute_metrics_from_scores_v2(
        target_scores=tgt_scores,
        target_correct_rank1=tgt_correct,
        impostor_scores=imp_scores,
        tar_targets=(0.70, 0.80, 0.85, 0.90, 0.95),
    )

    assert res["max_tar"] == pytest.approx(0.70)
    # TAR = 70% is reachable
    assert res["far_at_tar"][0.70] is not None
    assert res["far_at_tar"][0.70] == pytest.approx(0.0)

    # TAR = 80%, 85%, 90%, 95% exceed 0.70 and must return None, NEVER 1.0!
    assert res["far_at_tar"][0.80] is None
    assert res["far_at_tar"][0.85] is None
    assert res["far_at_tar"][0.90] is None
    assert res["far_at_tar"][0.95] is None


def test_known_auc_analytical():
    """Verify AUC against hand-calculated Wilcoxon-Mann-Whitney rank statistic."""
    # Target scores: [0.8, 0.6] (both rank-1 correct)
    # Impostor scores: [0.7, 0.4]
    # Concordant pairs: (0.8, 0.7)=1, (0.8, 0.4)=1, (0.6, 0.7)=0, (0.6, 0.4)=1 -> AUC = 3/4 = 0.75
    tgt_scores = np.array([0.8, 0.6])
    tgt_correct = np.array([True, True])
    imp_scores = np.array([0.7, 0.4])

    res = compute_metrics_from_scores_v2(
        target_scores=tgt_scores,
        target_correct_rank1=tgt_correct,
        impostor_scores=imp_scores,
    )
    assert res["auc"] == pytest.approx(0.75, abs=1e-5)


def test_multiseed_openset_v2_synthetic():
    """Test full multi-seed workflow on synthetic embeddings."""
    np.random.seed(42)
    dim = 16

    # 10 identities, 2 cameras, 4 samples each
    q_feats, q_pids, q_cams = [], [], []
    g_feats, g_pids, g_cams = [], [], []

    for pid in range(10):
        center = np.random.randn(dim)
        center /= np.linalg.norm(center)
        for cam in [1, 2]:
            q_vec = center + 0.05 * np.random.randn(dim)
            q_vec /= np.linalg.norm(q_vec)
            q_feats.append(q_vec)
            q_pids.append(pid)
            q_cams.append(cam)

            g_vec = center + 0.05 * np.random.randn(dim)
            g_vec /= np.linalg.norm(g_vec)
            g_feats.append(g_vec)
            g_pids.append(pid)
            g_cams.append(cam)

    q_feats = np.array(q_feats)
    q_pids = np.array(q_pids)
    q_cams = np.array(q_cams)
    g_feats = np.array(g_feats)
    g_pids = np.array(g_pids)
    g_cams = np.array(g_cams)

    v2_results = compute_openset_metrics_v2(
        q_feats=q_feats,
        q_pids=q_pids,
        q_cams=q_cams,
        g_feats=g_feats,
        g_pids=g_pids,
        g_cams=g_cams,
        known_ratio=0.5,
        seeds=[0, 1, 2],
    )

    assert v2_results["protocol"] == "openset_v2"
    assert len(v2_results["seeds"]) == 3
    assert len(v2_results["per_seed_results"]) == 3
    summary = v2_results["summary"]
    assert "max_tar" in summary
    assert "auc" in summary
    assert "eer" in summary
    assert 0.0 <= summary["max_tar"]["mean"] <= 1.0
    assert 0.0 <= summary["auc"]["mean"] <= 1.0
    assert 0.0 <= summary["eer"]["mean"] <= 1.0
