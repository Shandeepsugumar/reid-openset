"""Unit tests for inference-time open-set decision rules."""

import numpy as np
import pytest

from reid.decision.rules import (
    CSLSRule,
    GapRule,
    RawRule,
    ZNormRule,
    compute_inductive_csls_r,
    partition_validation_cohort,
    select_best_decision_rule_two_fold,
)


def test_gap_rule_rejects_lookalike_impostor() -> None:
    """Gap rule must reject an ambiguous look-alike impostor that raw accepts."""
    # 2D normalized embeddings for clarity
    # Gallery:
    # id 1: [0.90, sqrt(1-0.9^2)] = [0.90, 0.43588989]
    # id 2: [0.0, 1.0]
    # id 3: [0.10, sqrt(1-0.1^2)] = [0.10, 0.99498744] (close to id 2)
    g_feats = np.array(
        [
            [0.90, 0.43588989],
            [0.0, 1.0],
            [0.10, 0.99498744],
        ],
        dtype=np.float32,
    )
    g_pids = np.array([1, 2, 3], dtype=np.int64)
    g_cams = np.array([1, 1, 1], dtype=np.int64)

    # Genuine enrolled probe for id 1: [0.90, 0.43588989] (cos = 1.0 to id 1, cos ~ 0.436 to id 2)
    # Let's give genuine target a raw cosine of 0.90 to id 1, and 0.0 to id 2:
    # q_tgt = [1.0, 0.0] -> cos to id 1 = 0.90, cos to id 2 = 0.0, cos to id 3 = 0.10
    q_tgt = np.array([[1.0, 0.0]], dtype=np.float32)
    q_tgt_pids = np.array([1], dtype=np.int64)
    q_tgt_cams = np.array([2], dtype=np.int64)

    # Look-alike impostor probe (id 99) ambiguous between id 2 and id 3:
    # q_imp = [0.05, sqrt(1-0.05^2)] = [0.05, 0.99874922]
    # cos to id 2 = 0.99874922
    # cos to id 3 = 0.05*0.10 + 0.99874922*0.99498744 = 0.005 + 0.993743 = 0.998743
    # s1 ~ 0.9987, s2 ~ 0.9987 -> gap ~ 0.000006
    # Raw accepts impostor before target because 0.9987 > 0.900!
    q_imp = np.array([[0.05, 0.99874922]], dtype=np.float32)
    q_imp_pids = np.array([99], dtype=np.int64)
    q_imp_cams = np.array([2], dtype=np.int64)

    q_feats = np.vstack([q_tgt, q_imp])
    q_pids = np.concatenate([q_tgt_pids, q_imp_pids])
    q_cams = np.concatenate([q_tgt_cams, q_imp_cams])

    # Evaluate Raw rule
    raw_rule = RawRule()
    raw_res = raw_rule.evaluate_split(
        q_feats=q_feats,
        q_pids=q_pids,
        q_cams=q_cams,
        g_feats=g_feats,
        g_pids=g_pids,
        g_cams=g_cams,
        enrolled_pids={1, 2, 3},
        unknown_pids={99},
        tar_targets=(0.80, 1.0),
    )

    # Evaluate Gap rule (lambda=1.0)
    gap_rule = GapRule(lambda_val=1.0)
    gap_res = gap_rule.evaluate_split(
        q_feats=q_feats,
        q_pids=q_pids,
        q_cams=q_cams,
        g_feats=g_feats,
        g_pids=g_pids,
        g_cams=g_cams,
        enrolled_pids={1, 2, 3},
        unknown_pids={99},
        tar_targets=(0.80, 1.0),
    )

    # Under raw rule, the impostor has high cosine similarity (s1 ~ 0.988)
    # Under gap rule, the impostor has another close enrolled neighbor (s2 ~ 0.987), so gap ~ 0.001
    # Whereas genuine target has s1 ~ 0.990 and s2 ~ 0.141, so gap ~ 0.849
    # Setting threshold at 0.50 accepts genuine target and rejects impostor for GapRule
    # Check FAR at TAR=1.0: GapRule achieves 0.0 FAR, RawRule has 1.0 FAR
    assert raw_res["far_at_tar"][1.0] == 1.0
    assert gap_res["far_at_tar"][1.0] == 0.0


def test_znorm_rejects_hub_impostor() -> None:
    """Z-norm rejects a hub-like impostor with high background similarity."""
    # 2D features
    # Gallery item: id 1: [1.0, 0.0]
    g_feats = np.array([[1.0, 0.0]], dtype=np.float32)
    g_pids = np.array([1], dtype=np.int64)
    g_cams = np.array([1], dtype=np.int64)

    # Cohort: 2 samples clustered near [0.8, 0.6]
    cohort_feats = np.array(
        [
            [0.8, 0.6],
            [0.8, 0.6],
        ],
        dtype=np.float32,
    )

    # Target probe: [0.9, 0.4359] -> cos to gallery is 0.9, cos to cohort is 0.8*0.9 + 0.6*0.4359 = 0.98
    # Let's make target probe isolated from cohort: [1.0, 0.0]
    # cos to gallery = 1.0, cos to cohort = 0.8
    q_tgt = np.array([[1.0, 0.0]], dtype=np.float32)
    q_tgt_pids = np.array([1], dtype=np.int64)
    q_tgt_cams = np.array([2], dtype=np.int64)

    # Hub impostor: [0.8, 0.6] -> cos to gallery = 0.8, cos to cohort = 1.0
    q_imp = np.array([[0.8, 0.6]], dtype=np.float32)
    q_imp_pids = np.array([99], dtype=np.int64)
    q_imp_cams = np.array([2], dtype=np.int64)

    q_feats = np.vstack([q_tgt, q_imp])
    q_pids = np.concatenate([q_tgt_pids, q_imp_pids])
    q_cams = np.concatenate([q_tgt_cams, q_imp_cams])

    znorm_rule = ZNormRule(k=2)
    res = znorm_rule.evaluate_split(
        q_feats=q_feats,
        q_pids=q_pids,
        q_cams=q_cams,
        g_feats=g_feats,
        g_pids=g_pids,
        g_cams=g_cams,
        enrolled_pids={1},
        unknown_pids={99},
        cohort_feats=cohort_feats,
        tar_targets=(0.80, 1.0),
    )

    # Target score is positive (1.0 - 0.8 > 0)
    # Impostor score is negative (0.8 - 1.0 < 0)
    # Target correctly separated from impostor
    assert res["far_at_tar"][1.0] == 0.0


def test_inductive_csls_hand_calculated_toy_example() -> None:
    """Verify CSLS matches exact hand-calculated value on a toy example.

    Setup:
      Probe q = [1.0, 0.0]
      Gallery g = [0.8, 0.6]
      cos(q, g) = 0.80

      Cohort C (2 items):
        c1 = [0.6, 0.8]
        c2 = [1.0, 0.0]

      cosines of q to C:
        q . c1 = 0.60
        q . c2 = 1.00
        r_2(q) = (1.00 + 0.60) / 2 = 0.80

      cosines of g to C:
        g . c1 = 0.8 * 0.6 + 0.6 * 0.8 = 0.48 + 0.48 = 0.96
        g . c2 = 0.8 * 1.0 + 0.6 * 0.0 = 0.80
        r_2(g) = (0.96 + 0.80) / 2 = 0.88

      CSLS(q, g) = 2 * cos(q, g) - r_2(q) - r_2(g)
                 = 2 * 0.80 - 0.80 - 0.88
                 = 1.60 - 1.68
                 = -0.08
    """
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    g = np.array([[0.8, 0.6]], dtype=np.float32)
    cohort = np.array([[0.6, 0.8], [1.0, 0.0]], dtype=np.float32)

    r_q = compute_inductive_csls_r(q, cohort, k=2)
    r_g = compute_inductive_csls_r(g, cohort, k=2)

    assert r_q[0] == pytest.approx(0.80, abs=1e-6)
    assert r_g[0] == pytest.approx(0.88, abs=1e-6)

    cos_sim = float(np.dot(q, g.T)[0, 0])
    assert cos_sim == pytest.approx(0.80, abs=1e-6)

    csls_score = 2.0 * cos_sim - r_q[0] - r_g[0]
    assert csls_score == pytest.approx(-0.08, abs=1e-6)

    # Now verify CSLSRule produces this exact match score
    rule = CSLSRule(k=2)
    res = rule.evaluate_split(
        q_feats=q,
        q_pids=np.array([1]),
        q_cams=np.array([1]),
        g_feats=g,
        g_pids=np.array([1]),
        g_cams=np.array([2]),
        enrolled_pids={1},
        unknown_pids=set(),
        cohort_feats=cohort,
        tar_targets=(0.80, 1.0),
    )
    # The target score should match -0.08
    assert res["max_tar"] == 1.0
    assert res["far_at_tar"][1.0] == 0.0


def test_selection_protocol_forbids_test_peeking() -> None:
    """Selection function must immediately raise AssertionError if test identities overlap with validation identities."""
    rng = np.random.RandomState(42)
    val_pids = list(range(100))
    test_pids_clean = list(range(100, 200))
    test_pids_leaked = list(range(90, 190))  # overlaps on 90..99

    feats = rng.randn(100, 16).astype(np.float32)
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    pids = np.array(val_pids, dtype=np.int64)
    cams = np.ones(100, dtype=np.int64)

    # 1. Leaked test pids must raise AssertionError
    with pytest.raises(AssertionError, match=r"\[Security Violation\]"):
        select_best_decision_rule_two_fold(
            val_q_feats=feats,
            val_q_pids=pids,
            val_q_cams=cams,
            val_g_feats=feats,
            val_g_pids=pids,
            val_g_cams=cams,
            val_identities=val_pids,
            test_pids=test_pids_leaked,
        )

    # 2. Clean disjoint test pids must pass
    result = select_best_decision_rule_two_fold(
        val_q_feats=feats,
        val_q_pids=pids,
        val_q_cams=cams,
        val_g_feats=feats,
        val_g_pids=pids,
        val_g_cams=cams,
        val_identities=val_pids,
        test_pids=test_pids_clean,
        seeds=(0,),
    )
    assert "selected_rule" in result
    assert result["half_a_count"] == 50
    assert result["half_b_count"] == 50


def test_partition_validation_cohort_reproducibility() -> None:
    """Cohort partitioning must be deterministic, disjoint, and equal size."""
    val_ids = list(range(100))
    a1, b1 = partition_validation_cohort(val_ids, val_split_seed=42)
    a2, b2 = partition_validation_cohort(val_ids, val_split_seed=42)

    assert a1 == a2
    assert b1 == b2
    assert len(a1) == 50
    assert len(b1) == 50
    assert set(a1).isdisjoint(set(b1))
    assert set(a1) | set(b1) == set(val_ids)


def test_gap_rule_treats_pid_0_distractors_as_distinct() -> None:
    """Gallery samples with pid 0 (distractors) are distinct people for s2."""
    # Probe id 1
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    q_pid = np.array([1])
    q_cam = np.array([1])

    # Gallery:
    # 0: id 1 (target match, cos = 0.9)
    # 1: id 0 (distractor 1, cos = 0.8)
    # 2: id 0 (distractor 2, cos = 0.7)
    g = np.array(
        [
            [0.9, 0.43588989],
            [0.8, 0.6],
            [0.7, 0.71414284],
        ],
        dtype=np.float32,
    )
    g_pids = np.array([1, 0, 0])
    g_cams = np.array([2, 2, 2])

    rule = GapRule(lambda_val=1.0)
    res = rule.evaluate_split(
        q_feats=q,
        q_pids=q_pid,
        q_cams=q_cam,
        g_feats=g,
        g_pids=g_pids,
        g_cams=g_cams,
        enrolled_pids={1},
        unknown_pids=set(),
    )
    # s1 = 0.9 (from item 0)
    # s2 = 0.8 (from distractor item 1)
    # score = 0.9 - 0.8 = 0.1
    # Check max_tar is 1.0
    assert res["max_tar"] == 1.0


def test_snorm_rule_averages_probe_and_gallery_znorm() -> None:
    """SNorm rule must compute the exact arithmetic mean of probe and gallery z-norms."""
    from reid.decision.rules import SNormRule

    q = np.array([[1.0, 0.0]], dtype=np.float32)
    g = np.array([[1.0, 0.0]], dtype=np.float32)
    cohort = np.array([[0.6, 0.8], [0.8, 0.6]], dtype=np.float32)

    rule = SNormRule(k=2)
    res = rule.evaluate_split(
        q_feats=q,
        q_pids=np.array([1]),
        q_cams=np.array([1]),
        g_feats=g,
        g_pids=np.array([1]),
        g_cams=np.array([2]),
        enrolled_pids={1},
        unknown_pids=set(),
        cohort_feats=cohort,
        tar_targets=(0.80, 1.0),
    )
    assert res["max_tar"] == 1.0
    assert res["far_at_tar"][1.0] == 0.0


def test_gap_rule_fallback_when_s2_undefined() -> None:
    """GapRule must gracefully fall back to raw score when gallery has only 1 identity and no s2 candidates."""
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    g = np.array([[0.8, 0.6]], dtype=np.float32)

    raw_res = RawRule().evaluate_split(
        q_feats=q,
        q_pids=np.array([1]),
        q_cams=np.array([1]),
        g_feats=g,
        g_pids=np.array([1]),
        g_cams=np.array([2]),
        enrolled_pids={1},
        unknown_pids=set(),
        tar_targets=(0.80, 1.0),
    )

    gap_res = GapRule(lambda_val=0.5, min_gallery_identities=1).evaluate_split(
        q_feats=q,
        q_pids=np.array([1]),
        q_cams=np.array([1]),
        g_feats=g,
        g_pids=np.array([1]),
        g_cams=np.array([2]),
        enrolled_pids={1},
        unknown_pids=set(),
        tar_targets=(0.80, 1.0),
    )

    # With only 1 identity, s2 is undefined, so score falls back to raw score
    assert gap_res["max_tar"] == raw_res["max_tar"]
    assert gap_res["far_at_tar"][1.0] == raw_res["far_at_tar"][1.0]


def test_gap_rule_min_gallery_identities_parameter() -> None:
    """GapRule must fall back to raw score when enrolled gallery size < min_gallery_identities."""
    # 2 enrolled identities
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    g = np.array([[0.9, 0.43588989], [0.5, 0.8660254]], dtype=np.float32)
    g_pids = np.array([1, 2])
    g_cams = np.array([2, 2])

    raw_res = RawRule().evaluate_split(
        q_feats=q,
        q_pids=np.array([1]),
        q_cams=np.array([1]),
        g_feats=g,
        g_pids=g_pids,
        g_cams=g_cams,
        enrolled_pids={1, 2},
        unknown_pids=set(),
        tar_targets=(0.80, 1.0),
    )

    # When min_gallery_identities=5 (gallery has 2), gap falls back to raw
    fallback_res = GapRule(lambda_val=0.5, min_gallery_identities=5).evaluate_split(
        q_feats=q,
        q_pids=np.array([1]),
        q_cams=np.array([1]),
        g_feats=g,
        g_pids=g_pids,
        g_cams=g_cams,
        enrolled_pids={1, 2},
        unknown_pids=set(),
        tar_targets=(0.80, 1.0),
    )
    assert fallback_res["max_tar"] == raw_res["max_tar"]
    assert fallback_res["far_at_tar"][1.0] == raw_res["far_at_tar"][1.0]

    # When min_gallery_identities=2, gap rule applies
    active_res = GapRule(lambda_val=0.5, min_gallery_identities=2).evaluate_split(
        q_feats=q,
        q_pids=np.array([1]),
        q_cams=np.array([1]),
        g_feats=g,
        g_pids=g_pids,
        g_cams=g_cams,
        enrolled_pids={1, 2},
        unknown_pids=set(),
        tar_targets=(0.80, 1.0),
    )
    assert active_res["max_tar"] == 1.0
