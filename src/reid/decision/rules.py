"""Inference-time decision rules for open-set person re-identification.

Implements decision rules to compute probe acceptance scores and identity predictions:
1. raw: Baseline top-1 cosine similarity s1.
2. gap: Margin gap s1 - lambda * s2, where s2 is the maximum similarity to any
   enrolled gallery identity distinct from the top-1 identity. Distractors (pid 0)
   are treated as distinct individuals.
3. znorm: (s1 - mu_q) / sigma_q, where mu_q and sigma_q are the mean and std of probe q's
   top-k cosines against the cohort set.
4. snorm: Symmetric score normalization averaging probe-side and gallery-side z-norms against the cohort.
5. csls: Inductive CSLS: 2 * cos(q, g) - r_k(q) - r_k(g), where r_k(q) and r_k(g) are
   computed strictly from the cohort embeddings.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from reid.eval.metrics import compute_metrics_from_scores_v2


def partition_validation_cohort(
    val_identities: Sequence[int],
    val_split_seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Partition held-out validation identities into two disjoint halves.

    Args:
        val_identities: List or sequence of 100 validation person IDs.
        val_split_seed: Fixed deterministic random seed (default: 42).

    Returns:
        (cohort_a_pids, cohort_b_pids) each of size 50.
    """
    sorted_pids = sorted({int(x) for x in val_identities})
    if len(sorted_pids) < 2:
        raise ValueError(
            f"Need at least 2 validation identities to partition, got {len(sorted_pids)}"
        )
    rng = np.random.RandomState(val_split_seed)
    shuffled = rng.permutation(sorted_pids)
    mid = len(shuffled) // 2
    half_a = sorted([int(x) for x in shuffled[:mid]])
    half_b = sorted([int(x) for x in shuffled[mid:]])
    return half_a, half_b


def compute_cohort_stats(
    feats: np.ndarray,
    cohort_feats: np.ndarray,
    k: int,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of top-k cosine similarities to cohort embeddings.

    Args:
        feats: Query or gallery features, shape (N, D), assumed L2-normalized.
        cohort_feats: Cohort features, shape (M, D), assumed L2-normalized.
        k: Number of nearest cohort neighbors.
        eps: Numerical stability epsilon for standard deviation.

    Returns:
        (means, stds), each of shape (N,).
    """
    if len(cohort_feats) == 0:
        raise ValueError("Cohort features must not be empty.")
    k = min(k, len(cohort_feats))

    # Dot product similarity (N, M)
    sims = np.dot(feats, cohort_feats.T)

    # Top-k largest similarities
    if k == len(cohort_feats):
        topk_sims = sims
    else:
        # Partition along axis 1
        part_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
        topk_sims = np.take_along_axis(sims, part_idx, axis=1)

    means = np.mean(topk_sims, axis=1)
    stds = np.std(topk_sims, axis=1) + eps
    return means, stds


def compute_inductive_csls_r(
    feats: np.ndarray,
    cohort_feats: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute inductive CSLS penalty r_k from cohort embeddings only.

    r_k(x) = mean of top-k cosines of x to the cohort.

    Args:
        feats: Features (N, D), assumed L2-normalized.
        cohort_feats: Cohort features (M, D), assumed L2-normalized.
        k: Number of neighbors.

    Returns:
        r_k array of shape (N,).
    """
    if len(cohort_feats) == 0:
        raise ValueError("Cohort features must not be empty.")
    k = min(k, len(cohort_feats))
    sims = np.dot(feats, cohort_feats.T)
    if k == len(cohort_feats):
        topk_sims = sims
    else:
        part_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
        topk_sims = np.take_along_axis(sims, part_idx, axis=1)
    return np.mean(topk_sims, axis=1)


class BaseDecisionRule:
    """Base class for an open-set inference decision rule."""

    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params = params

    def evaluate_split(
        self,
        q_feats: np.ndarray,
        q_pids: np.ndarray,
        q_cams: np.ndarray,
        g_feats: np.ndarray,
        g_pids: np.ndarray,
        g_cams: np.ndarray,
        enrolled_pids: Set[int],
        unknown_pids: Set[int],
        extra_gallery_pids: Optional[Set[int]] = None,
        cohort_feats: Optional[np.ndarray] = None,
        tar_targets: Sequence[float] = (0.70, 0.80, 0.85, 0.90, 0.95),
        far_targets: Sequence[float] = (0.001, 0.01, 0.05, 0.10),
    ) -> Dict[str, Any]:
        """Evaluate open-set metrics under this decision rule."""
        raise NotImplementedError


class RawRule(BaseDecisionRule):
    """Raw top-1 cosine similarity baseline rule."""

    name: str = "raw"

    def evaluate_split(
        self,
        q_feats: np.ndarray,
        q_pids: np.ndarray,
        q_cams: np.ndarray,
        g_feats: np.ndarray,
        g_pids: np.ndarray,
        g_cams: np.ndarray,
        enrolled_pids: Set[int],
        unknown_pids: Set[int],
        extra_gallery_pids: Optional[Set[int]] = None,
        cohort_feats: Optional[np.ndarray] = None,
        tar_targets: Sequence[float] = (0.70, 0.80, 0.85, 0.90, 0.95),
        far_targets: Sequence[float] = (0.001, 0.01, 0.05, 0.10),
    ) -> Dict[str, Any]:
        valid_g_pids = set(enrolled_pids)
        if extra_gallery_pids is not None:
            valid_g_pids.update(extra_gallery_pids)

        g_mask = np.isin(g_pids, list(valid_g_pids)) | (g_pids == 0)
        sub_g_feats = g_feats[g_mask]
        sub_g_pids = g_pids[g_mask]
        sub_g_cams = g_cams[g_mask]

        q_mask = np.isin(q_pids, list(enrolled_pids | unknown_pids))
        sub_q_feats = q_feats[q_mask]
        sub_q_pids = q_pids[q_mask]
        sub_q_cams = q_cams[q_mask]

        sim_mat = np.dot(sub_q_feats, sub_g_feats.T)

        target_scores = []
        target_correct_rank1 = []
        impostor_scores = []

        for i in range(len(sub_q_pids)):
            pid = sub_q_pids[i]
            cam = sub_q_cams[i]

            valid_mask = ~((sub_g_pids == pid) & (sub_g_cams == cam)) & (
                sub_g_pids != -1
            )
            if not np.any(valid_mask):
                continue

            valid_sims = sim_mat[i][valid_mask]
            valid_match_pids = sub_g_pids[valid_mask]

            best_idx = np.argmax(valid_sims)
            score = float(valid_sims[best_idx])
            is_rank1 = bool(valid_match_pids[best_idx] == pid)

            if pid in enrolled_pids:
                target_scores.append(score)
                target_correct_rank1.append(is_rank1)
            elif pid in unknown_pids:
                impostor_scores.append(score)

        return compute_metrics_from_scores_v2(
            np.array(target_scores, dtype=np.float64),
            np.array(target_correct_rank1, dtype=bool),
            np.array(impostor_scores, dtype=np.float64),
            tar_targets=tar_targets,
            far_targets=far_targets,
        )


class GapRule(BaseDecisionRule):
    """Gap rule: score = s1 - lambda * s2.

    s2 is the maximum similarity to any enrolled gallery instance belonging to
    an identity distinct from the top-1 identity. Distractors (pid 0) are treated
    as distinct unique individuals.
    """

    name: str = "gap"

    def __init__(self, lambda_val: float = 1.0) -> None:
        super().__init__(lambda_val=lambda_val)
        self.lambda_val = float(lambda_val)

    def evaluate_split(
        self,
        q_feats: np.ndarray,
        q_pids: np.ndarray,
        q_cams: np.ndarray,
        g_feats: np.ndarray,
        g_pids: np.ndarray,
        g_cams: np.ndarray,
        enrolled_pids: Set[int],
        unknown_pids: Set[int],
        extra_gallery_pids: Optional[Set[int]] = None,
        cohort_feats: Optional[np.ndarray] = None,
        tar_targets: Sequence[float] = (0.70, 0.80, 0.85, 0.90, 0.95),
        far_targets: Sequence[float] = (0.001, 0.01, 0.05, 0.10),
    ) -> Dict[str, Any]:
        valid_g_pids = set(enrolled_pids)
        if extra_gallery_pids is not None:
            valid_g_pids.update(extra_gallery_pids)

        g_mask = np.isin(g_pids, list(valid_g_pids)) | (g_pids == 0)
        sub_g_feats = g_feats[g_mask]
        sub_g_pids = g_pids[g_mask]
        sub_g_cams = g_cams[g_mask]

        q_mask = np.isin(q_pids, list(enrolled_pids | unknown_pids))
        sub_q_feats = q_feats[q_mask]
        sub_q_pids = q_pids[q_mask]
        sub_q_cams = q_cams[q_mask]

        sim_mat = np.dot(sub_q_feats, sub_g_feats.T)

        target_scores = []
        target_correct_rank1 = []
        impostor_scores = []

        for i in range(len(sub_q_pids)):
            pid = sub_q_pids[i]
            cam = sub_q_cams[i]

            valid_mask = ~((sub_g_pids == pid) & (sub_g_cams == cam)) & (
                sub_g_pids != -1
            )
            if not np.any(valid_mask):
                continue

            valid_sims = sim_mat[i][valid_mask]
            valid_match_pids = sub_g_pids[valid_mask]

            best_idx = np.argmax(valid_sims)
            s1 = float(valid_sims[best_idx])
            top_pid = valid_match_pids[best_idx]
            is_rank1 = bool(top_pid == pid)

            # Determine s2: distractor pid 0 are distinct people
            # If top_pid == 0, that specific distractor index is excluded; other distractors and pids are valid
            # If top_pid > 0, all gallery items with pid != top_pid (including all distractors) are valid
            if top_pid == 0:
                s2_mask = np.ones(len(valid_sims), dtype=bool)
                s2_mask[best_idx] = False
            else:
                s2_mask = valid_match_pids != top_pid

            if np.any(s2_mask):
                s2 = float(np.max(valid_sims[s2_mask]))
            else:
                s2 = -1.0

            score = s1 - (self.lambda_val * s2)

            if pid in enrolled_pids:
                target_scores.append(score)
                target_correct_rank1.append(is_rank1)
            elif pid in unknown_pids:
                impostor_scores.append(score)

        return compute_metrics_from_scores_v2(
            np.array(target_scores, dtype=np.float64),
            np.array(target_correct_rank1, dtype=bool),
            np.array(impostor_scores, dtype=np.float64),
            tar_targets=tar_targets,
            far_targets=far_targets,
        )


class ZNormRule(BaseDecisionRule):
    """Probe-side z-norm rule: score = (s1 - mu_q) / sigma_q against cohort."""

    name: str = "znorm"

    def __init__(self, k: int = 10) -> None:
        super().__init__(k=k)
        self.k = int(k)

    def evaluate_split(
        self,
        q_feats: np.ndarray,
        q_pids: np.ndarray,
        q_cams: np.ndarray,
        g_feats: np.ndarray,
        g_pids: np.ndarray,
        g_cams: np.ndarray,
        enrolled_pids: Set[int],
        unknown_pids: Set[int],
        extra_gallery_pids: Optional[Set[int]] = None,
        cohort_feats: Optional[np.ndarray] = None,
        tar_targets: Sequence[float] = (0.70, 0.80, 0.85, 0.90, 0.95),
        far_targets: Sequence[float] = (0.001, 0.01, 0.05, 0.10),
    ) -> Dict[str, Any]:
        if cohort_feats is None or len(cohort_feats) == 0:
            raise ValueError("ZNormRule requires non-empty cohort_feats.")

        valid_g_pids = set(enrolled_pids)
        if extra_gallery_pids is not None:
            valid_g_pids.update(extra_gallery_pids)

        g_mask = np.isin(g_pids, list(valid_g_pids)) | (g_pids == 0)
        sub_g_feats = g_feats[g_mask]
        sub_g_pids = g_pids[g_mask]
        sub_g_cams = g_cams[g_mask]

        q_mask = np.isin(q_pids, list(enrolled_pids | unknown_pids))
        sub_q_feats = q_feats[q_mask]
        sub_q_pids = q_pids[q_mask]
        sub_q_cams = q_cams[q_mask]

        # Precompute probe-side cohort stats
        q_means, q_stds = compute_cohort_stats(sub_q_feats, cohort_feats, k=self.k)

        sim_mat = np.dot(sub_q_feats, sub_g_feats.T)

        target_scores = []
        target_correct_rank1 = []
        impostor_scores = []

        for i in range(len(sub_q_pids)):
            pid = sub_q_pids[i]
            cam = sub_q_cams[i]

            valid_mask = ~((sub_g_pids == pid) & (sub_g_cams == cam)) & (
                sub_g_pids != -1
            )
            if not np.any(valid_mask):
                continue

            valid_sims = sim_mat[i][valid_mask]
            valid_match_pids = sub_g_pids[valid_mask]

            best_idx = np.argmax(valid_sims)
            s1 = float(valid_sims[best_idx])
            is_rank1 = bool(valid_match_pids[best_idx] == pid)

            score = (s1 - q_means[i]) / q_stds[i]

            if pid in enrolled_pids:
                target_scores.append(score)
                target_correct_rank1.append(is_rank1)
            elif pid in unknown_pids:
                impostor_scores.append(score)

        return compute_metrics_from_scores_v2(
            np.array(target_scores, dtype=np.float64),
            np.array(target_correct_rank1, dtype=bool),
            np.array(impostor_scores, dtype=np.float64),
            tar_targets=tar_targets,
            far_targets=far_targets,
        )


class SNormRule(BaseDecisionRule):
    """Symmetric z-norm rule: score = 0.5 * (znorm_probe + znorm_gallery)."""

    name: str = "snorm"

    def __init__(self, k: int = 10) -> None:
        super().__init__(k=k)
        self.k = int(k)

    def evaluate_split(
        self,
        q_feats: np.ndarray,
        q_pids: np.ndarray,
        q_cams: np.ndarray,
        g_feats: np.ndarray,
        g_pids: np.ndarray,
        g_cams: np.ndarray,
        enrolled_pids: Set[int],
        unknown_pids: Set[int],
        extra_gallery_pids: Optional[Set[int]] = None,
        cohort_feats: Optional[np.ndarray] = None,
        tar_targets: Sequence[float] = (0.70, 0.80, 0.85, 0.90, 0.95),
        far_targets: Sequence[float] = (0.001, 0.01, 0.05, 0.10),
    ) -> Dict[str, Any]:
        if cohort_feats is None or len(cohort_feats) == 0:
            raise ValueError("SNormRule requires non-empty cohort_feats.")

        valid_g_pids = set(enrolled_pids)
        if extra_gallery_pids is not None:
            valid_g_pids.update(extra_gallery_pids)

        g_mask = np.isin(g_pids, list(valid_g_pids)) | (g_pids == 0)
        sub_g_feats = g_feats[g_mask]
        sub_g_pids = g_pids[g_mask]
        sub_g_cams = g_cams[g_mask]

        q_mask = np.isin(q_pids, list(enrolled_pids | unknown_pids))
        sub_q_feats = q_feats[q_mask]
        sub_q_pids = q_pids[q_mask]
        sub_q_cams = q_cams[q_mask]

        # Precompute probe-side and gallery-side cohort stats
        q_means, q_stds = compute_cohort_stats(sub_q_feats, cohort_feats, k=self.k)
        g_means, g_stds = compute_cohort_stats(sub_g_feats, cohort_feats, k=self.k)

        sim_mat = np.dot(sub_q_feats, sub_g_feats.T)

        target_scores = []
        target_correct_rank1 = []
        impostor_scores = []

        for i in range(len(sub_q_pids)):
            pid = sub_q_pids[i]
            cam = sub_q_cams[i]

            valid_mask = ~((sub_g_pids == pid) & (sub_g_cams == cam)) & (
                sub_g_pids != -1
            )
            if not np.any(valid_mask):
                continue

            valid_indices = np.where(valid_mask)[0]
            valid_sims = sim_mat[i][valid_mask]

            best_local_idx = np.argmax(valid_sims)
            best_g_idx = valid_indices[best_local_idx]
            s1 = float(valid_sims[best_local_idx])
            is_rank1 = bool(sub_g_pids[best_g_idx] == pid)

            probe_znorm = (s1 - q_means[i]) / q_stds[i]
            gallery_znorm = (s1 - g_means[best_g_idx]) / g_stds[best_g_idx]
            score = 0.5 * (probe_znorm + gallery_znorm)

            if pid in enrolled_pids:
                target_scores.append(score)
                target_correct_rank1.append(is_rank1)
            elif pid in unknown_pids:
                impostor_scores.append(score)

        return compute_metrics_from_scores_v2(
            np.array(target_scores, dtype=np.float64),
            np.array(target_correct_rank1, dtype=bool),
            np.array(impostor_scores, dtype=np.float64),
            tar_targets=tar_targets,
            far_targets=far_targets,
        )


class CSLSRule(BaseDecisionRule):
    """Inductive CSLS rule: CSLS(q, g) = 2 * cos(q, g) - r_k(q) - r_k(g).

    r_k(q) and r_k(g) are computed strictly from the cohort embeddings.
    Used for both ranking and acceptance.
    """

    name: str = "csls"

    def __init__(self, k: int = 10) -> None:
        super().__init__(k=k)
        self.k = int(k)

    def evaluate_split(
        self,
        q_feats: np.ndarray,
        q_pids: np.ndarray,
        q_cams: np.ndarray,
        g_feats: np.ndarray,
        g_pids: np.ndarray,
        g_cams: np.ndarray,
        enrolled_pids: Set[int],
        unknown_pids: Set[int],
        extra_gallery_pids: Optional[Set[int]] = None,
        cohort_feats: Optional[np.ndarray] = None,
        tar_targets: Sequence[float] = (0.70, 0.80, 0.85, 0.90, 0.95),
        far_targets: Sequence[float] = (0.001, 0.01, 0.05, 0.10),
    ) -> Dict[str, Any]:
        if cohort_feats is None or len(cohort_feats) == 0:
            raise ValueError("CSLSRule requires non-empty cohort_feats.")

        valid_g_pids = set(enrolled_pids)
        if extra_gallery_pids is not None:
            valid_g_pids.update(extra_gallery_pids)

        g_mask = np.isin(g_pids, list(valid_g_pids)) | (g_pids == 0)
        sub_g_feats = g_feats[g_mask]
        sub_g_pids = g_pids[g_mask]
        sub_g_cams = g_cams[g_mask]

        q_mask = np.isin(q_pids, list(enrolled_pids | unknown_pids))
        sub_q_feats = q_feats[q_mask]
        sub_q_pids = q_pids[q_mask]
        sub_q_cams = q_cams[q_mask]

        # Compute inductive hubness penalties from cohort only
        r_q = compute_inductive_csls_r(sub_q_feats, cohort_feats, k=self.k)
        r_g = compute_inductive_csls_r(sub_g_feats, cohort_feats, k=self.k)

        sim_mat = np.dot(sub_q_feats, sub_g_feats.T)

        target_scores = []
        target_correct_rank1 = []
        impostor_scores = []

        for i in range(len(sub_q_pids)):
            pid = sub_q_pids[i]
            cam = sub_q_cams[i]

            valid_mask = ~((sub_g_pids == pid) & (sub_g_cams == cam)) & (
                sub_g_pids != -1
            )
            if not np.any(valid_mask):
                continue

            valid_sims = sim_mat[i][valid_mask]
            valid_r_g = r_g[valid_mask]
            valid_match_pids = sub_g_pids[valid_mask]

            # CSLS scores: 2 * cos(q, g) - r_q - r_g
            csls_sims = 2.0 * valid_sims - r_q[i] - valid_r_g

            best_idx = np.argmax(csls_sims)
            score = float(csls_sims[best_idx])
            is_rank1 = bool(valid_match_pids[best_idx] == pid)

            if pid in enrolled_pids:
                target_scores.append(score)
                target_correct_rank1.append(is_rank1)
            elif pid in unknown_pids:
                impostor_scores.append(score)

        return compute_metrics_from_scores_v2(
            np.array(target_scores, dtype=np.float64),
            np.array(target_correct_rank1, dtype=bool),
            np.array(impostor_scores, dtype=np.float64),
            tar_targets=tar_targets,
            far_targets=far_targets,
        )


def get_all_candidate_rules() -> Dict[str, BaseDecisionRule]:
    """Return dictionary of all pre-declared candidate decision rules."""
    return {
        "raw": RawRule(),
        "gap_lambda_1.0": GapRule(lambda_val=1.0),
        "gap_lambda_0.75": GapRule(lambda_val=0.75),
        "gap_lambda_0.50": GapRule(lambda_val=0.50),
        "gap_lambda_0.25": GapRule(lambda_val=0.25),
        "znorm_k10": ZNormRule(k=10),
        "znorm_k50": ZNormRule(k=50),
        "snorm_k10": SNormRule(k=10),
        "snorm_k50": SNormRule(k=50),
        "csls_k5": CSLSRule(k=5),
        "csls_k10": CSLSRule(k=10),
    }


def select_best_decision_rule_two_fold(
    val_q_feats: np.ndarray,
    val_q_pids: np.ndarray,
    val_q_cams: np.ndarray,
    val_g_feats: np.ndarray,
    val_g_pids: np.ndarray,
    val_g_cams: np.ndarray,
    val_identities: Sequence[int],
    test_pids: Optional[Sequence[int]] = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> Dict[str, Any]:
    """Two-fold offline selection of decision rule on held-out validation set.

    Strictly forbids test peeking: asserts that validation identities have zero
    overlap with test identities.

    Fold 1: Cohort = Half A (50 IDs), Selection = Half B (50 IDs).
    Fold 2: Cohort = Half B (50 IDs), Selection = Half A (50 IDs).

    Selection criterion:
    - Primary: DIR@FAR=5%
    - Secondary: FAR@TAR=80%
    A non-raw rule is selected only if it beats raw on both metrics by more than
    one std of the paired per-seed differences across the 10 fold-seed runs.
    Otherwise, raw is retained.
    """
    val_id_set = {int(x) for x in val_identities}

    # Strict isolation check
    if test_pids is not None:
        test_id_set = {int(x) for x in test_pids}
        overlap = val_id_set & test_id_set
        if overlap:
            raise AssertionError(
                f"[Security Violation] Test identities detected in validation selection! "
                f"Overlap count: {len(overlap)}, sample: {list(overlap)[:5]}"
            )

    half_a, half_b = partition_validation_cohort(val_identities, val_split_seed=42)
    candidate_rules = get_all_candidate_rules()

    folds = [
        ("Fold_1", set(half_a), set(half_b)),
        ("Fold_2", set(half_b), set(half_a)),
    ]

    # Store per-seed metrics: rule_name -> list of (dir5, far80)
    rule_results: Dict[str, List[Dict[str, float]]] = {
        name: [] for name in candidate_rules
    }

    # All val embeddings stacked for cohort lookup
    all_val_feats = np.vstack([val_q_feats, val_g_feats])
    all_val_pids = np.concatenate([val_q_pids, val_g_pids])

    for fold_name, cohort_set, selection_set in folds:
        cohort_mask = np.isin(all_val_pids, list(cohort_set))
        cohort_feats = all_val_feats[cohort_mask]

        sel_all_pids = sorted(selection_set)

        for seed in seeds:
            rng = np.random.RandomState(seed)
            shuffled = rng.permutation(sel_all_pids)
            n_enrolled = len(shuffled) // 2
            enrolled = {int(x) for x in shuffled[:n_enrolled]}
            unknown = {int(x) for x in shuffled[n_enrolled:]}

            for rule_name, rule_obj in candidate_rules.items():
                res = rule_obj.evaluate_split(
                    val_q_feats,
                    val_q_pids,
                    val_q_cams,
                    val_g_feats,
                    val_g_pids,
                    val_g_cams,
                    enrolled_pids=enrolled,
                    unknown_pids=unknown,
                    extra_gallery_pids=None,
                    cohort_feats=cohort_feats,
                    tar_targets=(0.70, 0.80, 0.85, 0.90),
                    far_targets=(0.001, 0.01, 0.05, 0.10),
                )
                dir5 = res["dir_at_far"].get(0.05, 0.0) or 0.0
                far80 = res["far_at_tar"].get(0.80)
                far80_val = 1.0 if far80 is None else float(far80)

                rule_results[rule_name].append(
                    {
                        "fold": fold_name,
                        "seed": seed,
                        "dir5": float(dir5),
                        "far80": float(far80_val),
                    }
                )

    # Compute aggregate and paired differences vs raw
    raw_runs = rule_results["raw"]
    raw_dir5 = np.array([r["dir5"] for r in raw_runs])
    raw_far80 = np.array([r["far80"] for r in raw_runs])

    summary_table = {}
    selected_rule = "raw"
    qualifying_rules = []

    for rule_name, runs in rule_results.items():
        arr_dir5 = np.array([r["dir5"] for r in runs])
        arr_far80 = np.array([r["far80"] for r in runs])

        diff_dir5 = arr_dir5 - raw_dir5  # positive is better
        diff_far80 = raw_far80 - arr_far80  # positive means rule has lower FAR

        mean_diff_dir5 = float(np.mean(diff_dir5))
        std_diff_dir5 = float(np.std(diff_dir5))
        mean_diff_far80 = float(np.mean(diff_far80))
        std_diff_far80 = float(np.std(diff_far80))

        summary_table[rule_name] = {
            "mean_dir5": float(np.mean(arr_dir5)),
            "std_dir5": float(np.std(arr_dir5)),
            "mean_far80": float(np.mean(arr_far80)),
            "std_far80": float(np.std(arr_far80)),
            "paired_diff_dir5_mean": mean_diff_dir5,
            "paired_diff_dir5_std": std_diff_dir5,
            "paired_diff_far80_mean": mean_diff_far80,
            "paired_diff_far80_std": std_diff_far80,
            "beats_raw_both_1std": (
                rule_name != "raw"
                and mean_diff_dir5 > std_diff_dir5
                and mean_diff_far80 > std_diff_far80
            ),
        }

        if summary_table[rule_name]["beats_raw_both_1std"]:
            qualifying_rules.append(rule_name)

    if qualifying_rules:
        # Sort qualifying rules by DIR@FAR=5% descending, then FAR@TAR=80 ascending
        qualifying_rules.sort(
            key=lambda name: (
                summary_table[name]["mean_dir5"],
                -summary_table[name]["mean_far80"],
            ),
            reverse=True,
        )
        selected_rule = qualifying_rules[0]

    return {
        "selected_rule": selected_rule,
        "qualifying_rules": qualifying_rules,
        "summary": summary_table,
        "half_a_count": len(half_a),
        "half_b_count": len(half_b),
    }
