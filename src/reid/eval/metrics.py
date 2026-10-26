"""Evaluation metrics for person re-identification (closed-set and open-set)."""

from typing import Dict, Tuple
import numpy as np


def compute_closed_set_cmc_and_map(
    q_feats: np.ndarray,
    q_pids: np.ndarray,
    q_cams: np.ndarray,
    g_feats: np.ndarray,
    g_pids: np.ndarray,
    g_cams: np.ndarray,
    max_rank: int = 20,
) -> Tuple[np.ndarray, float]:
    """Market-1501 standard closed-set evaluation.

    Discards gallery samples with identical identity and camera (junk condition 1),
    and background/junk distractors with identity -1 (junk condition 2).

    Args:
        q_feats: Query embeddings array (N_q, D), assumed L2-normalized.
        q_pids: Query identity labels (N_q,).
        q_cams: Query camera IDs (N_q,).
        g_feats: Gallery embeddings array (N_g, D), assumed L2-normalized.
        g_pids: Gallery identity labels (N_g,).
        g_cams: Gallery camera IDs (N_g,).
        max_rank: Maximum rank for CMC calculation.

    Returns:
        Tuple of (mean_cmc, mAP) where mean_cmc is array of length max_rank.
    """
    sim_mat = np.dot(q_feats, g_feats.T)
    num_queries = len(q_pids)

    all_cmc = []
    all_ap = []

    for i in range(num_queries):
        pid = q_pids[i]
        cam = q_cams[i]

        order = np.argsort(-sim_mat[i])
        valid_mask = ~((g_pids[order] == pid) & (g_cams[order] == cam)) & (g_pids[order] != -1)
        order = order[valid_mask]

        matches = (g_pids[order] == pid).astype(np.int32)
        total_positives = np.sum(matches)
        if total_positives == 0:
            continue

        cmc = np.zeros(max_rank)
        first_match_idx = np.where(matches == 1)[0]
        if len(first_match_idx) > 0 and first_match_idx[0] < max_rank:
            cmc[first_match_idx[0]:] = 1.0
        all_cmc.append(cmc)

        cum_matches = np.cumsum(matches)
        ranks = np.arange(1, len(matches) + 1)
        precisions = cum_matches / ranks
        ap = np.sum(precisions * matches) / total_positives
        all_ap.append(ap)

    mean_cmc = np.mean(all_cmc, axis=0) if all_cmc else np.zeros(max_rank)
    mAP = float(np.mean(all_ap)) if all_ap else 0.0
    return mean_cmc, mAP


def compute_openset_tar_far(
    q_feats: np.ndarray,
    q_pids: np.ndarray,
    q_cams: np.ndarray,
    g_feats: np.ndarray,
    g_pids: np.ndarray,
    g_cams: np.ndarray,
    known_ratio: float = 0.5,
    seed: int = 42,
    num_thresholds: int = 500,
) -> Dict:
    """Open-set evaluation protocol partitioning queries into Enrolled vs Unknown probes.

    Args:
        q_feats: Query embeddings (N_q, D).
        q_pids: Query identity labels (N_q,).
        q_cams: Query camera IDs (N_q,).
        g_feats: Gallery embeddings (N_g, D).
        g_pids: Gallery identity labels (N_g,).
        g_cams: Gallery camera IDs (N_g,).
        known_ratio: Fraction of query identities enrolled into gallery (default: 0.5).
        seed: Random seed for deterministic partition.
        num_thresholds: Number of threshold evaluation points between -0.2 and 1.0.

    Returns:
        Dictionary containing thresholds, FAR, TAR curves, and operating points.
    """
    np.random.seed(seed)
    unique_q_pids = np.unique(q_pids)
    np.random.shuffle(unique_q_pids)

    n_known = int(len(unique_q_pids) * known_ratio)
    enrolled_pids = set(unique_q_pids[:n_known])
    unknown_pids = set(unique_q_pids[n_known:])

    # Gallery retains enrolled identities + distractors/junk. Remove impostor identities.
    gallery_mask = np.array([pid not in unknown_pids for pid in g_pids])
    sub_g_feats = g_feats[gallery_mask]
    sub_g_pids = g_pids[gallery_mask]
    sub_g_cams = g_cams[gallery_mask]

    sim_mat = np.dot(q_feats, sub_g_feats.T)

    target_scores = []
    target_correct_rank1 = []
    impostor_scores = []

    for i in range(len(q_pids)):
        pid = q_pids[i]
        cam = q_cams[i]

        valid_mask = ~((sub_g_pids == pid) & (sub_g_cams == cam)) & (sub_g_pids != -1)
        if not np.any(valid_mask):
            continue

        valid_sims = sim_mat[i][valid_mask]
        valid_pids = sub_g_pids[valid_mask]
        best_idx = np.argmax(valid_sims)
        max_score = valid_sims[best_idx]
        is_rank1 = (valid_pids[best_idx] == pid)

        if pid in enrolled_pids:
            target_scores.append(max_score)
            target_correct_rank1.append(is_rank1)
        elif pid in unknown_pids:
            impostor_scores.append(max_score)

    target_scores = np.array(target_scores)
    target_correct_rank1 = np.array(target_correct_rank1)
    impostor_scores = np.array(impostor_scores)

    thresholds = np.linspace(-0.2, 1.0, num_thresholds)
    far_list, tar_list = [], []

    for tau in thresholds:
        far = float(np.mean(impostor_scores >= tau)) if len(impostor_scores) > 0 else 0.0
        tar = (
            float(np.mean((target_scores >= tau) & target_correct_rank1))
            if len(target_scores) > 0
            else 0.0
        )
        far_list.append(far)
        tar_list.append(tar)

    far_arr = np.array(far_list)
    tar_arr = np.array(tar_list)

    def get_far_at_tar(desired_tar: float) -> float:
        valid_indices = np.where(tar_arr >= desired_tar)[0]
        if len(valid_indices) == 0:
            return 1.0
        return float(np.min(far_arr[valid_indices]))

    return {
        "thresholds": thresholds.tolist(),
        "far": far_arr.tolist(),
        "tar": tar_arr.tolist(),
        "far_at_tar_80": get_far_at_tar(0.80),
        "far_at_tar_90": get_far_at_tar(0.90),
        "far_at_tar_95": get_far_at_tar(0.95),
        "num_enrolled_probes": len(target_scores),
        "num_unknown_probes": len(impostor_scores),
    }
