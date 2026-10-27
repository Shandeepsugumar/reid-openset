"""Evaluation metrics for person re-identification (closed-set and open-set)."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

matplotlib.use("Agg")


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
        valid_mask = ~((g_pids[order] == pid) & (g_cams[order] == cam)) & (
            g_pids[order] != -1
        )
        order = order[valid_mask]

        matches = (g_pids[order] == pid).astype(np.int32)
        total_positives = np.sum(matches)
        if total_positives == 0:
            continue

        cmc = np.zeros(max_rank)
        first_match_idx = np.where(matches == 1)[0]
        if len(first_match_idx) > 0 and first_match_idx[0] < max_rank:
            cmc[first_match_idx[0] :] = 1.0
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
        is_rank1 = valid_pids[best_idx] == pid

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
        far = (
            float(np.mean(impostor_scores >= tau)) if len(impostor_scores) > 0 else 0.0
        )
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


def compute_metrics_from_scores_v2(
    target_scores: np.ndarray,
    target_correct_rank1: np.ndarray,
    impostor_scores: np.ndarray,
    tar_targets: Sequence[float] = (0.70, 0.80, 0.85, 0.90, 0.95),
    far_targets: Sequence[float] = (0.001, 0.01, 0.05, 0.10),
) -> Dict[str, Any]:
    """Compute open-set evaluation metrics directly from probe match scores.

    Thresholds are derived from the exact sorted scores rather than an arbitrary
    linspace grid. If a TAR target exceeds the maximum achievable Rank-1 identification
    accuracy, it returns None (unreachable), avoiding the legacy 100% artifact.

    Args:
        target_scores: Max gallery similarity scores for enrolled probes (N_tgt,).
        target_correct_rank1: Boolean array whether top-1 retrieval was correct identity (N_tgt,).
        impostor_scores: Max gallery similarity scores for unknown/impostor probes (N_imp,).
        tar_targets: Target TAR thresholds to measure FAR at (default: 0.70, 0.80, 0.85, 0.90).
        far_targets: Target FAR thresholds to measure DIR at (default: 0.001, 0.01, 0.05, 0.10).

    Returns:
        Dict containing exact curves, FAR@TAR, DIR@FAR, max_tar, AUC, and EER.
    """
    target_scores = np.asarray(target_scores, dtype=np.float64)
    target_correct_rank1 = np.asarray(target_correct_rank1, dtype=bool)
    impostor_scores = np.asarray(impostor_scores, dtype=np.float64)

    n_tgt = len(target_scores)
    n_imp = len(impostor_scores)

    max_tar = float(np.mean(target_correct_rank1)) if n_tgt > 0 else 0.0

    if n_tgt > 0 or n_imp > 0:
        all_scores = np.concatenate([target_scores, impostor_scores])
        unique_scores = np.sort(np.unique(all_scores))[::-1]
        thresholds = np.concatenate(
            [[unique_scores[0] + 1e-5], unique_scores, [unique_scores[-1] - 1e-5]]
        )
    else:
        thresholds = np.array([0.0])

    sorted_imp = np.sort(impostor_scores)
    sorted_tgt_correct = np.sort(target_scores[target_correct_rank1])

    far_curve = (
        (n_imp - np.searchsorted(sorted_imp, thresholds, side="left")) / float(n_imp)
        if n_imp > 0
        else np.zeros_like(thresholds)
    )
    tar_curve = (
        (
            len(sorted_tgt_correct)
            - np.searchsorted(sorted_tgt_correct, thresholds, side="left")
        )
        / float(n_tgt)
        if n_tgt > 0
        else np.zeros_like(thresholds)
    )

    # FAR at TAR
    far_at_tar: Dict[float, Optional[float]] = {}
    for t in tar_targets:
        if t > max_tar:
            far_at_tar[t] = None
        else:
            valid = np.where(tar_curve >= t)[0]
            far_at_tar[t] = float(np.min(far_curve[valid])) if len(valid) > 0 else None

    # DIR at FAR
    dir_at_far: Dict[float, float] = {}
    for f in far_targets:
        valid = np.where(far_curve <= f)[0]
        dir_at_far[f] = float(np.max(tar_curve[valid])) if len(valid) > 0 else 0.0

    # Binary detection metrics: enrolled (1) vs impostor (0)
    if n_tgt > 0 and n_imp > 0:
        y_true = np.concatenate([np.ones(n_tgt), np.zeros(n_imp)])
        y_scores = np.concatenate([target_scores, impostor_scores])
        try:
            auc = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            auc = 0.5

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fnr = 1.0 - tpr
        diffs = fpr - fnr
        sign_changes = np.where(np.diff(np.sign(diffs)))[0]
        if len(sign_changes) > 0:
            i = sign_changes[0]
            d0 = fnr[i] - fpr[i]
            d1 = fpr[i + 1] - fnr[i + 1]
            if (d0 + d1) > 0:
                eer = float(fpr[i] + (d0 / (d0 + d1)) * (fpr[i + 1] - fpr[i]))
            else:
                eer = float((fpr[i] + fnr[i]) / 2.0)
        else:
            idx = int(np.argmin(np.abs(diffs)))
            eer = float((fpr[idx] + fnr[idx]) / 2.0)
    else:
        auc = 0.5
        eer = 0.5

    return {
        "max_tar": max_tar,
        "far_at_tar": far_at_tar,
        "dir_at_far": dir_at_far,
        "auc": auc,
        "eer": eer,
        "thresholds": thresholds,
        "far_curve": far_curve,
        "tar_curve": tar_curve,
        "num_enrolled_probes": n_tgt,
        "num_impostor_probes": n_imp,
    }


def compute_openset_single_seed_v2(
    q_feats: np.ndarray,
    q_pids: np.ndarray,
    q_cams: np.ndarray,
    g_feats: np.ndarray,
    g_pids: np.ndarray,
    g_cams: np.ndarray,
    known_ratio: float = 0.5,
    seed: int = 0,
    tar_targets: Sequence[float] = (0.70, 0.80, 0.85, 0.90, 0.95),
    far_targets: Sequence[float] = (0.001, 0.01, 0.05, 0.10),
) -> Dict[str, Any]:
    """Evaluate open-set ReID on a single deterministic seed partition."""
    rng = np.random.RandomState(seed)
    unique_q_pids = np.unique(q_pids)
    shuffled_pids = rng.permutation(unique_q_pids)
    n_known = int(len(shuffled_pids) * known_ratio)
    enrolled_pids = set(shuffled_pids[:n_known])
    unknown_pids = set(shuffled_pids[n_known:])

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
        max_score = float(valid_sims[best_idx])
        is_rank1 = bool(valid_pids[best_idx] == pid)

        if pid in enrolled_pids:
            target_scores.append(max_score)
            target_correct_rank1.append(is_rank1)
        elif pid in unknown_pids:
            impostor_scores.append(max_score)

    metrics = compute_metrics_from_scores_v2(
        target_scores=np.array(target_scores),
        target_correct_rank1=np.array(target_correct_rank1),
        impostor_scores=np.array(impostor_scores),
        tar_targets=tar_targets,
        far_targets=far_targets,
    )
    metrics["seed"] = seed
    return metrics


def compute_openset_metrics_v2(
    q_feats: np.ndarray,
    q_pids: np.ndarray,
    q_cams: np.ndarray,
    g_feats: np.ndarray,
    g_pids: np.ndarray,
    g_cams: np.ndarray,
    known_ratio: float = 0.5,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    tar_targets: Sequence[float] = (0.70, 0.80, 0.85, 0.90, 0.95),
    far_targets: Sequence[float] = (0.001, 0.01, 0.05, 0.10),
    num_grid_points: int = 1000,
) -> Dict[str, Any]:
    """Evaluator v2: Multi-seed open-set ReID evaluation with exact sorted thresholds.

    Repeats the enrolled/unknown identity split across a deterministic seed list,
    reporting mean +- std for all open-set metrics, and interpolating a mean ROC curve.

    Args:
        q_feats: Query embeddings (N_q, D).
        q_pids: Query identity labels (N_q,).
        q_cams: Query camera IDs (N_q,).
        g_feats: Gallery embeddings (N_g, D).
        g_pids: Gallery identity labels (N_g,).
        g_cams: Gallery camera IDs (N_g,).
        known_ratio: Fraction of query identities enrolled into gallery (default: 0.5).
        seeds: List of random seeds for paired cross-seed comparison (default: 0..4).
        tar_targets: Target TAR values for FAR measurement.
        far_targets: Target FAR values for DIR measurement.
        num_grid_points: Number of points for common ROC interpolation grid.

    Returns:
        Dict containing aggregated summary (mean, std), per-seed breakdowns,
        interpolated mean ROC curves, and ceiling indicators.
    """
    eval_far = np.linspace(0.0, 1.0, num_grid_points)
    per_seed_results = []
    all_interp_tars = []

    for s in seeds:
        seed_res = compute_openset_single_seed_v2(
            q_feats=q_feats,
            q_pids=q_pids,
            q_cams=q_cams,
            g_feats=g_feats,
            g_pids=g_pids,
            g_cams=g_cams,
            known_ratio=known_ratio,
            seed=s,
            tar_targets=tar_targets,
            far_targets=far_targets,
        )
        per_seed_results.append(seed_res)
        interp_tar = np.interp(eval_far, seed_res["far_curve"], seed_res["tar_curve"])
        all_interp_tars.append(interp_tar)

    mean_tar_curve = np.mean(all_interp_tars, axis=0)
    std_tar_curve = np.std(all_interp_tars, axis=0)

    # Aggregate scalar metrics across seeds
    max_tars = [r["max_tar"] for r in per_seed_results]
    mean_max_tar = float(np.mean(max_tars))
    std_max_tar = float(np.std(max_tars))

    aucs = [r["auc"] for r in per_seed_results]
    eers = [r["eer"] for r in per_seed_results]

    summary: Dict[str, Any] = {
        "max_tar": {"mean": mean_max_tar, "std": std_max_tar},
        "auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))},
        "eer": {"mean": float(np.mean(eers)), "std": float(np.std(eers))},
    }

    # Aggregate FAR at TAR
    for t in tar_targets:
        vals = [r["far_at_tar"][t] for r in per_seed_results]
        valid_vals = [v for v in vals if v is not None]
        key = f"far_at_tar_{int(round(t * 100))}"
        if len(valid_vals) == len(seeds):
            summary[key] = {
                "mean": float(np.mean(valid_vals)),
                "std": float(np.std(valid_vals)),
            }
        else:
            summary[key] = f"N/A (max TAR = {mean_max_tar * 100:.2f}%)"

    # Aggregate DIR at FAR
    def _far_label(f_val: float) -> str:
        return (
            f"{f_val * 100:.1f}".replace(".", "_")
            if f_val < 0.01
            else f"{int(round(f_val * 100))}"
        )

    for f in far_targets:
        key = f"dir_at_far_{_far_label(f)}"
        vals = [r["dir_at_far"][f] for r in per_seed_results]
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    return {
        "protocol": "openset_v2",
        "known_ratio": known_ratio,
        "seeds": list(seeds),
        "tar_targets": list(tar_targets),
        "far_targets": list(far_targets),
        "summary": summary,
        "eval_far_grid": eval_far.tolist(),
        "mean_tar_curve": mean_tar_curve.tolist(),
        "std_tar_curve": std_tar_curve.tolist(),
        "mean_max_tar": mean_max_tar,
        "per_seed_results": [
            {
                "seed": r["seed"],
                "num_enrolled_probes": r["num_enrolled_probes"],
                "num_impostor_probes": r["num_impostor_probes"],
                "max_tar": r["max_tar"],
                "auc": r["auc"],
                "eer": r["eer"],
                "far_at_tar": {
                    f"far_at_tar_{int(round(t * 100))}": (
                        r["far_at_tar"][t]
                        if r["far_at_tar"][t] is not None
                        else f"N/A (max TAR = {r['max_tar'] * 100:.2f}%)"
                    )
                    for t in tar_targets
                },
                "dir_at_far": {
                    f"dir_at_far_{_far_label(f)}": r["dir_at_far"][f]
                    for f in far_targets
                },
            }
            for r in per_seed_results
        ],
    }


def plot_roc_curve_v2(
    v2_results: Dict[str, Any],
    save_path: Union[str, Path],
    closed_set_rank1: Optional[float] = None,
) -> None:
    """Generate dual-scale / detailed ROC plot with mean curve and max-TAR ceiling marked."""
    eval_far = np.array(v2_results["eval_far_grid"])
    mean_tar = np.array(v2_results["mean_tar_curve"])
    std_tar = np.array(v2_results["std_tar_curve"])
    mean_max_tar = v2_results["mean_max_tar"]
    summary = v2_results["summary"]

    plt.figure(figsize=(8.5, 6.5), dpi=150)
    ax = plt.gca()

    # Mean ROC Curve & ±1 std dev band
    ax.plot(
        eval_far * 100,
        mean_tar * 100,
        color="#1f77b4",
        lw=2.5,
        label=f"Mean ROC ({len(v2_results['seeds'])} seeds, AUC={summary['auc']['mean'] * 100:.2f}%)",
    )
    ax.fill_between(
        eval_far * 100,
        np.maximum(0, (mean_tar - std_tar) * 100),
        np.minimum(mean_max_tar * 100, (mean_tar + std_tar) * 100),
        color="#1f77b4",
        alpha=0.2,
        label="±1 std dev band",
    )

    # Max Achievable TAR Ceiling
    ax.axhline(
        mean_max_tar * 100,
        color="#d62728",
        linestyle="--",
        lw=1.8,
        label=f"Max TAR Ceiling ({mean_max_tar * 100:.2f}%)",
    )

    # Mark key operating points (DIR @ FAR)
    colors_dir = ["#9467bd", "#2ca02c", "#ff7f0e", "#17becf"]
    markers_dir = ["s", "o", "^", "D"]
    for f_tgt, col, mk in zip([0.001, 0.01, 0.05, 0.10], colors_dir, markers_dir):
        pct_label = (
            f"{f_tgt * 100:.1f}".replace(".", "_")
            if f_tgt < 0.01
            else f"{int(round(f_tgt * 100))}"
        )
        key = f"dir_at_far_{pct_label}"
        if key in summary and isinstance(summary[key], dict):
            dir_mean = summary[key]["mean"] * 100
            f_pct = f_tgt * 100
            ax.scatter(
                [f_pct],
                [dir_mean],
                color=col,
                marker=mk,
                s=55,
                zorder=5,
                label=f"DIR @ FAR={f_pct:g}%: {dir_mean:.2f}%",
            )

    ax.set_xlim(-1, 101)
    ax.set_ylim(-1, 101)
    ax.set_title(
        "Open-Set Person Re-ID: Detection & Identification Rate (TAR/DIR) vs FAR",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("False Alarm Rate on Impostors (FAR) [%]", fontsize=11)
    ax.set_ylabel("True Accept Rate / Identification Rate (TAR / DIR) [%]", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
