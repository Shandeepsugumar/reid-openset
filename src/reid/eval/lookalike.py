"""Model-independent look-alike test protocol and clothing descriptor extraction.

Extracts HSV color histogram descriptors from upper/lower body regions,
computes pairwise identity Bhattacharyya distances, and groups identities via
complete-linkage agglomerative clustering to evaluate ReID robustness to look-alikes.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering

from reid.eval.metrics import compute_metrics_from_scores_v2


def compute_clothing_descriptor(
    image: Union[Image.Image, np.ndarray],
    upper_rows: Tuple[float, float] = (0.20, 0.55),
    lower_rows: Tuple[float, float] = (0.55, 0.95),
    cols: Tuple[float, float] = (0.15, 0.85),
    bins: Tuple[int, int, int] = (8, 4, 4),
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 3D HSV color histograms for upper and lower body crops.

    Args:
        image: PIL Image or RGB numpy array of shape (H, W, 3).
        upper_rows: (start_frac, end_frac) row range for upper body (default: 0.20 to 0.55).
        lower_rows: (start_frac, end_frac) row range for lower body (default: 0.55 to 0.95).
        cols: (start_frac, end_frac) column range for central crop (default: 0.15 to 0.85).
        bins: (H_bins, S_bins, V_bins) histogram bin count (default: 8, 4, 4 = 128 bins).

    Returns:
        Tuple of (upper_histogram, lower_histogram), each L1-normalized (sum = 1.0).
    """
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8))
    else:
        pil_img = image

    hsv_img = pil_img.convert("HSV")
    hsv_arr = np.asarray(hsv_img, dtype=np.int32)
    h, w, _ = hsv_arr.shape

    u_r0, u_r1 = int(round(h * upper_rows[0])), int(round(h * upper_rows[1]))
    l_r0, l_r1 = int(round(h * lower_rows[0])), int(round(h * lower_rows[1]))
    c0, c1 = int(round(w * cols[0])), int(round(w * cols[1]))

    # Ensure valid ranges
    u_r0, u_r1 = max(0, u_r0), min(h, max(u_r0 + 1, u_r1))
    l_r0, l_r1 = max(0, l_r0), min(h, max(l_r0 + 1, l_r1))
    c0, c1 = max(0, c0), min(w, max(c0 + 1, c1))

    upper_crop = hsv_arr[u_r0:u_r1, c0:c1]
    lower_crop = hsv_arr[l_r0:l_r1, c0:c1]

    def _hist3d(crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.ones(bins[0] * bins[1] * bins[2], dtype=np.float64) / (
                bins[0] * bins[1] * bins[2]
            )
        h_channel = crop[:, :, 0]
        s_channel = crop[:, :, 1]
        v_channel = crop[:, :, 2]

        h_idx = np.clip((h_channel * bins[0]) // 256, 0, bins[0] - 1)
        s_idx = np.clip((s_channel * bins[1]) // 256, 0, bins[1] - 1)
        v_idx = np.clip((v_channel * bins[2]) // 256, 0, bins[2] - 1)

        linear_idx = h_idx * (bins[1] * bins[2]) + s_idx * bins[2] + v_idx
        num_bins = bins[0] * bins[1] * bins[2]
        counts = np.bincount(linear_idx.ravel(), minlength=num_bins).astype(np.float64)
        total = np.sum(counts)
        if total > 0:
            counts /= total
        return counts

    upper_hist = _hist3d(upper_crop)
    lower_hist = _hist3d(lower_crop)
    return upper_hist, lower_hist


def bhattacharyya_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute Bhattacharyya distance between two L1-normalized histograms.

    Distance D_B(p, q) = sqrt(1.0 - sum(sqrt(p_i * q_i))).
    Bounded in [0.0, 1.0].
    """
    bc = np.sum(np.sqrt(np.clip(hist1, 0.0, None) * np.clip(hist2, 0.0, None)))
    bc = float(np.clip(bc, 0.0, 1.0))
    return float(np.sqrt(1.0 - bc))


def compute_or_load_identity_descriptors(
    data_root: Union[str, Path],
    cache_path: Optional[Union[str, Path]] = None,
    upper_rows: Tuple[float, float] = (0.20, 0.55),
    lower_rows: Tuple[float, float] = (0.55, 0.95),
    cols: Tuple[float, float] = (0.15, 0.85),
    bins: Tuple[int, int, int] = (8, 4, 4),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean clothing descriptors and pairwise distance matrix for all test identities.

    Loads from cache_path if existing, otherwise processes query and gallery splits.

    Returns:
        Tuple of (pids, upper_descriptors, lower_descriptors, distance_matrix)
        where pids is (N,), descriptors are (N, 128), and distance_matrix is (N, N).
    """
    if cache_path is not None:
        c_path = Path(cache_path)
        if c_path.is_file():
            data = np.load(c_path)
            return (
                data["pids"],
                data["upper_descriptors"],
                data["lower_descriptors"],
                data["dist_matrix"],
            )

    from reid.data import Market1501

    data_dir = Path(data_root)
    query_set = Market1501(root_dir=data_dir, split="query")
    gallery_set = Market1501(root_dir=data_dir, split="gallery")

    all_samples = query_set.samples + gallery_set.samples
    # Group image paths by valid pid (ignore junk -1 and distractor 0)
    samples_by_pid = defaultdict(list)
    for img_path, pid, _ in all_samples:
        if pid > 0:
            samples_by_pid[pid].append(img_path)

    sorted_pids = np.array(sorted(samples_by_pid.keys()), dtype=np.int32)
    n_pids = len(sorted_pids)
    num_bins = bins[0] * bins[1] * bins[2]

    upper_descriptors = np.zeros((n_pids, num_bins), dtype=np.float64)
    lower_descriptors = np.zeros((n_pids, num_bins), dtype=np.float64)

    for i, pid in enumerate(sorted_pids):
        img_paths = samples_by_pid[pid]
        u_list = []
        l_list = []
        for p in img_paths:
            try:
                with Image.open(p) as img:
                    u_h, l_h = compute_clothing_descriptor(
                        img,
                        upper_rows=upper_rows,
                        lower_rows=lower_rows,
                        cols=cols,
                        bins=bins,
                    )
                    u_list.append(u_h)
                    l_list.append(l_h)
            except Exception as e:
                print(f"[Warning] Could not process {p}: {e}")

        if u_list:
            mean_u = np.mean(u_list, axis=0)
            mean_l = np.mean(l_list, axis=0)
            upper_descriptors[i] = mean_u / (np.sum(mean_u) + 1e-12)
            lower_descriptors[i] = mean_l / (np.sum(mean_l) + 1e-12)

    # Compute pairwise distance matrix
    dist_matrix = np.zeros((n_pids, n_pids), dtype=np.float64)
    for i in range(n_pids):
        for j in range(i + 1, n_pids):
            d_u = bhattacharyya_distance(upper_descriptors[i], upper_descriptors[j])
            d_l = bhattacharyya_distance(lower_descriptors[i], lower_descriptors[j])
            d = 0.5 * (d_u + d_l)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    if cache_path is not None:
        c_path = Path(cache_path)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            c_path,
            pids=sorted_pids,
            upper_descriptors=upper_descriptors,
            lower_descriptors=lower_descriptors,
            dist_matrix=dist_matrix,
        )

    return sorted_pids, upper_descriptors, lower_descriptors, dist_matrix


def _split_oversized_cluster(
    cluster_indices: List[int],
    sub_dist_mat: np.ndarray,
    threshold: float,
    max_size: int = 6,
) -> List[List[int]]:
    """Recursively split a cluster exceeding max_size using complete linkage."""
    if len(cluster_indices) <= max_size:
        return [cluster_indices] if len(cluster_indices) >= 2 else []

    # Try clustering with 2 clusters
    agg = AgglomerativeClustering(
        n_clusters=2,
        metric="precomputed",
        linkage="complete",
    )
    labels = agg.fit_predict(sub_dist_mat)
    idx_a = [cluster_indices[k] for k in range(len(cluster_indices)) if labels[k] == 0]
    idx_b = [cluster_indices[k] for k in range(len(cluster_indices)) if labels[k] == 1]

    result = []
    for sub_idx in (idx_a, idx_b):
        if len(sub_idx) < 2:
            continue
        sub_sub_mat = sub_dist_mat[
            np.ix_(
                [
                    k
                    for k in range(len(cluster_indices))
                    if cluster_indices[k] in sub_idx
                ],
                [
                    k
                    for k in range(len(cluster_indices))
                    if cluster_indices[k] in sub_idx
                ],
            )
        ]
        result.extend(
            _split_oversized_cluster(sub_idx, sub_sub_mat, threshold, max_size=max_size)
        )
    return result


def cluster_lookalike_groups(
    dist_matrix: np.ndarray,
    pids: np.ndarray,
    percentile: float = 5.0,
    max_group_size: int = 6,
) -> Dict[str, Any]:
    """Group identities via complete-linkage agglomerative clustering.

    Guarantees that every pair inside a group has pairwise distance <= threshold.
    Cuts clusters of size > max_group_size further.

    Args:
        dist_matrix: (N, N) pairwise distance matrix.
        pids: (N,) identity labels.
        percentile: Percentile of pairwise distances to set threshold (default: 5.0).
        max_group_size: Maximum allowed cluster size (default: 6).

    Returns:
        Dict with grouping results and summary statistics.
    """
    n = len(pids)
    iu = np.triu_indices(n, k=1)
    condensed_dists = dist_matrix[iu]
    threshold = float(np.percentile(condensed_dists, percentile))
    global_median = float(np.median(condensed_dists))

    # Complete linkage clustering with distance_threshold
    agg = AgglomerativeClustering(
        distance_threshold=threshold,
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
    )
    labels = agg.fit_predict(dist_matrix)

    initial_clusters = defaultdict(list)
    for idx, lbl in enumerate(labels):
        initial_clusters[lbl].append(idx)

    final_groups_indices: List[List[int]] = []
    for _, idx_list in initial_clusters.items():
        if len(idx_list) < 2:
            continue
        if len(idx_list) <= max_group_size:
            final_groups_indices.append(idx_list)
        else:
            sub_mat = dist_matrix[np.ix_(idx_list, idx_list)]
            split_groups = _split_oversized_cluster(
                idx_list, sub_mat, threshold, max_size=max_group_size
            )
            final_groups_indices.extend(split_groups)

    # Sort groups deterministically by lowest pid in group
    groups_pids: List[List[int]] = []
    intra_dists: List[float] = []
    max_intra = 0.0

    for idx_list in final_groups_indices:
        g_p = sorted([int(pids[k]) for k in idx_list])
        groups_pids.append(g_p)
        for i_a in range(len(idx_list)):
            for i_b in range(i_a + 1, len(idx_list)):
                d = float(dist_matrix[idx_list[i_a], idx_list[i_b]])
                intra_dists.append(d)
                if d > max_intra:
                    max_intra = d

    groups_pids.sort(key=lambda g: g[0])

    # Group size histogram
    size_histogram: Dict[int, int] = defaultdict(int)
    total_grouped_ids = 0
    for g in groups_pids:
        size_histogram[len(g)] += 1
        total_grouped_ids += len(g)

    return {
        "percentile": percentile,
        "distance_threshold": threshold,
        "global_median_distance": global_median,
        "num_groups": len(groups_pids),
        "total_identities_in_groups": total_grouped_ids,
        "group_size_histogram": dict(sorted(size_histogram.items())),
        "mean_intra_group_distance": (
            float(np.mean(intra_dists)) if intra_dists else 0.0
        ),
        "max_intra_group_distance": max_intra,
        "groups": groups_pids,
    }


def partition_lookalike_split(
    groups: List[List[int]],
    all_test_pids: Sequence[int],
    seed: int = 0,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Stratified partition of look-alike groups into enrolled and unknown identities.

    In every group:
    - ~half (at least 1, at most size-1) are assigned to enrolled.
    - Remaining are assigned to unknown (look-alike impostors).
    - Non-group test identities are designated as extra_gallery_pids.

    Returns:
        (enrolled_pids, unknown_pids, extra_gallery_pids)
    """
    rng = np.random.RandomState(seed)
    enrolled: Set[int] = set()
    unknown: Set[int] = set()

    for group in groups:
        perm = rng.permutation(sorted(group))
        n_enrolled = max(1, len(group) // 2)
        enrolled.update(int(x) for x in perm[:n_enrolled])
        unknown.update(int(x) for x in perm[n_enrolled:])

    all_set = {int(x) for x in all_test_pids}
    extra_gallery = all_set - (enrolled | unknown)
    return enrolled, unknown, extra_gallery


def partition_matched_control_split(
    num_enrolled: int,
    num_unknown: int,
    all_test_pids: Sequence[int],
    seed: int = 0,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Matched random control partition with identical enrolled/unknown counts.

    Returns:
        (enrolled_pids, unknown_pids, extra_gallery_pids)
    """
    rng = np.random.RandomState(seed)
    sorted_all = sorted([int(x) for x in all_test_pids])
    shuffled = rng.permutation(sorted_all)

    enrolled = {int(x) for x in shuffled[:num_enrolled]}
    unknown = {int(x) for x in shuffled[num_enrolled : num_enrolled + num_unknown]}
    extra_gallery = {int(x) for x in shuffled[num_enrolled + num_unknown :]}
    return enrolled, unknown, extra_gallery


def evaluate_partition(
    q_feats: np.ndarray,
    q_pids: np.ndarray,
    q_cams: np.ndarray,
    g_feats: np.ndarray,
    g_pids: np.ndarray,
    g_cams: np.ndarray,
    enrolled_pids: Set[int],
    unknown_pids: Set[int],
    extra_gallery_pids: Set[int],
    tar_targets: Sequence[float] = (0.70, 0.80, 0.85, 0.90, 0.95),
    far_targets: Sequence[float] = (0.001, 0.01, 0.05, 0.10),
) -> Dict[str, Any]:
    """Evaluate open-set metrics for a specific identity partition.

    Gallery: enrolled_pids + extra_gallery_pids + distractor pid 0.
    Probes: queries of enrolled_pids (targets) + queries of unknown_pids (impostors).
    """
    valid_gallery_pids = enrolled_pids | extra_gallery_pids
    g_mask = np.isin(g_pids, list(valid_gallery_pids)) | (g_pids == 0)
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

        valid_match_mask = ~((sub_g_pids == pid) & (sub_g_cams == cam)) & (
            sub_g_pids != -1
        )
        if not np.any(valid_match_mask):
            continue

        valid_sims = sim_mat[i][valid_match_mask]
        valid_match_pids = sub_g_pids[valid_match_mask]

        best_idx = np.argmax(valid_sims)
        max_score = float(valid_sims[best_idx])
        is_rank1 = bool(valid_match_pids[best_idx] == pid)

        if pid in enrolled_pids:
            target_scores.append(max_score)
            target_correct_rank1.append(is_rank1)
        elif pid in unknown_pids:
            impostor_scores.append(max_score)

    target_scores_arr = np.array(target_scores, dtype=np.float64)
    target_correct_arr = np.array(target_correct_rank1, dtype=bool)
    impostor_scores_arr = np.array(impostor_scores, dtype=np.float64)

    metrics = compute_metrics_from_scores_v2(
        target_scores=target_scores_arr,
        target_correct_rank1=target_correct_arr,
        impostor_scores=impostor_scores_arr,
        tar_targets=tar_targets,
        far_targets=far_targets,
    )

    mean_max_imp = (
        float(np.mean(impostor_scores_arr)) if len(impostor_scores_arr) > 0 else 0.0
    )
    metrics["mean_max_impostor_similarity"] = mean_max_imp
    metrics["num_enrolled_identities"] = len(enrolled_pids)
    metrics["num_unknown_identities"] = len(unknown_pids)
    metrics["num_extra_gallery_identities"] = len(extra_gallery_pids)
    metrics["num_target_probes"] = len(target_scores_arr)
    metrics["num_impostor_probes"] = len(impostor_scores_arr)

    metrics["raw_target_scores"] = target_scores_arr
    metrics["raw_target_correct_rank1"] = target_correct_arr
    metrics["raw_impostor_scores"] = impostor_scores_arr

    return metrics


def evaluate_threshold_transfer(
    std_metrics: Dict[str, Any],
    lookalike_metrics: Dict[str, Any],
    target_tar: float = 0.80,
) -> Dict[str, Optional[float]]:
    """Determine threshold on standard split achieving target_tar, and measure FAR/TAR on lookalike split."""
    std_tar_curve = std_metrics.get("tar_curve", np.array([]))
    std_thresholds = std_metrics.get("thresholds", np.array([]))
    std_max_tar = std_metrics.get("max_tar", 0.0)

    if target_tar > std_max_tar or len(std_tar_curve) == 0:
        return {
            "standard_tau_80": None,
            "transfer_far": None,
            "transfer_tar": None,
        }

    valid_idx = np.where(std_tar_curve >= target_tar)[0]
    if len(valid_idx) == 0:
        return {
            "standard_tau_80": None,
            "transfer_far": None,
            "transfer_tar": None,
        }

    tau_80 = float(std_thresholds[valid_idx[0]])

    lk_tgt_scores = lookalike_metrics.get("raw_target_scores", np.array([]))
    lk_tgt_correct = lookalike_metrics.get("raw_target_correct_rank1", np.array([]))
    lk_imp_scores = lookalike_metrics.get("raw_impostor_scores", np.array([]))

    if len(lk_tgt_scores) > 0:
        transfer_tar = float(
            np.sum((lk_tgt_scores >= tau_80) & lk_tgt_correct) / len(lk_tgt_scores)
        )
    else:
        transfer_tar = None

    if len(lk_imp_scores) > 0:
        transfer_far = float(np.sum(lk_imp_scores >= tau_80) / len(lk_imp_scores))
    else:
        transfer_far = None

    return {
        "standard_tau_80": tau_80,
        "transfer_far": transfer_far,
        "transfer_tar": transfer_tar,
    }
