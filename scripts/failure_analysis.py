"""Failure analysis for open-set Person ReID on look-alike protocol.

Identifies worst false accepts and false rejects at the standard TAR=80% operating point,
verifies file list integrity against saved .npy arrays, and generates composite image grids.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure src/ is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reid.data import Market1501
from reid.eval.lookalike import partition_lookalike_split
from reid.eval.metrics import compute_openset_single_seed_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Failure analysis for open-set ReID look-alike protocol."
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="artifacts/baseline_resnet18",
        help="Path to directory containing saved .npy embeddings.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="results/lookalike/groups_p05.json",
        help="Path to look-alike groups JSON file.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("REID_DATA_ROOT", "data/Market-1501-v15.09.15"),
        help="Path to Market-1501 dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/failures",
        help="Directory to save failure visualization grids.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic partition seed (default: 0).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of worst false accepts and false rejects to extract.",
    )
    return parser.parse_args()


def verify_file_lists(
    data_root: Path,
    q_pids_saved: np.ndarray,
    q_cams_saved: np.ndarray,
    g_pids_saved: np.ndarray,
    g_cams_saved: np.ndarray,
) -> Tuple[List[str], List[str]]:
    """Rebuild query/gallery file lists with Market1501 and strictly verify against .npy arrays."""
    query_set = Market1501(root_dir=data_root, split="query")
    gallery_set = Market1501(root_dir=data_root, split="gallery")

    q_paths = [s[0] for s in query_set.samples]
    q_pids = np.array([s[1] for s in query_set.samples], dtype=np.int32)
    q_cams = np.array([s[2] for s in query_set.samples], dtype=np.int32)

    g_paths = [s[0] for s in gallery_set.samples]
    g_pids = np.array([s[1] for s in gallery_set.samples], dtype=np.int32)
    g_cams = np.array([s[2] for s in gallery_set.samples], dtype=np.int32)

    if len(q_paths) != len(q_pids_saved):
        raise ValueError(
            f"Query file count mismatch: dataset has {len(q_paths)}, saved .npy has {len(q_pids_saved)}"
        )
    if not np.array_equal(q_pids, q_pids_saved):
        raise ValueError("Query identity IDs in dataset do not match query_pids.npy!")
    if not np.array_equal(q_cams, q_cams_saved):
        raise ValueError("Query camera IDs in dataset do not match query_camids.npy!")

    if len(g_paths) != len(g_pids_saved):
        raise ValueError(
            f"Gallery file count mismatch: dataset has {len(g_paths)}, saved .npy has {len(g_pids_saved)}"
        )
    if not np.array_equal(g_pids, g_pids_saved):
        raise ValueError(
            "Gallery identity IDs in dataset do not match gallery_pids.npy!"
        )
    if not np.array_equal(g_cams, g_cams_saved):
        raise ValueError(
            "Gallery camera IDs in dataset do not match gallery_camids.npy!"
        )

    print(
        f"[Verify] File lists strictly verified against .npy arrays ({len(q_paths)} queries, {len(g_paths)} gallery)."
    )
    return q_paths, g_paths


def create_comparison_grid(
    pairs: List[Dict[str, Any]],
    output_path: Path,
    title: str,
    is_false_accept: bool = True,
) -> None:
    """Create a high-resolution grid of probe and match pairs with detailed captions."""
    img_h, img_w = 256, 128
    caption_h = 70
    cell_w = img_w * 2 + 30
    cell_h = img_h + caption_h + 20

    cols = 2
    rows = (len(pairs) + cols - 1) // cols
    header_h = 50

    canvas_w = cols * cell_w + 30
    canvas_h = header_h + rows * cell_h + 20

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Draw header
    draw.text((20, 15), title, fill=(0, 0, 0), font=font)

    for idx, item in enumerate(pairs):
        r = idx // cols
        c = idx % cols
        x0 = 20 + c * cell_w
        y0 = header_h + r * cell_h

        # Probe image
        try:
            with Image.open(item["probe_path"]) as p_img:
                p_resized = p_img.convert("RGB").resize((img_w, img_h))
        except Exception:
            p_resized = Image.new("RGB", (img_w, img_h), (200, 200, 200))

        # Match image
        try:
            with Image.open(item["match_path"]) as m_img:
                m_resized = m_img.convert("RGB").resize((img_w, img_h))
        except Exception:
            m_resized = Image.new("RGB", (img_w, img_h), (200, 200, 200))

        canvas.paste(p_resized, (x0, y0))
        canvas.paste(m_resized, (x0 + img_w + 10, y0))

        # Bounding border
        border_color = (200, 0, 0) if is_false_accept else (0, 0, 200)
        draw.rectangle(
            [x0 - 2, y0 - 2, x0 + cell_w - 20, y0 + cell_h - 10],
            outline=(220, 220, 220),
            width=1,
        )
        draw.rectangle([x0, y0, x0 + img_w, y0 + img_h], outline=border_color, width=2)
        draw.rectangle(
            [x0 + img_w + 10, y0, x0 + img_w * 2 + 10, y0 + img_h],
            outline=border_color,
            width=2,
        )

        # Captions
        cy = y0 + img_h + 5
        line1 = f"#{idx+1} Score: {item['score']:.4f} (tau80: {item['tau']:.4f})"
        line2 = f"Probe: ID {item['probe_pid']} (cam {item['probe_cam']})"
        line3 = f"Match: ID {item['match_pid']} (cam {item['match_cam']})"
        draw.text((x0, cy), line1, fill=(0, 0, 0), font=font)
        draw.text((x0, cy + 18), line2, fill=(80, 80, 80), font=font)
        draw.text(
            (x0, cy + 36),
            line3,
            fill=(180, 0, 0) if is_false_accept else (0, 100, 0),
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"[Visualization] Saved composite grid to: {output_path}")


def main() -> None:
    args = parse_args()
    emb_dir = Path(args.embeddings_dir)
    data_root = Path(args.data_root)
    groups_path = Path(args.groups)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Embeddings
    print(f"[FailureAnalysis] Loading embeddings from {emb_dir}...")
    q_feats = np.load(emb_dir / "query_embeddings.npy")
    q_pids = np.load(emb_dir / "query_pids.npy")
    q_cams = np.load(emb_dir / "query_camids.npy")
    g_feats = np.load(emb_dir / "gallery_embeddings.npy")
    g_pids = np.load(emb_dir / "gallery_pids.npy")
    g_cams = np.load(emb_dir / "gallery_camids.npy")

    # 2. Strict Verification of File Lists
    q_paths, g_paths = verify_file_lists(
        data_root=data_root,
        q_pids_saved=q_pids,
        q_cams_saved=q_cams,
        g_pids_saved=g_pids,
        g_cams_saved=g_cams,
    )

    # 3. Determine Standard Protocol TAR=80% operating point
    print(
        f"[FailureAnalysis] Computing standard protocol TAR=80% operating point for seed {args.seed}..."
    )
    std_metrics = compute_openset_single_seed_v2(
        q_feats,
        q_pids,
        q_cams,
        g_feats,
        g_pids,
        g_cams,
        known_ratio=0.5,
        seed=args.seed,
    )
    tar_curve = std_metrics["tar_curve"]
    thresholds = std_metrics["thresholds"]
    valid_tar_idx = np.where(tar_curve >= 0.80)[0]
    if len(valid_tar_idx) == 0:
        raise RuntimeError("Standard protocol could not reach TAR=80%!")
    tau_80 = float(thresholds[valid_tar_idx[0]])
    actual_std_tar = float(tar_curve[valid_tar_idx[0]])
    actual_std_far = float(std_metrics["far_curve"][valid_tar_idx[0]])
    print(
        f"[FailureAnalysis] Operating Threshold tau_80 = {tau_80:.4f} (Standard TAR: {actual_std_tar*100:.2f}%, FAR: {actual_std_far*100:.2f}%)"
    )

    # 4. Partition Look-Alike Split on Seed
    import json

    with open(groups_path, "r", encoding="utf-8") as f:
        groups_data = json.load(f)
    groups = groups_data["groups"]

    all_test_pids = sorted(set(q_pids[q_pids > 0]) | set(g_pids[g_pids > 0]))
    enrolled_pids, unknown_pids, extra_gallery_pids = partition_lookalike_split(
        groups=groups, all_test_pids=all_test_pids, seed=args.seed
    )

    # Gallery and Probe Indices
    valid_gallery_pids = enrolled_pids | extra_gallery_pids
    gallery_indices = [
        i
        for i in range(len(g_pids))
        if (g_pids[i] in valid_gallery_pids) or (g_pids[i] == 0)
    ]
    gallery_indices = np.array(gallery_indices, dtype=np.int32)

    sub_g_feats = g_feats[gallery_indices]
    sub_g_pids = g_pids[gallery_indices]
    sub_g_cams = g_cams[gallery_indices]

    false_accepts = []
    false_rejects = []

    # Iterate over queries
    for q_idx in range(len(q_pids)):
        pid = q_pids[q_idx]
        cam = q_cams[q_idx]

        if pid not in (enrolled_pids | unknown_pids):
            continue

        q_feat = q_feats[q_idx]
        sims = np.dot(sub_g_feats, q_feat)

        # Invalidate same-camera same-identity pairs and distractor -1
        valid_mask = ~((sub_g_pids == pid) & (sub_g_cams == cam)) & (sub_g_pids != -1)
        if not np.any(valid_mask):
            continue

        valid_sims = sims[valid_mask]
        valid_g_indices = gallery_indices[valid_mask]
        valid_match_pids = sub_g_pids[valid_mask]

        best_sub_idx = np.argmax(valid_sims)
        max_score = float(valid_sims[best_sub_idx])
        best_g_idx = valid_g_indices[best_sub_idx]
        match_pid = int(valid_match_pids[best_sub_idx])
        is_rank1 = bool(match_pid == pid)

        item = {
            "probe_idx": q_idx,
            "probe_path": q_paths[q_idx],
            "probe_pid": int(pid),
            "probe_cam": int(cam),
            "match_idx": int(best_g_idx),
            "match_path": g_paths[best_g_idx],
            "match_pid": match_pid,
            "match_cam": int(g_cams[best_g_idx]),
            "score": max_score,
            "tau": tau_80,
            "is_rank1": is_rank1,
        }

        # Impostor Query False Accept: score >= tau_80
        if pid in unknown_pids:
            if max_score >= tau_80:
                false_accepts.append(item)

        # Enrolled Query False Reject: score < tau_80 or wrong top-1 match
        elif pid in enrolled_pids:
            if (max_score < tau_80) or (not is_rank1):
                item["error_margin"] = tau_80 - max_score
                false_rejects.append(item)

    # Sort False Accepts by highest score (most egregious accepts)
    false_accepts.sort(key=lambda x: x["score"], reverse=True)
    top_fa = false_accepts[: args.top_k]

    # Sort False Rejects by lowest score (most severe misses)
    false_rejects.sort(key=lambda x: x["score"])
    top_fr = false_rejects[: args.top_k]

    print("\n" + "=" * 70)
    print(
        f"TOP {len(top_fa)} WORST FALSE ACCEPTS (Look-Alike Impostor Probes >= tau_80={tau_80:.4f})"
    )
    print("=" * 70)
    for i, fa in enumerate(top_fa):
        p_name = Path(fa["probe_path"]).name
        m_name = Path(fa["match_path"]).name
        print(
            f"#{i+1:02d} Score: {fa['score']:.4f} | Probe PID {fa['probe_pid']:<4} (cam {fa['probe_cam']}, {p_name}) -> Match PID {fa['match_pid']:<4} (cam {fa['match_cam']}, {m_name})"
        )

    print("\n" + "=" * 70)
    print(
        f"TOP {len(top_fr)} WORST FALSE REJECTS (Enrolled Probes < tau_80={tau_80:.4f} or Rank-1 Failure)"
    )
    print("=" * 70)
    for i, fr in enumerate(top_fr):
        p_name = Path(fr["probe_path"]).name
        m_name = Path(fr["match_path"]).name
        r1_status = "Correct ID" if fr["is_rank1"] else f"Wrong ID ({fr['match_pid']})"
        print(
            f"#{i+1:02d} Score: {fr['score']:.4f} | Probe PID {fr['probe_pid']:<4} (cam {fr['probe_cam']}, {p_name}) -> Match {r1_status} (cam {fr['match_cam']}, {m_name})"
        )

    # Generate Image Grids
    fa_grid_path = output_dir / f"worst_false_accepts_{emb_dir.name}.png"
    fr_grid_path = output_dir / f"worst_false_rejects_{emb_dir.name}.png"

    create_comparison_grid(
        pairs=top_fa,
        output_path=fa_grid_path,
        title=f"Worst False Accepts: {emb_dir.name} (Operating tau80 = {tau_80:.4f})",
        is_false_accept=True,
    )
    create_comparison_grid(
        pairs=top_fr,
        output_path=fr_grid_path,
        title=f"Worst False Rejects: {emb_dir.name} (Operating tau80 = {tau_80:.4f})",
        is_false_accept=False,
    )

    # Save metadata JSON for failure patterns
    metadata = {
        "experiment": emb_dir.name,
        "seed": args.seed,
        "tau_80": tau_80,
        "total_impostor_probes": sum(1 for p in q_pids if p in unknown_pids),
        "total_false_accepts": len(false_accepts),
        "total_enrolled_probes": sum(1 for p in q_pids if p in enrolled_pids),
        "total_false_rejects": len(false_rejects),
        "top_false_accepts": top_fa,
        "top_false_rejects": top_fr,
    }
    with open(output_dir / f"failures_{emb_dir.name}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(
        f"\n[FailureAnalysis] Completed failure analysis. Artifacts saved in {output_dir}"
    )


if __name__ == "__main__":
    main()
