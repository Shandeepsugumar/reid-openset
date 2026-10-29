"""Sanity check visualization for look-alike groups and random identity pairs."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure src/ is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reid.data import Market1501


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity check visualization for look-alike groups."
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="results/lookalike/groups_p05.json",
        help="Path to look-alike groups JSON file.",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="artifacts/lookalike_cache.npz",
        help="Path to cached descriptors .npz file.",
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
        default="docs/failures/group_samples",
        help="Output directory for sample grids.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for group sampling.",
    )
    return parser.parse_args()


def load_identity_sample_images(data_root: Path) -> Dict[int, str]:
    """Collect one representative image path for each identity from query and gallery."""
    query_set = Market1501(root_dir=data_root, split="query")
    gallery_set = Market1501(root_dir=data_root, split="gallery")
    all_samples = query_set.samples + gallery_set.samples

    img_by_pid: Dict[int, str] = {}
    for img_path, pid, _ in all_samples:
        if pid > 0 and pid not in img_by_pid:
            img_by_pid[pid] = img_path
    return img_by_pid


def render_lookalike_groups_grid(
    groups_info: List[Dict],
    img_by_pid: Dict[int, str],
    output_path: Path,
    title: str = "12 Random Look-Alike Groups (One Image Per Identity)",
) -> None:
    """Render a vertical stack of group rows, each showing all identities in the group side-by-side."""
    img_h, img_w = 200, 100
    row_h = img_h + 40
    max_group_len = max(len(g["pids"]) for g in groups_info)
    canvas_w = max(900, max_group_len * (img_w + 15) + 270)
    canvas_h = 60 + len(groups_info) * row_h + 20

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((20, 20), title, fill=(0, 0, 0), font=font)

    y_offset = 60
    for idx, g in enumerate(groups_info):
        pids = g["pids"]
        intra_d = g["intra_dist"]

        # Label on left
        caption = f"Group #{idx + 1} (size {len(pids)})\nIntra Dist: {intra_d:.4f}\nPIDs: {pids}"
        draw.text((20, y_offset + 20), caption, fill=(0, 0, 0), font=font)

        x_start = 260
        for i, pid in enumerate(pids):
            x = x_start + i * (img_w + 15)
            img_path = img_by_pid.get(pid)
            if img_path and os.path.isfile(img_path):
                try:
                    with Image.open(img_path) as im:
                        im_r = im.convert("RGB").resize((img_w, img_h))
                except Exception:
                    im_r = Image.new("RGB", (img_w, img_h), (200, 200, 200))
            else:
                im_r = Image.new("RGB", (img_w, img_h), (200, 200, 200))

            canvas.paste(im_r, (x, y_offset))
            draw.rectangle(
                [x, y_offset, x + img_w, y_offset + img_h],
                outline=(150, 150, 150),
                width=1,
            )
            draw.text(
                (x + 5, y_offset + img_h + 5), f"ID {pid}", fill=(50, 50, 50), font=font
            )

        # Divider line
        draw.line(
            [(20, y_offset + row_h - 5), (canvas_w - 20, y_offset + row_h - 5)],
            fill=(230, 230, 230),
            width=1,
        )
        y_offset += row_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"[SanityCheck] Saved lookalike groups grid to: {output_path}")


def render_random_pairs_grid(
    pairs_info: List[Dict],
    img_by_pid: Dict[int, str],
    output_path: Path,
    title: str = "12 Random Identity Pairs for Contrast (One Image Per Identity)",
) -> None:
    """Render a grid of random identity pairs for contrast."""
    img_h, img_w = 200, 100
    cell_w = img_w * 2 + 30
    cell_h = img_h + 50
    cols = 3
    rows = (len(pairs_info) + cols - 1) // cols

    canvas_w = cols * (cell_w + 30) + 40
    canvas_h = 60 + rows * (cell_h + 20) + 20

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((20, 20), title, fill=(0, 0, 0), font=font)

    for idx, p in enumerate(pairs_info):
        r = idx // cols
        c = idx % cols
        x0 = 30 + c * (cell_w + 30)
        y0 = 60 + r * (cell_h + 20)

        pid1, pid2 = p["pids"]
        dist = p["dist"]

        # Paste img 1
        path1 = img_by_pid.get(pid1)
        if path1 and os.path.isfile(path1):
            with Image.open(path1) as im1:
                im1_r = im1.convert("RGB").resize((img_w, img_h))
        else:
            im1_r = Image.new("RGB", (img_w, img_h), (200, 200, 200))
        canvas.paste(im1_r, (x0, y0))
        draw.rectangle(
            [x0, y0, x0 + img_w, y0 + img_h], outline=(150, 150, 150), width=1
        )

        # Paste img 2
        path2 = img_by_pid.get(pid2)
        if path2 and os.path.isfile(path2):
            with Image.open(path2) as im2:
                im2_r = im2.convert("RGB").resize((img_w, img_h))
        else:
            im2_r = Image.new("RGB", (img_w, img_h), (200, 200, 200))
        canvas.paste(im2_r, (x0 + img_w + 10, y0))
        draw.rectangle(
            [x0 + img_w + 10, y0, x0 + img_w * 2 + 10, y0 + img_h],
            outline=(150, 150, 150),
            width=1,
        )

        # Captions
        caption = f"Pair #{idx + 1}: IDs ({pid1}, {pid2})\nDistance: {dist:.4f}"
        draw.text((x0, y0 + img_h + 8), caption, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"[SanityCheck] Saved random pairs contrast grid to: {output_path}")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    cache_path = Path(args.cache_path)
    groups_path = Path(args.groups)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(groups_path, "r", encoding="utf-8") as f:
        groups_data = json.load(f)

    groups: List[List[int]] = groups_data["groups"]

    # Load cache for distance matrix
    cache = np.load(cache_path)
    cache_pids = list(cache["pids"])
    pid_to_idx = {int(pid): idx for idx, pid in enumerate(cache_pids)}
    dist_matrix = cache["dist_matrix"]

    # Compute intra-group distances
    group_stats = []
    for g in groups:
        indices = [pid_to_idx[pid] for pid in g if pid in pid_to_idx]
        dists = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                dists.append(dist_matrix[indices[i], indices[j]])
        mean_d = float(np.mean(dists)) if dists else 0.0
        max_d = float(np.max(dists)) if dists else 0.0
        group_stats.append(
            {
                "pids": g,
                "intra_dist": mean_d,
                "max_dist": max_d,
                "size": len(g),
            }
        )

    # Sort groups by intra-group distance
    sorted_groups = sorted(group_stats, key=lambda x: x["intra_dist"])

    print("\n" + "=" * 65)
    print("5 MOST SIMILAR LOOK-ALIKE GROUPS (Lowest Intra-Group Distance)")
    print("=" * 65)
    for i, g in enumerate(sorted_groups[:5]):
        print(
            f"#{i + 1}: Mean Dist = {g['intra_dist']:.4f} (Max: {g['max_dist']:.4f}) | Size {g['size']} | PIDs: {g['pids']}"
        )

    print("\n" + "=" * 65)
    print("5 LEAST SIMILAR LOOK-ALIKE GROUPS (Highest Intra-Group Distance)")
    print("=" * 65)
    for i, g in enumerate(sorted_groups[-5:]):
        print(
            f"#{i + 1}: Mean Dist = {g['intra_dist']:.4f} (Max: {g['max_dist']:.4f}) | Size {g['size']} | PIDs: {g['pids']}"
        )

    # Load image paths
    print(f"\n[SanityCheck] Loading sample images from {data_root}...")
    img_by_pid = load_identity_sample_images(data_root)

    # 12 Random Look-Alike Groups
    rng = np.random.RandomState(args.seed)
    random_12_indices = rng.choice(len(group_stats), size=12, replace=False)
    random_12_groups = [group_stats[i] for i in random_12_indices]

    groups_grid_path = output_dir / "lookalike_groups_12_sample.png"
    render_lookalike_groups_grid(random_12_groups, img_by_pid, groups_grid_path)

    # 12 Random Identity Pairs (Contrast)
    pid_to_group = {}
    for g_idx, g in enumerate(groups):
        for pid in g:
            pid_to_group[pid] = g_idx

    all_test_pids = sorted(img_by_pid.keys())
    contrast_pairs = []
    attempts = 0
    while len(contrast_pairs) < 12 and attempts < 1000:
        attempts += 1
        p_a, p_b = rng.choice(all_test_pids, size=2, replace=False)
        p_a, p_b = int(p_a), int(p_b)
        if (
            (p_a in pid_to_group)
            and (p_b in pid_to_group)
            and (pid_to_group[p_a] == pid_to_group[p_b])
        ):
            continue
        if p_a in pid_to_idx and p_b in pid_to_idx:
            d = float(dist_matrix[pid_to_idx[p_a], pid_to_idx[p_b]])
            contrast_pairs.append(
                {
                    "pids": (p_a, p_b),
                    "dist": d,
                }
            )

    pairs_grid_path = output_dir / "random_pairs_12_contrast.png"
    render_random_pairs_grid(contrast_pairs, img_by_pid, pairs_grid_path)
    print("\n[SanityCheck] Group sanity check completed successfully.")


if __name__ == "__main__":
    main()
