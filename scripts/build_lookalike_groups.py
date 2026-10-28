"""Build look-alike groups from cached clothing descriptors via complete-linkage clustering."""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure src/ is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reid.eval.lookalike import (
    cluster_lookalike_groups,
    compute_or_load_identity_descriptors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build look-alike groups from clothing descriptors."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("REID_DATA_ROOT", "data/Market-1501-v15.09.15"),
        help="Path to Market-1501 dataset directory.",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="artifacts/lookalike_cache.npz",
        help="Path to cached descriptors .npz file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/lookalike",
        help="Directory to save group JSON files.",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=[2.0, 5.0, 10.0],
        help="Percentiles to evaluate for clustering threshold.",
    )
    parser.add_argument(
        "--sat-threshold",
        type=float,
        default=0.15,
        help="Saturation threshold below which pixels are achromatic (default: 0.15).",
    )
    parser.add_argument(
        "--val-threshold",
        type=float,
        default=0.15,
        help="Value threshold below which pixels are achromatic (default: 0.15).",
    )
    parser.add_argument(
        "--num-gray-bins",
        type=int,
        default=4,
        help="Number of dedicated gray-level bins for achromatic pixels (default: 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    cache_path = Path(args.cache_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Lookalike] Loading descriptors (cache: {cache_path})...")
    pids, _, _, dist_matrix = compute_or_load_identity_descriptors(
        data_root=data_root,
        cache_path=cache_path,
        sat_threshold=args.sat_threshold,
        val_threshold=args.val_threshold,
        num_gray_bins=args.num_gray_bins,
    )
    print(f"[Lookalike] Loaded {len(pids)} test identities.")

    for p in args.percentiles:
        res = cluster_lookalike_groups(dist_matrix, pids, percentile=p)
        tag = f"p{int(round(p)):02d}"
        out_file = output_dir / f"groups_{tag}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print("\n" + "=" * 55)
        print(f"LOOK-ALIKE GROUPS SUMMARY ({tag}, percentile={p}%)")
        print("=" * 55)
        print(f"  Distance Threshold         : {res['distance_threshold']:.4f}")
        print(f"  Global Median Pairwise Dist: {res['global_median_distance']:.4f}")
        print(
            f"  Total Identities in Groups : {res['total_identities_in_groups']} / {len(pids)}"
        )
        print(f"  Number of Groups           : {res['num_groups']}")
        print(f"  Group Size Histogram       : {res['group_size_histogram']}")
        print(f"  Mean Intra-Group Distance  : {res['mean_intra_group_distance']:.4f}")
        print(f"  Max Intra-Group Distance   : {res['max_intra_group_distance']:.4f}")
        print(f"  Saved to                   : {out_file}")


if __name__ == "__main__":
    main()
