"""Evaluate ReID metrics directly from saved .npy embedding arrays on CPU."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reid.eval import (
    compute_closed_set_cmc_and_map,
    compute_openset_tar_far,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ReID metrics from saved .npy arrays."
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="artifacts/baseline_resnet18",
        help="Directory containing the 6 saved .npy files.",
    )
    parser.add_argument(
        "--known-ratio",
        type=float,
        default=0.5,
        help="Fraction of query identities to treat as enrolled targets (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for open-set identity partition (default: 42).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save evaluation results as JSON.",
    )
    return parser.parse_args()


def evaluate_embeddings(
    embeddings_dir: Path,
    known_ratio: float = 0.5,
    seed: int = 42,
    output_json: Optional[Path] = None,
) -> Dict[str, float]:
    embeddings_dir = Path(embeddings_dir)

    q_feats = np.load(embeddings_dir / "query_embeddings.npy")
    q_pids = np.load(embeddings_dir / "query_pids.npy")
    q_cams = np.load(embeddings_dir / "query_camids.npy")
    g_feats = np.load(embeddings_dir / "gallery_embeddings.npy")
    g_pids = np.load(embeddings_dir / "gallery_pids.npy")
    g_cams = np.load(embeddings_dir / "gallery_camids.npy")

    print(
        f"[Eval Embeddings] Loaded {len(q_pids)} queries and {len(g_pids)} gallery items from {embeddings_dir}"
    )

    # Closed-set metrics
    cmc, mAP = compute_closed_set_cmc_and_map(
        q_feats, q_pids, q_cams, g_feats, g_pids, g_cams
    )
    print("\n" + "=" * 45)
    print("  CLOSED-SET RE-IDENTIFICATION RESULTS")
    print("=" * 45)
    print(f"  Rank-1  Accuracy: {cmc[0] * 100:.2f}%")
    print(f"  Rank-5  Accuracy: {cmc[4] * 100:.2f}%")
    print(f"  Rank-10 Accuracy: {cmc[9] * 100:.2f}%")
    print(f"  Mean AP (mAP):   {mAP * 100:.2f}%")
    print("=" * 45)

    # Open-set metrics
    openset_results = compute_openset_tar_far(
        q_feats,
        q_pids,
        q_cams,
        g_feats,
        g_pids,
        g_cams,
        known_ratio=known_ratio,
        seed=seed,
    )
    print("\n" + "=" * 45)
    print("  OPEN-SET PERSON RE-ID METRICS")
    print("=" * 45)
    print(f"  Enrolled Probes : {openset_results['num_enrolled_probes']}")
    print(f"  Impostor Probes : {openset_results['num_unknown_probes']}")
    print(f"  FAR @ TAR=80%   : {openset_results['far_at_tar_80'] * 100:.2f}%")
    print(f"  FAR @ TAR=90%   : {openset_results['far_at_tar_90'] * 100:.2f}%")
    print(f"  FAR @ TAR=95%   : {openset_results['far_at_tar_95'] * 100:.2f}%")
    print("=" * 45)

    results = {
        "rank_1": float(cmc[0] * 100),
        "rank_5": float(cmc[4] * 100),
        "rank_10": float(cmc[9] * 100),
        "mAP": float(mAP * 100),
        "far_at_tar_80": float(openset_results["far_at_tar_80"] * 100),
        "far_at_tar_90": float(openset_results["far_at_tar_90"] * 100),
        "far_at_tar_95": float(openset_results["far_at_tar_95"] * 100),
        "num_enrolled_probes": int(openset_results["num_enrolled_probes"]),
        "num_unknown_probes": int(openset_results["num_unknown_probes"]),
    }

    if output_json:
        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Eval Embeddings] Results saved to: {output_json_path}")

    return results


def main() -> None:
    args = parse_args()
    evaluate_embeddings(
        embeddings_dir=Path(args.embeddings_dir),
        known_ratio=args.known_ratio,
        seed=args.seed,
        output_json=Path(args.output_json) if args.output_json else None,
    )


if __name__ == "__main__":
    main()
