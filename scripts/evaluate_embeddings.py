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
    compute_openset_metrics_v2,
    compute_openset_tar_far,
    plot_roc_curve_v2,
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
        "--eval-version",
        type=str,
        choices=["v2", "legacy"],
        default="v2",
        help="Evaluator version: 'v2' (multi-seed, exact thresholds) or 'legacy' (single-seed linspace).",
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
        help="Random seed for legacy open-set identity partition (default: 42).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds list for v2 open-set evaluation (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save evaluation results as JSON.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=None,
        help="Optional path to save ROC curve plot (.png).",
    )
    return parser.parse_args()


def evaluate_embeddings(
    embeddings_dir: Path,
    eval_version: str = "v2",
    known_ratio: float = 0.5,
    seed: int = 42,
    seeds: Optional[list] = None,
    output_json: Optional[Path] = None,
    output_plot: Optional[Path] = None,
) -> Dict:
    embeddings_dir = Path(embeddings_dir)
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    q_feats = np.load(embeddings_dir / "query_embeddings.npy")
    q_pids = np.load(embeddings_dir / "query_pids.npy")
    q_cams = np.load(embeddings_dir / "query_camids.npy")
    g_feats = np.load(embeddings_dir / "gallery_embeddings.npy")
    g_pids = np.load(embeddings_dir / "gallery_pids.npy")
    g_cams = np.load(embeddings_dir / "gallery_camids.npy")

    print(
        f"[Eval Embeddings] Loaded {len(q_pids)} queries and {len(g_pids)} gallery items from {embeddings_dir}"
    )

    # Closed-set metrics (identical for both versions)
    cmc, mAP = compute_closed_set_cmc_and_map(
        q_feats, q_pids, q_cams, g_feats, g_pids, g_cams
    )
    print("\n" + "=" * 50)
    print("  CLOSED-SET RE-IDENTIFICATION RESULTS")
    print("=" * 50)
    print(f"  Rank-1  Accuracy: {cmc[0] * 100:.2f}%")
    print(f"  Rank-5  Accuracy: {cmc[4] * 100:.2f}%")
    print(f"  Rank-10 Accuracy: {cmc[9] * 100:.2f}%")
    print(f"  Mean AP (mAP):   {mAP * 100:.2f}%")
    print("=" * 50)

    if eval_version == "legacy":
        # Legacy single-seed evaluation
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
        print("\n" + "=" * 50)
        print("  OPEN-SET PERSON RE-ID METRICS (LEGACY)")
        print("=" * 50)
        print(f"  Enrolled Probes : {openset_results['num_enrolled_probes']}")
        print(f"  Impostor Probes : {openset_results['num_unknown_probes']}")
        print(f"  FAR @ TAR=80%   : {openset_results['far_at_tar_80'] * 100:.2f}%")
        print(f"  FAR @ TAR=90%   : {openset_results['far_at_tar_90'] * 100:.2f}%")
        print(f"  FAR @ TAR=95%   : {openset_results['far_at_tar_95'] * 100:.2f}%")
        print("=" * 50)

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

    else:
        # Evaluator v2: Multi-seed, exact sorted thresholds, DIR@FAR, AUC/EER
        v2_raw = compute_openset_metrics_v2(
            q_feats,
            q_pids,
            q_cams,
            g_feats,
            g_pids,
            g_cams,
            known_ratio=known_ratio,
            seeds=seeds,
        )
        s = v2_raw["summary"]
        print("\n" + "=" * 55)
        print(f"  OPEN-SET PERSON RE-ID METRICS (V2, {len(seeds)} SEEDS)")
        print("=" * 55)
        print(
            f"  Max TAR (Ceiling): {s['max_tar']['mean'] * 100:.2f}% ± {s['max_tar']['std'] * 100:.2f}%"
        )
        print(
            f"  ROC AUC          : {s['auc']['mean'] * 100:.2f}% ± {s['auc']['std'] * 100:.2f}%"
        )
        print(
            f"  EER              : {s['eer']['mean'] * 100:.2f}% ± {s['eer']['std'] * 100:.2f}%"
        )
        print("-" * 55)

        for t in [70, 80, 85, 90, 95]:
            k = f"far_at_tar_{t}"
            if k in s:
                val = s[k]
                if isinstance(val, dict):
                    print(
                        f"  FAR @ TAR={t}%    : {val['mean'] * 100:.2f}% ± {val['std'] * 100:.2f}%"
                    )
                else:
                    print(f"  FAR @ TAR={t}%    : {val}")
        print("-" * 55)

        for f_label, f_pct in [
            ("0_1", 0.1),
            ("1", 1.0),
            ("5", 5.0),
            ("10", 10.0),
        ]:
            k = f"dir_at_far_{f_label}"
            if k in s and isinstance(s[k], dict):
                val = s[k]
                print(
                    f"  DIR @ FAR={f_pct:g}%    : {val['mean'] * 100:.2f}% ± {val['std'] * 100:.2f}%"
                )
        print("=" * 55)

        if output_plot:
            output_plot_path = Path(output_plot)
            output_plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_roc_curve_v2(v2_raw, output_plot_path)
            print(f"[Eval Embeddings] v2 ROC plot saved to: {output_plot_path}")

        results = {
            "protocol": "openset_v2",
            "rank_1": float(cmc[0] * 100),
            "rank_5": float(cmc[4] * 100),
            "rank_10": float(cmc[9] * 100),
            "mAP": float(mAP * 100),
            "known_ratio": known_ratio,
            "seeds": seeds,
            "summary": s,
            "per_seed": v2_raw["per_seed_results"],
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
        eval_version=args.eval_version,
        known_ratio=args.known_ratio,
        seed=args.seed,
        seeds=args.seeds,
        output_json=Path(args.output_json) if args.output_json else None,
        output_plot=Path(args.output_plot) if args.output_plot else None,
    )


if __name__ == "__main__":
    main()
