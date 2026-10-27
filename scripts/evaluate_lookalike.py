"""Evaluate Person ReID embeddings on the paired look-alike open-set protocol and matched control."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Ensure src/ is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reid.eval.lookalike import (
    evaluate_partition,
    evaluate_threshold_transfer,
    partition_lookalike_split,
    partition_matched_control_split,
)
from reid.eval.metrics import compute_openset_single_seed_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ReID embeddings on look-alike and matched control open-set protocols."
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=None,
        help="Path to directory containing saved query and gallery .npy embeddings.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="results/lookalike/groups_p05.json",
        help="Path to look-alike groups JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for evaluation JSON (defaults to results/<exp>/lookalike_v2.json).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Deterministic seeds for multi-seed evaluation (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Automatically evaluate all existing experiment directories in artifacts/.",
    )
    return parser.parse_args()


def aggregate_protocol_seeds(
    seed_results: Sequence[Dict[str, Any]],
    far_targets: Sequence[float] = (0.001, 0.01, 0.05, 0.10),
    tar_targets: Sequence[float] = (0.70, 0.80, 0.85, 0.90, 0.95),
) -> Dict[str, Any]:
    """Aggregate per-seed evaluation dictionaries into mean and std summary."""
    n_seeds = len(seed_results)
    summary: Dict[str, Any] = {}

    scalar_keys = [
        "auc",
        "eer",
        "max_tar",
        "mean_max_impostor_similarity",
        "num_enrolled_identities",
        "num_unknown_identities",
        "num_extra_gallery_identities",
        "num_target_probes",
        "num_impostor_probes",
    ]
    for k in scalar_keys:
        vals = [r[k] for r in seed_results if k in r and r[k] is not None]
        if vals:
            summary[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            }

    # FAR @ TAR
    for t in tar_targets:
        t_key = f"far_at_tar_{int(t*100)}"
        vals = []
        for r in seed_results:
            fat = r.get("far_at_tar", {})
            v = fat.get(t) if t in fat else fat.get(str(t))
            if v is not None:
                vals.append(v)
        if len(vals) == n_seeds:
            summary[t_key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            }
        elif len(vals) > 0:
            summary[t_key] = (
                f"Partial ({len(vals)}/{n_seeds} reached {int(t*100)}% TAR)"
            )
        else:
            mean_max_tar = summary.get("max_tar", {}).get("mean", 0.0)
            summary[t_key] = f"N/A (max TAR = {mean_max_tar * 100:.2f}%)"

    # DIR @ FAR
    for f in far_targets:
        f_label = str(f * 100).replace(".", "_").rstrip("0").rstrip("_")
        f_key = f"dir_at_far_{f_label}"
        vals = []
        for r in seed_results:
            daf = r.get("dir_at_far", {})
            v = daf.get(f) if f in daf else daf.get(str(f))
            if v is not None:
                vals.append(v)
        if vals:
            summary[f_key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            }

    return summary


def evaluate_embeddings_dir(
    emb_dir: Path,
    groups_data: Dict[str, Any],
    seeds: Sequence[int],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full look-alike, matched control, and threshold transfer evaluation on one experiment."""
    q_feats_file = emb_dir / "query_embeddings.npy"
    q_pids_file = emb_dir / "query_pids.npy"
    q_cams_file = emb_dir / "query_camids.npy"
    g_feats_file = emb_dir / "gallery_embeddings.npy"
    g_pids_file = emb_dir / "gallery_pids.npy"
    g_cams_file = emb_dir / "gallery_camids.npy"

    for p in (
        q_feats_file,
        q_pids_file,
        q_cams_file,
        g_feats_file,
        g_pids_file,
        g_cams_file,
    ):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required embedding array: {p}")

    q_feats = np.load(q_feats_file)
    q_pids = np.load(q_pids_file)
    q_cams = np.load(q_cams_file)
    g_feats = np.load(g_feats_file)
    g_pids = np.load(g_pids_file)
    g_cams = np.load(g_cams_file)

    groups = groups_data["groups"]
    percentile = groups_data.get("percentile", 5.0)
    all_test_pids = sorted(set(q_pids[q_pids > 0]) | set(g_pids[g_pids > 0]))

    lookalike_seed_results = []
    control_seed_results = []
    transfer_seed_results = []
    std_seed_results = []

    for seed in seeds:
        # 1. Look-Alike Partition
        lk_enrolled, lk_unknown, lk_extra = partition_lookalike_split(
            groups=groups, all_test_pids=all_test_pids, seed=seed
        )
        lk_metrics = evaluate_partition(
            q_feats,
            q_pids,
            q_cams,
            g_feats,
            g_pids,
            g_cams,
            enrolled_pids=lk_enrolled,
            unknown_pids=lk_unknown,
            extra_gallery_pids=lk_extra,
        )
        lk_metrics["seed"] = seed
        lookalike_seed_results.append(lk_metrics)

        # 2. Matched Random Control Partition
        num_enrolled = len(lk_enrolled)
        num_unknown = len(lk_unknown)
        ctrl_enrolled, ctrl_unknown, ctrl_extra = partition_matched_control_split(
            num_enrolled=num_enrolled,
            num_unknown=num_unknown,
            all_test_pids=all_test_pids,
            seed=seed,
        )
        ctrl_metrics = evaluate_partition(
            q_feats,
            q_pids,
            q_cams,
            g_feats,
            g_pids,
            g_cams,
            enrolled_pids=ctrl_enrolled,
            unknown_pids=ctrl_unknown,
            extra_gallery_pids=ctrl_extra,
        )
        ctrl_metrics["seed"] = seed
        control_seed_results.append(ctrl_metrics)

        # 3. Standard Evaluator v2 Split (for reference & threshold transfer)
        std_metrics = compute_openset_single_seed_v2(
            q_feats,
            q_pids,
            q_cams,
            g_feats,
            g_pids,
            g_cams,
            known_ratio=0.5,
            seed=seed,
        )
        std_seed_results.append(std_metrics)

        # 4. Threshold Transfer (standard tau_80 -> lookalike)
        transfer_res = evaluate_threshold_transfer(
            std_metrics=std_metrics, lookalike_metrics=lk_metrics, target_tar=0.80
        )
        transfer_res["seed"] = seed
        transfer_seed_results.append(transfer_res)

    lk_summary = aggregate_protocol_seeds(lookalike_seed_results)
    ctrl_summary = aggregate_protocol_seeds(control_seed_results)
    std_summary = aggregate_protocol_seeds(std_seed_results)

    # Transfer summary
    valid_transfer_fars = [
        r["transfer_far"]
        for r in transfer_seed_results
        if r["transfer_far"] is not None
    ]
    valid_transfer_tars = [
        r["transfer_tar"]
        for r in transfer_seed_results
        if r["transfer_tar"] is not None
    ]
    valid_taus = [
        r["standard_tau_80"]
        for r in transfer_seed_results
        if r["standard_tau_80"] is not None
    ]

    transfer_summary = {
        "operating_point_standard_tar": 0.80,
        "standard_tau_80": {
            "mean": float(np.mean(valid_taus)) if valid_taus else None,
            "std": float(np.std(valid_taus)) if len(valid_taus) > 1 else 0.0,
        },
        "transfer_far": {
            "mean": (
                float(np.mean(valid_transfer_fars)) if valid_transfer_fars else None
            ),
            "std": (
                float(np.std(valid_transfer_fars))
                if len(valid_transfer_fars) > 1
                else 0.0
            ),
        },
        "transfer_tar": {
            "mean": (
                float(np.mean(valid_transfer_tars)) if valid_transfer_tars else None
            ),
            "std": (
                float(np.std(valid_transfer_tars))
                if len(valid_transfer_tars) > 1
                else 0.0
            ),
        },
    }

    # Format serializable output without huge numpy arrays
    def _strip_raw(res_list: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clean_list = []
        for r in res_list:
            c = {
                k: v
                for k, v in r.items()
                if not k.startswith("raw_")
                and k not in ("thresholds", "far_curve", "tar_curve")
            }
            clean_list.append(c)
        return clean_list

    out_json = {
        "embeddings_dir": str(emb_dir),
        "experiment": emb_dir.name,
        "percentile": percentile,
        "seeds": list(seeds),
        "group_statistics": {
            "distance_threshold": groups_data.get("distance_threshold"),
            "global_median_distance": groups_data.get("global_median_distance"),
            "num_groups": groups_data.get("num_groups"),
            "total_identities_in_groups": groups_data.get("total_identities_in_groups"),
            "group_size_histogram": groups_data.get("group_size_histogram"),
            "mean_intra_group_distance": groups_data.get("mean_intra_group_distance"),
            "max_intra_group_distance": groups_data.get("max_intra_group_distance"),
        },
        "lookalike_protocol": {
            "summary": lk_summary,
            "per_seed": _strip_raw(lookalike_seed_results),
        },
        "matched_control_protocol": {
            "summary": ctrl_summary,
            "per_seed": _strip_raw(control_seed_results),
        },
        "standard_protocol_reference": {
            "summary": std_summary,
            "per_seed": _strip_raw(std_seed_results),
        },
        "threshold_transfer": {
            "summary": transfer_summary,
            "per_seed": transfer_seed_results,
        },
    }

    if output_path is None:
        p_tag = f"p{int(round(percentile)):02d}"
        if round(percentile) == 5:
            output_path = PROJECT_ROOT / "results" / emb_dir.name / "lookalike_v2.json"
        else:
            output_path = (
                PROJECT_ROOT / "results" / emb_dir.name / f"lookalike_{p_tag}.json"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"[Lookalike] Saved evaluation to: {output_path}")

    # Print Difficulty Check Comparison
    print("\n" + "=" * 65)
    print(f"  DIFFICULTY CHECK: {emb_dir.name} (percentile={percentile}%)")
    print("=" * 65)
    lk_far80 = lk_summary.get("far_at_tar_80", {})
    ctrl_far80 = ctrl_summary.get("far_at_tar_80", {})
    std_far80 = std_summary.get("far_at_tar_80", {})

    lk_dir1 = lk_summary.get("dir_at_far_1", {})
    ctrl_dir1 = ctrl_summary.get("dir_at_far_1", {})
    std_dir1 = std_summary.get("dir_at_far_1", {})

    lk_imp = lk_summary.get("mean_max_impostor_similarity", {})
    ctrl_imp = ctrl_summary.get("mean_max_impostor_similarity", {})

    def _fmt(v):
        if isinstance(v, dict) and "mean" in v and "std" in v:
            return f"{v['mean']*100:.2f}% ± {v['std']*100:.2f}%"
        if isinstance(v, str):
            return v
        return str(v)

    def _fmt_sim(v):
        if isinstance(v, dict) and "mean" in v and "std" in v:
            return f"{v['mean']:.4f} ± {v['std']:.4f}"
        return str(v)

    print(
        "Metric                 | Look-Alike       | Matched Control  | Standard v2 (Ref)"
    )
    print(
        "-----------------------|------------------|------------------|------------------"
    )
    print(
        f"FAR @ TAR=80%          | {_fmt(lk_far80):<16} | {_fmt(ctrl_far80):<16} | {_fmt(std_far80):<16}"
    )
    print(
        f"DIR @ FAR=1%           | {_fmt(lk_dir1):<16} | {_fmt(ctrl_dir1):<16} | {_fmt(std_dir1):<16}"
    )
    print(
        f"Max Impostor Sim       | {_fmt_sim(lk_imp):<16} | {_fmt_sim(ctrl_imp):<16} | N/A"
    )
    print(
        f"Max Achievable TAR     | {_fmt(lk_summary.get('max_tar', {})):<16} | {_fmt(ctrl_summary.get('max_tar', {})):<16} | {_fmt(std_summary.get('max_tar', {})):<16}"
    )
    print(
        f"Transfer FAR (at tau80)| {_fmt(transfer_summary.get('transfer_far', {})):<16} | N/A              | N/A"
    )
    print(
        f"Transfer TAR (at tau80)| {_fmt(transfer_summary.get('transfer_tar', {})):<16} | N/A              | 80.00% ± 0.00%"
    )
    print("=" * 65 + "\n")

    return out_json


def main() -> None:
    args = parse_args()
    groups_path = Path(args.groups)
    if not groups_path.is_file():
        raise FileNotFoundError(
            f"Groups file not found: {groups_path}. Run scripts/build_lookalike_groups.py first."
        )

    with open(groups_path, "r", encoding="utf-8") as f:
        groups_data = json.load(f)

    if args.run_all:
        artifacts_dir = PROJECT_ROOT / "artifacts"
        available_dirs = [
            d
            for d in artifacts_dir.iterdir()
            if d.is_dir() and (d / "query_embeddings.npy").is_file()
        ]
        print(
            f"[Lookalike] Found {len(available_dirs)} artifact directories to evaluate."
        )
        for d in sorted(available_dirs):
            print(f"\n>>> Evaluating {d.name} on {groups_path.name}...")
            evaluate_embeddings_dir(
                emb_dir=d, groups_data=groups_data, seeds=args.seeds
            )
    else:
        if args.embeddings_dir is None:
            raise ValueError("Must specify --embeddings-dir or --run-all")
        emb_dir = Path(args.embeddings_dir)
        output_path = Path(args.output) if args.output else None
        evaluate_embeddings_dir(
            emb_dir=emb_dir,
            groups_data=groups_data,
            seeds=args.seeds,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
