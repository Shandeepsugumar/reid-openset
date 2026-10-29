"""Evaluate inference-time decision rules on saved embeddings.

Evaluates raw baseline and alternative decision rules (gap, znorm, snorm, csls)
on the test set across Standard open-set, Look-Alike p05, and Look-Alike p01 protocols.
Cohort statistics are computed strictly from held-out validation embeddings (never test).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from reid.decision.rules import (
    get_all_candidate_rules,
    partition_validation_cohort,
    select_best_decision_rule_two_fold,
)
from reid.eval.lookalike import partition_lookalike_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate inference-time decision rules."
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="artifacts/e0_baseline_val",
        help="Directory containing saved query, gallery, and val embeddings (.npy).",
    )
    parser.add_argument(
        "--val-identities-json",
        type=str,
        default="results/e0_baseline_val/val_identities.json",
        help="Path to val_identities.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/decision_rules",
        help="Output directory to save evaluation metrics and comparison tables.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds for split evaluation.",
    )
    return parser.parse_args()


def load_data(exp_dir: Path) -> Dict[str, np.ndarray]:
    files = {
        "q_feats": exp_dir / "query_embeddings.npy",
        "q_pids": exp_dir / "query_pids.npy",
        "q_cams": exp_dir / "query_camids.npy",
        "g_feats": exp_dir / "gallery_embeddings.npy",
        "g_pids": exp_dir / "gallery_pids.npy",
        "g_cams": exp_dir / "gallery_camids.npy",
        "val_q_feats": exp_dir / "val_query_embeddings.npy",
        "val_q_pids": exp_dir / "val_query_pids.npy",
        "val_q_cams": exp_dir / "val_query_camids.npy",
        "val_g_feats": exp_dir / "val_gallery_embeddings.npy",
        "val_g_pids": exp_dir / "val_gallery_pids.npy",
        "val_g_cams": exp_dir / "val_gallery_camids.npy",
    }
    loaded = {}
    for k, p in files.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing required embedding file: {p}")
        loaded[k] = np.load(p)
    return loaded


def evaluate_rule_on_seeds(
    rule_name: str,
    rule_obj: Any,
    q_feats: np.ndarray,
    q_pids: np.ndarray,
    q_cams: np.ndarray,
    g_feats: np.ndarray,
    g_pids: np.ndarray,
    g_cams: np.ndarray,
    splits: List[Tuple[Set[int], Set[int], Set[int]]],
    cohort_feats: np.ndarray,
) -> Dict[str, Any]:
    per_seed = []
    tar_targets = (0.70, 0.80, 0.85, 0.90, 0.95)
    far_targets = (0.001, 0.01, 0.05, 0.10)

    for enrolled, unknown, extra_gallery in splits:
        res = rule_obj.evaluate_split(
            q_feats=q_feats,
            q_pids=q_pids,
            q_cams=q_cams,
            g_feats=g_feats,
            g_pids=g_pids,
            g_cams=g_cams,
            enrolled_pids=enrolled,
            unknown_pids=unknown,
            extra_gallery_pids=extra_gallery,
            cohort_feats=cohort_feats,
            tar_targets=tar_targets,
            far_targets=far_targets,
        )
        per_seed.append(res)

    # Compute aggregates
    max_tars = [r["max_tar"] for r in per_seed]
    aucs = [r["auc"] for r in per_seed]
    eers = [r["eer"] for r in per_seed]

    agg: Dict[str, Any] = {
        "max_tar": {"mean": float(np.mean(max_tars)), "std": float(np.std(max_tars))},
        "auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))},
        "eer": {"mean": float(np.mean(eers)), "std": float(np.std(eers))},
    }

    for t in tar_targets:
        vals = [r["far_at_tar"].get(t) for r in per_seed]
        valid_vals = [v for v in vals if v is not None]
        key = f"far_at_tar_{int(round(t * 100))}"
        if len(valid_vals) == len(splits):
            agg[key] = {
                "mean": float(np.mean(valid_vals)),
                "std": float(np.std(valid_vals)),
            }
        else:
            agg[key] = None

    for f in far_targets:
        vals = [r["dir_at_far"].get(f, 0.0) for r in per_seed]
        if f == 0.001:
            lbl = "dir_at_far_0_1"
        else:
            lbl = f"dir_at_far_{int(round(f * 100))}"
        agg[lbl] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    return {"summary": agg, "per_seed": per_seed}


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Decision Rules] Loading embeddings from {exp_dir}...")
    data = load_data(exp_dir)

    with open(args.val_identities_json, "r", encoding="utf-8") as vf:
        val_meta = json.load(vf)
    val_identities = val_meta["val_identities"]

    test_q_pids = np.unique(data["q_pids"])
    # Security isolation check
    overlap = set(val_identities) & set(test_q_pids)
    if overlap:
        raise AssertionError(f"Validation and test identities overlap! IDs: {overlap}")

    print(
        "[Decision Rules] Step 1: Performing 2-fold cross-validation rule selection..."
    )
    selection_result = select_best_decision_rule_two_fold(
        val_q_feats=data["val_q_feats"],
        val_q_pids=data["val_q_pids"],
        val_q_cams=data["val_q_cams"],
        val_g_feats=data["val_g_feats"],
        val_g_pids=data["val_g_pids"],
        val_g_cams=data["val_g_cams"],
        val_identities=val_identities,
        test_pids=data["q_pids"],
        seeds=args.seeds,
    )

    with open(out_dir / "val_selection.json", "w", encoding="utf-8") as sf:
        json.dump(selection_result, sf, indent=2)

    print(
        f"[Decision Rules] Validation selection complete: selected '{selection_result['selected_rule']}'"
    )

    # Cohort for test evaluation:
    # Use Half A (50 validation identities) as the cohort, completely disjoint from test
    half_a, half_b = partition_validation_cohort(val_identities, val_split_seed=42)
    all_val_feats = np.vstack([data["val_q_feats"], data["val_g_feats"]])
    all_val_pids = np.concatenate([data["val_q_pids"], data["val_g_pids"]])

    cohort_mask = np.isin(all_val_pids, half_a)
    cohort_feats = all_val_feats[cohort_mask]
    print(
        f"[Decision Rules] Test Cohort: {len(half_a)} validation identities ({len(cohort_feats)} samples)"
    )

    # Prepare protocols
    unique_test_pids = sorted([int(x) for x in np.unique(data["q_pids"]) if x > 0])

    # 1. Standard Protocol Splits
    standard_splits = []
    for s in args.seeds:
        rng = np.random.RandomState(s)
        shuffled = rng.permutation(unique_test_pids)
        mid = len(shuffled) // 2
        enrolled = {int(x) for x in shuffled[:mid]}
        unknown = {int(x) for x in shuffled[mid:]}
        # In standard open-set, remaining gallery (distractors etc.) are handled via non-unknown
        # extra_gallery is empty because all test gallery is enrolled/unknown/distractor
        standard_splits.append((enrolled, unknown, set()))

    # 2. Look-Alike p05 Splits
    with open("results/lookalike/groups_p05.json", "r", encoding="utf-8") as f:
        groups_p05 = json.load(f)["groups"]
    p05_splits = [
        partition_lookalike_split(groups_p05, unique_test_pids, seed=s)
        for s in args.seeds
    ]

    # 3. Look-Alike p01 Splits
    with open("results/lookalike/groups_p01.json", "r", encoding="utf-8") as f:
        groups_p01 = json.load(f)["groups"]
    p01_splits = [
        partition_lookalike_split(groups_p01, unique_test_pids, seed=s)
        for s in args.seeds
    ]

    protocols = {
        "standard": standard_splits,
        "lookalike_p05": p05_splits,
        "lookalike_p01": p01_splits,
    }

    candidate_rules = get_all_candidate_rules()
    test_results: Dict[str, Dict[str, Any]] = {proto: {} for proto in protocols}

    print(
        "[Decision Rules] Step 2: Evaluating all candidate rules on test set protocols..."
    )
    for proto_name, proto_splits in protocols.items():
        print(f"  Evaluating protocol: {proto_name}...")
        for rule_name, rule_obj in candidate_rules.items():
            res = evaluate_rule_on_seeds(
                rule_name=rule_name,
                rule_obj=rule_obj,
                q_feats=data["q_feats"],
                q_pids=data["q_pids"],
                q_cams=data["q_cams"],
                g_feats=data["g_feats"],
                g_pids=data["g_pids"],
                g_cams=data["g_cams"],
                splits=proto_splits,
                cohort_feats=cohort_feats,
            )
            test_results[proto_name][rule_name] = res["summary"]

    with open(out_dir / "test_evaluation.json", "w", encoding="utf-8") as tf:
        json.dump(test_results, tf, indent=2)

    print(f"[Decision Rules] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
