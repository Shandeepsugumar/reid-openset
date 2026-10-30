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

    clean_per_seed = []
    for r in per_seed:
        clean_per_seed.append(
            {
                "max_tar": float(r["max_tar"]),
                "auc": float(r["auc"]),
                "eer": float(r["eer"]),
                "far_at_tar": {
                    str(k): (float(v) if v is not None else None)
                    for k, v in r["far_at_tar"].items()
                },
                "dir_at_far": {str(k): float(v) for k, v in r["dir_at_far"].items()},
            }
        )

    return {"summary": agg, "per_seed": clean_per_seed}


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
            test_results[proto_name][rule_name] = {
                "summary": res["summary"],
                "per_seed": res["per_seed"],
            }

        # Compute paired differences against raw
        def _get_far80(run_dict: Dict[str, Any]) -> float:
            f_dict = run_dict["far_at_tar"]
            val = f_dict.get("0.8", f_dict.get(0.80, 1.0))
            return 1.0 if val is None else float(val)

        def _get_dir1(run_dict: Dict[str, Any]) -> float:
            d_dict = run_dict["dir_at_far"]
            return float(d_dict.get("0.01", d_dict.get(0.01, 0.0)))

        def _get_dir5(run_dict: Dict[str, Any]) -> float:
            d_dict = run_dict["dir_at_far"]
            return float(d_dict.get("0.05", d_dict.get(0.05, 0.0)))

        raw_runs = test_results[proto_name]["raw"]["per_seed"]
        raw_far80 = np.array([_get_far80(r) for r in raw_runs])
        raw_dir1 = np.array([_get_dir1(r) for r in raw_runs])
        raw_dir5 = np.array([_get_dir5(r) for r in raw_runs])

        print(f"\n  --- PAIRED DIFFERENCES VS RAW: {proto_name} ---")
        print(
            f"  {'Rule':<16} | {'FAR@80 diff':<18} | {'DIR@1 diff':<18} | {'DIR@5 diff':<18} | Verdict"
        )
        print("  " + "-" * 85)

        for rule_name, rule_data in test_results[proto_name].items():
            runs = rule_data["per_seed"]
            arr_far80 = np.array([_get_far80(r) for r in runs])
            arr_dir1 = np.array([_get_dir1(r) for r in runs])
            arr_dir5 = np.array([_get_dir5(r) for r in runs])

            # Positive means improvement: lower FAR, higher DIR
            d_far80 = raw_far80 - arr_far80
            d_dir1 = arr_dir1 - raw_dir1
            d_dir5 = arr_dir5 - raw_dir5

            m_far80, s_far80 = float(np.mean(d_far80)), float(np.std(d_far80))
            m_dir1, s_dir1 = float(np.mean(d_dir1)), float(np.std(d_dir1))
            m_dir5, s_dir5 = float(np.mean(d_dir5)), float(np.std(d_dir5))

            pos_far80 = int(np.sum(d_far80 > 0))
            pos_dir1 = int(np.sum(d_dir1 > 0))
            pos_dir5 = int(np.sum(d_dir5 > 0))

            # Improvement rule:
            # All of FAR@TAR=80, DIR@FAR=1%, DIR@FAR=5% better by more than about one std,
            # positive paired sign in at least 4 of 5 seeds; otherwise "tie" or "loss"
            beat_far80 = (m_far80 > 0.95 * s_far80) and (pos_far80 >= 4)
            beat_dir1 = (m_dir1 > 0.95 * s_dir1) and (pos_dir1 >= 4)
            beat_dir5 = (m_dir5 > 0.95 * s_dir5) and (pos_dir5 >= 4)

            if rule_name == "raw":
                verdict = "baseline"
            elif beat_far80 and beat_dir1 and beat_dir5:
                verdict = "win"
            elif (
                (m_far80 < -0.95 * s_far80)
                or (m_dir1 < -0.95 * s_dir1)
                or (m_dir5 < -0.95 * s_dir5)
            ):
                verdict = "loss"
            else:
                verdict = "tie"

            rule_data["paired_vs_raw"] = {
                "d_far80_mean": m_far80,
                "d_far80_std": s_far80,
                "d_far80_pos": pos_far80,
                "d_dir1_mean": m_dir1,
                "d_dir1_std": s_dir1,
                "d_dir1_pos": pos_dir1,
                "d_dir5_mean": m_dir5,
                "d_dir5_std": s_dir5,
                "d_dir5_pos": pos_dir5,
                "verdict": verdict,
            }

            s_f = f"{m_far80 * 100:+5.2f}+-{s_far80 * 100:4.2f} ({pos_far80}/5)"
            s_d1 = f"{m_dir1 * 100:+5.2f}+-{s_dir1 * 100:4.2f} ({pos_dir1}/5)"
            s_d5 = f"{m_dir5 * 100:+5.2f}+-{s_dir5 * 100:4.2f} ({pos_dir5}/5)"
            print(
                f"  {rule_name:<16} | {s_f:<18} | {s_d1:<18} | {s_d5:<18} | {verdict}"
            )

    with open(out_dir / "test_evaluation.json", "w", encoding="utf-8") as tf:
        json.dump(test_results, tf, indent=2)

    print(f"\n[Decision Rules] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
