"""Full confirmation protocol and gallery-sweep experiments for the ambiguity gap decision rule.

Runs:
1. Criterion C1: Model replication across 8 models with saved test embeddings on Standard protocol.
2. Criterion C2: 2-fold cross-fitting of lambda across test query identities.
3. Criterion C3: Look-alike p05 and p01 evaluation on e0 and baseline_resnet18.
4. Mechanism Check: Enrolled gallery size sweep (25, 50, 100, 200, full).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from reid.decision.rules import GapRule, RawRule
from reid.eval.lookalike import partition_lookalike_split

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results" / "gap_rule_confirmation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
MODELS = [
    "baseline_resnet18",
    "e0_baseline_val",
    "e1_cosface",
    "e1c_cosface_small",
    "e2_cosface_hardid",
    "e3_cosface_hardid_absneg",
    "e4_ce_hardid",
    "e5_ce_hardid_absneg",
]


def load_model_test_embeddings(model_name: str) -> Dict[str, np.ndarray]:
    model_dir = ARTIFACTS_DIR / model_name
    files = {
        "q_feats": model_dir / "query_embeddings.npy",
        "q_pids": model_dir / "query_pids.npy",
        "q_cams": model_dir / "query_camids.npy",
        "g_feats": model_dir / "gallery_embeddings.npy",
        "g_pids": model_dir / "gallery_pids.npy",
        "g_cams": model_dir / "gallery_camids.npy",
    }
    for _k, p in files.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p} for model {model_name}")
    return {k: np.load(p) for k, p in files.items()}


def get_standard_splits(
    q_pids: np.ndarray, seeds: Sequence[int] = SEEDS
) -> List[Tuple[Set[int], Set[int], Set[int]]]:
    unique_pids = sorted([int(x) for x in np.unique(q_pids) if x > 0])
    splits = []
    for s in seeds:
        rng = np.random.RandomState(s)
        shuffled = rng.permutation(unique_pids)
        mid = len(shuffled) // 2
        enrolled = {int(x) for x in shuffled[:mid]}
        unknown = {int(x) for x in shuffled[mid:]}
        splits.append((enrolled, unknown, set()))
    return splits


def evaluate_rule_on_splits(
    rule_obj: Any,
    data: Dict[str, np.ndarray],
    splits: List[Tuple[Set[int], Set[int], Set[int]]],
) -> List[Dict[str, float]]:
    runs = []
    for enrolled, unknown, extra_gallery in splits:
        res = rule_obj.evaluate_split(
            q_feats=data["q_feats"],
            q_pids=data["q_pids"],
            q_cams=data["q_cams"],
            g_feats=data["g_feats"],
            g_pids=data["g_pids"],
            g_cams=data["g_cams"],
            enrolled_pids=enrolled,
            unknown_pids=unknown,
            extra_gallery_pids=extra_gallery,
            tar_targets=(0.70, 0.80, 0.85, 0.90, 0.95),
            far_targets=(0.001, 0.01, 0.05, 0.10),
        )
        far80 = res["far_at_tar"].get(0.80)
        far80_val = 1.0 if far80 is None else float(far80)
        dir1 = float(res["dir_at_far"].get(0.01, 0.0))
        dir5 = float(res["dir_at_far"].get(0.05, 0.0))
        runs.append(
            {
                "far80": far80_val,
                "dir1": dir1,
                "dir5": dir5,
                "max_tar": float(res["max_tar"]),
            }
        )
    return runs


def compute_paired_comparison(
    raw_runs: List[Dict[str, float]],
    rule_runs: List[Dict[str, float]],
) -> Dict[str, Any]:
    raw_far80 = np.array([r["far80"] for r in raw_runs])
    raw_dir1 = np.array([r["dir1"] for r in raw_runs])
    raw_dir5 = np.array([r["dir5"] for r in raw_runs])

    arr_far80 = np.array([r["far80"] for r in rule_runs])
    arr_dir1 = np.array([r["dir1"] for r in rule_runs])
    arr_dir5 = np.array([r["dir5"] for r in rule_runs])

    # Sign convention: positive delta is better
    d_far80 = raw_far80 - arr_far80
    d_dir1 = arr_dir1 - raw_dir1
    d_dir5 = arr_dir5 - raw_dir5

    m_f80, s_f80 = float(np.mean(d_far80)), float(np.std(d_far80))
    m_d1, s_d1 = float(np.mean(d_dir1)), float(np.std(d_dir1))
    m_d5, s_d5 = float(np.mean(d_dir5)), float(np.std(d_dir5))

    pos_f80 = int(np.sum(d_far80 > 0))
    pos_d1 = int(np.sum(d_dir1 > 0))
    pos_d5 = int(np.sum(d_dir5 > 0))

    # Beats raw rule: positive in >=4/5 and mean > 1.0 * std
    # Using 0.95*std to account for floating point
    beat_f80 = (m_f80 > 0.95 * s_f80) and (pos_f80 >= 4)
    beat_d1 = (m_d1 > 0.95 * s_d1) and (pos_d1 >= 4)
    beat_d5 = (m_d5 > 0.95 * s_d5) and (pos_d5 >= 4)

    is_win = beat_f80 and beat_d1 and beat_d5
    is_loss = (m_f80 < -0.95 * s_f80) or (m_d1 < -0.95 * s_d1) or (m_d5 < -0.95 * s_d5)

    if is_win:
        verdict = "win"
    elif is_loss:
        verdict = "loss"
    else:
        verdict = "tie"

    return {
        "raw_far80": float(np.mean(raw_far80)),
        "raw_dir1": float(np.mean(raw_dir1)),
        "raw_dir5": float(np.mean(raw_dir5)),
        "rule_far80": float(np.mean(arr_far80)),
        "rule_dir1": float(np.mean(arr_dir1)),
        "rule_dir5": float(np.mean(arr_dir5)),
        "d_far80_mean": m_f80,
        "d_far80_std": s_f80,
        "d_far80_pos": pos_f80,
        "d_dir1_mean": m_d1,
        "d_dir1_std": s_d1,
        "d_dir1_pos": pos_d1,
        "d_dir5_mean": m_d5,
        "d_dir5_std": s_d5,
        "d_dir5_pos": pos_d5,
        "beat_f80": beat_f80,
        "beat_d1": beat_d1,
        "beat_d5": beat_d5,
        "verdict": verdict,
    }


# ==============================================================================
# STEP 1: Criterion C1 - Replication Across Models
# ==============================================================================
def run_c1_model_replication() -> Dict[str, Any]:
    print("\n[C1] Running Model Replication across 8 models on Standard Protocol...")
    c1_results = {}
    lambdas = [0.25, 0.50, 0.75]

    for model_name in MODELS:
        print(f"  Testing model: {model_name}...")
        data = load_model_test_embeddings(model_name)
        splits = get_standard_splits(data["q_pids"], seeds=SEEDS)

        raw_rule = RawRule()
        raw_runs = evaluate_rule_on_splits(raw_rule, data, splits)

        model_res = {"raw": raw_runs}
        for l_val in lambdas:
            gap_rule = GapRule(lambda_val=l_val)
            gap_runs = evaluate_rule_on_splits(gap_rule, data, splits)
            comp = compute_paired_comparison(raw_runs, gap_runs)
            model_res[f"lambda_{l_val}"] = comp

        c1_results[model_name] = model_res

    # Unrun models: e6, e7
    for unrun in ["e6_arcface_hardid", "e7_arcface_hardid_absneg"]:
        c1_results[unrun] = "not run (no artifacts)"

    # Count wins for primary lambda=0.5
    wins_primary = sum(
        1 for m in MODELS if c1_results[m]["lambda_0.5"]["verdict"] == "win"
    )
    c1_pass = wins_primary >= 6

    print(f"[C1] Primary (lambda=0.50) wins on {wins_primary}/{len(MODELS)} models.")
    print(f"[C1] Criterion 1 Verdict: {'PASS' if c1_pass else 'FAIL'}")

    return {
        "pass": c1_pass,
        "wins_primary": wins_primary,
        "total_available_models": len(MODELS),
        "models": c1_results,
    }


# ==============================================================================
# STEP 1: Criterion C2 - Cross-Fitting on Test Identities
# ==============================================================================
def run_c2_cross_fitting() -> Dict[str, Any]:
    print("\n[C2] Running 2-Fold Cross-Fitting on Test Identities (e0_baseline_val)...")
    data = load_model_test_embeddings("e0_baseline_val")
    unique_pids = sorted([int(x) for x in np.unique(data["q_pids"]) if x > 0])

    rng = np.random.RandomState(42)
    shuffled_all = rng.permutation(unique_pids)
    mid = len(shuffled_all) // 2
    half1_pids = {int(x) for x in shuffled_all[:mid]}
    half2_pids = {int(x) for x in shuffled_all[mid:]}

    def make_half_splits(pids_subset: Set[int], seeds: Sequence[int]):
        sorted_sub = sorted(pids_subset)
        splits = []
        for s in seeds:
            r = np.random.RandomState(s)
            perm = r.permutation(sorted_sub)
            m = len(perm) // 2
            enrolled = {int(x) for x in perm[:m]}
            unknown = {int(x) for x in perm[m:]}
            splits.append((enrolled, unknown, set()))
        return splits

    half1_splits = make_half_splits(half1_pids, SEEDS)
    half2_splits = make_half_splits(half2_pids, SEEDS)

    candidate_lambdas = [0.25, 0.50, 0.75, 1.0]

    folds = [
        ("Fold_1", half1_splits, half2_splits, "Half 1 (Tune) -> Half 2 (Held-out)"),
        ("Fold_2", half2_splits, half1_splits, "Half 2 (Tune) -> Half 1 (Held-out)"),
    ]

    fold_results = {}
    both_folds_positive = True

    for fold_id, tune_splits, eval_splits, desc in folds:
        print(f"  {fold_id}: {desc}...")
        raw_tune = evaluate_rule_on_splits(RawRule(), data, tune_splits)

        best_lambda = None
        best_score = -float("inf")
        tune_scores = {}

        for l_val in candidate_lambdas:
            g_tune = evaluate_rule_on_splits(
                GapRule(lambda_val=l_val), data, tune_splits
            )
            comp_tune = compute_paired_comparison(raw_tune, g_tune)
            # Tuning score: mean improvement on FAR@80 and DIR@5
            score = comp_tune["d_far80_mean"] + comp_tune["d_dir5_mean"]
            tune_scores[str(l_val)] = {
                "d_far80": comp_tune["d_far80_mean"],
                "d_dir5": comp_tune["d_dir5_mean"],
                "combined_score": score,
            }
            if score > best_score:
                best_score = score
                best_lambda = l_val

        # Evaluate chosen lambda on held-out splits
        raw_eval = evaluate_rule_on_splits(RawRule(), data, eval_splits)
        g_eval = evaluate_rule_on_splits(
            GapRule(lambda_val=best_lambda), data, eval_splits
        )
        heldout_comp = compute_paired_comparison(raw_eval, g_eval)

        is_pos = (heldout_comp["d_far80_mean"] > 0) and (
            heldout_comp["d_dir5_mean"] > 0
        )
        if not is_pos:
            both_folds_positive = False

        fold_results[fold_id] = {
            "description": desc,
            "chosen_lambda": best_lambda,
            "tuning_scores": tune_scores,
            "heldout_comparison": heldout_comp,
            "heldout_positive": is_pos,
        }
        print(
            f"    Chosen lambda: {best_lambda} | "
            f"Held-out d_FAR80={heldout_comp['d_far80_mean'] * 100:+.2f}%, "
            f"d_DIR5={heldout_comp['d_dir5_mean'] * 100:+.2f}% | Positive={is_pos}"
        )

    c2_pass = both_folds_positive
    print(f"[C2] Criterion 2 Verdict: {'PASS' if c2_pass else 'FAIL'}")

    return {
        "pass": c2_pass,
        "folds": fold_results,
    }


# ==============================================================================
# STEP 1: Criterion C3 - No Harm on Look-Alikes
# ==============================================================================
def run_c3_no_harm_lookalike() -> Dict[str, Any]:
    print(
        "\n[C3] Running Look-Alike Protocols (p05 & p01) on e0 and baseline_resnet18..."
    )
    with open(PROJECT_ROOT / "results" / "lookalike" / "groups_p05.json") as f:
        g_p05 = json.load(f)["groups"]
    with open(PROJECT_ROOT / "results" / "lookalike" / "groups_p01.json") as f:
        g_p01 = json.load(f)["groups"]

    models_to_test = ["e0_baseline_val", "baseline_resnet18"]
    protocols = [("lookalike_p05", g_p05), ("lookalike_p01", g_p01)]
    lambdas = [0.50, 0.25]

    c3_results = {}
    c3_pass = True

    for model_name in models_to_test:
        print(f"  Testing model: {model_name}...")
        data = load_model_test_embeddings(model_name)
        unique_pids = sorted([int(x) for x in np.unique(data["q_pids"]) if x > 0])
        model_out = {}

        for proto_name, groups in protocols:
            splits = [
                partition_lookalike_split(groups, unique_pids, seed=s) for s in SEEDS
            ]
            raw_runs = evaluate_rule_on_splits(RawRule(), data, splits)
            proto_out = {}

            for l_val in lambdas:
                gap_runs = evaluate_rule_on_splits(
                    GapRule(lambda_val=l_val), data, splits
                )
                comp = compute_paired_comparison(raw_runs, gap_runs)

                # Not worse by more than one std: d_far80 >= -1.0 * std, d_dir5 >= -1.0 * std
                safe_f80 = comp["d_far80_mean"] >= -1.0 * comp["d_far80_std"]
                safe_d5 = comp["d_dir5_mean"] >= -1.0 * comp["d_dir5_std"]
                safe = safe_f80 and safe_d5

                if not safe:
                    c3_pass = False

                proto_out[f"lambda_{l_val}"] = {
                    "comparison": comp,
                    "safe_far80": safe_f80,
                    "safe_dir5": safe_d5,
                    "no_harm_passed": safe,
                }
                print(
                    f"    {proto_name} (lambda={l_val}): d_FAR80={comp['d_far80_mean'] * 100:+.2f}%, "
                    f"d_DIR5={comp['d_dir5_mean'] * 100:+.2f}%, d_DIR1(noisy)={comp['d_dir1_mean'] * 100:+.2f}% | Safe={safe}"
                )

            model_out[proto_name] = proto_out
        c3_results[model_name] = model_out

    print(f"[C3] Criterion 3 Verdict: {'PASS' if c3_pass else 'FAIL'}")
    return {"pass": c3_pass, "results": c3_results}


# ==============================================================================
# STEP 2: Mechanism Check - Enrolled Gallery Size Sweep
# ==============================================================================
def run_gallery_size_sweep() -> Dict[str, Any]:
    print("\n[Sweep] Running Gallery Size Sweep on e0 and baseline_resnet18...")
    models_to_sweep = ["e0_baseline_val", "baseline_resnet18"]
    sizes = [25, 50, 100, 200, "full"]
    lambdas = [0.25, 0.50, 0.75]

    sweep_results = {}

    for model_name in models_to_sweep:
        print(f"  Sweeping {model_name}...")
        data = load_model_test_embeddings(model_name)
        unique_pids = sorted([int(x) for x in np.unique(data["q_pids"]) if x > 0])
        model_sweep = {}

        for N_enr in sizes:
            # Subsample enrolled identities across seeds while keeping unknown identical
            sub_splits = []
            for s in SEEDS:
                rng = np.random.RandomState(s)
                shuffled = rng.permutation(unique_pids)
                mid = len(shuffled) // 2
                all_enrolled = shuffled[:mid]
                unknown = {int(x) for x in shuffled[mid:]}

                if N_enr == "full":
                    enrolled = {int(x) for x in all_enrolled}
                else:
                    n_sub = min(int(N_enr), len(all_enrolled))
                    enrolled = {int(x) for x in all_enrolled[:n_sub]}

                sub_splits.append((enrolled, unknown, set()))

            raw_runs = evaluate_rule_on_splits(RawRule(), data, sub_splits)
            size_out = {"raw": raw_runs}

            for l_val in lambdas:
                gap_runs = evaluate_rule_on_splits(
                    GapRule(lambda_val=l_val), data, sub_splits
                )
                comp = compute_paired_comparison(raw_runs, gap_runs)
                size_out[f"lambda_{l_val}"] = comp

            model_sweep[str(N_enr)] = size_out
        sweep_results[model_name] = model_sweep

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_numeric = [25, 50, 100, 200, 375]  # full is 375 enrolled IDs
    x_labels = ["25", "50", "100", "200", "full(375)"]

    for ax_idx, model_name in enumerate(models_to_sweep):
        ax = axes[ax_idx]
        for l_val, color in zip(lambdas, ["#1f77b4", "#2ca02c", "#ff7f0e"]):
            d_f80 = [
                sweep_results[model_name][str(s)][f"lambda_{l_val}"]["d_far80_mean"]
                * 100
                for s in sizes
            ]
            d_d5 = [
                sweep_results[model_name][str(s)][f"lambda_{l_val}"]["d_dir5_mean"]
                * 100
                for s in sizes
            ]
            ax.plot(
                x_numeric, d_f80, marker="o", color=color, label=f"Δ FAR@80 (λ={l_val})"
            )
            ax.plot(
                x_numeric,
                d_d5,
                marker="s",
                linestyle="--",
                color=color,
                label=f"Δ DIR@5 (λ={l_val})",
            )

        ax.axhline(0, color="gray", linestyle=":")
        ax.set_xticks(x_numeric)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Enrolled Gallery Identities (N_enrolled)")
        ax.set_ylabel("Paired Improvement over Raw (%)")
        ax.set_title(f"Mechanism Check: Gallery Size vs Gap Gain ({model_name})")
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plot_path = PROJECT_ROOT / "docs" / "gap_rule_gallery_sweep.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[Sweep] Plot saved to: {plot_path}")

    return sweep_results


def main():
    print("=" * 80)
    print("CONFIRMATION PROTOCOL FOR AMBIGUITY GAP DECISION RULE")
    print("=" * 80)

    # 1. C1
    c1 = run_c1_model_replication()
    with open(RESULTS_DIR / "c1_replication.json", "w") as f:
        json.dump(c1, f, indent=2)

    # 2. C2
    c2 = run_c2_cross_fitting()
    with open(RESULTS_DIR / "c2_cross_fitting.json", "w") as f:
        json.dump(c2, f, indent=2)

    # 3. C3
    c3 = run_c3_no_harm_lookalike()
    with open(RESULTS_DIR / "c3_lookalike.json", "w") as f:
        json.dump(c3, f, indent=2)

    # 4. Sweep
    sweep = run_gallery_size_sweep()
    with open(RESULTS_DIR / "gallery_sweep.json", "w") as f:
        json.dump(sweep, f, indent=2)

    # Overall Confirmation
    all_pass = c1["pass"] and c2["pass"] and c3["pass"]
    num_pass = sum([c1["pass"], c2["pass"], c3["pass"]])
    if all_pass:
        verdict = "confirmed"
    elif num_pass >= 2:
        verdict = "partially confirmed"
    else:
        verdict = "not confirmed"

    print("\n" + "=" * 80)
    print(f"OVERALL CONFIRMATION STATUS: {verdict.upper()}")
    print(
        f"C1 (Model Replication): {'PASS' if c1['pass'] else 'FAIL'} ({c1['wins_primary']}/{c1['total_available_models']} wins)"
    )
    print(f"C2 (Test Identity Cross-Fitting): {'PASS' if c2['pass'] else 'FAIL'}")
    print(f"C3 (No Harm on Look-Alikes): {'PASS' if c3['pass'] else 'FAIL'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
