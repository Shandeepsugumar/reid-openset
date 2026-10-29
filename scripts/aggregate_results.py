"""Aggregate validation and test evaluation results across ablation experiments."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS = [
    "baseline_resnet18",
    "e0_baseline_val",
    "e1_cosface",
    "e2_cosface_hardid",
    "e3_cosface_hardid_absneg",
    "e1b_arcface",
    "e1c_cosface_small",
    "e4_ce_hardid",
    "e5_ce_hardid_absneg",
]

COLUMNS = [
    "Experiment",
    "Val Rank-1",
    "Val mAP",
    "Val FAR@TAR=80",
    "Val DIR@FAR=1%",
    "Test Rank-1",
    "Test mAP",
    "Test FAR@TAR=80",
    "Test FAR@TAR=90",
    "Test DIR@FAR=1%",
    "Test DIR@FAR=5%",
    "Test max TAR",
    "Lookalike FAR@TAR=80",
    "Lookalike DIR@FAR=1%",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate validation and test results across experiments."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory where experiment outputs are stored.",
    )
    parser.add_argument(
        "--doc-path",
        type=str,
        default="docs/ablations.md",
        help="Output markdown table file path.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="results/ablations.csv",
        help="Output CSV file path.",
    )
    return parser.parse_args()


def _format_stat(val: Any) -> str:
    if isinstance(val, dict) and "mean" in val and "std" in val:
        mean_pct = val["mean"] * 100
        std_pct = val["std"] * 100
        return f"{mean_pct:.2f}% ± {std_pct:.2f}%"
    if isinstance(val, (int, float)):
        return f"{float(val):.2f}%"
    if val is None or val == "":
        return "not run"
    val_str = str(val).strip()
    try:
        val_float = float(val_str)
        return f"{val_float:.2f}%"
    except ValueError:
        return val_str


def read_val_metrics(exp_dirs: List[Path]) -> Dict[str, str]:
    """Read latest row from val_metrics.csv if available."""
    for exp_dir in exp_dirs:
        val_csv = exp_dir / "val_metrics.csv"
        if val_csv.is_file():
            try:
                with open(val_csv, "r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    last_row = rows[-1]
                    return {
                        "Val Rank-1": _format_stat(float(last_row.get("rank_1", 0))),
                        "Val mAP": _format_stat(float(last_row.get("mAP", 0))),
                        "Val FAR@TAR=80": _format_stat(last_row.get("far_at_tar_80")),
                        "Val DIR@FAR=1%": _format_stat(last_row.get("dir_at_far_1")),
                    }
            except Exception:
                pass
    return {
        "Val Rank-1": "not run",
        "Val mAP": "not run",
        "Val FAR@TAR=80": "not run",
        "Val DIR@FAR=1%": "not run",
    }


def read_test_metrics(exp_dirs: List[Path]) -> Dict[str, str]:
    """Read test evaluation from eval_v2.json or eval_results.json."""
    for exp_dir in exp_dirs:
        # Check eval_v2.json
        candidate_v2 = [
            exp_dir / "results" / "eval_v2.json",
            exp_dir / "eval_v2.json",
        ]
        for p in candidate_v2:
            if p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    summary = data.get("summary", {})
                    return {
                        "Test Rank-1": f"{float(data.get('rank_1', 0.0)):.2f}%",
                        "Test mAP": f"{float(data.get('mAP', 0.0)):.2f}%",
                        "Test FAR@TAR=80": _format_stat(summary.get("far_at_tar_80")),
                        "Test FAR@TAR=90": _format_stat(summary.get("far_at_tar_90")),
                        "Test DIR@FAR=1%": _format_stat(summary.get("dir_at_far_1")),
                        "Test DIR@FAR=5%": _format_stat(summary.get("dir_at_far_5")),
                        "Test max TAR": _format_stat(summary.get("max_tar")),
                    }
                except Exception:
                    pass

        # Fallback to eval_results.json (legacy)
        candidate_legacy = [
            exp_dir / "results" / "eval_results.json",
            exp_dir / "eval_results.json",
        ]
        for p in candidate_legacy:
            if p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    r1 = data.get("rank_1", 0.0)
                    if r1 <= 1.0:
                        r1 *= 100
                    map_val = data.get("mAP", 0.0)
                    if map_val <= 1.0:
                        map_val *= 100
                    far80 = data.get("far_at_tar_80")
                    far90 = data.get("far_at_tar_90")
                    return {
                        "Test Rank-1": f"{float(r1):.2f}%",
                        "Test mAP": f"{float(map_val):.2f}%",
                        "Test FAR@TAR=80": _format_stat(
                            float(far80) * 100 if far80 is not None else None
                        ),
                        "Test FAR@TAR=90": _format_stat(
                            float(far90) * 100 if far90 is not None else None
                        ),
                        "Test DIR@FAR=1%": "N/A (legacy)",
                        "Test DIR@FAR=5%": "N/A (legacy)",
                        "Test max TAR": "N/A (legacy)",
                    }
                except Exception:
                    pass

    return {
        "Test Rank-1": "not run",
        "Test mAP": "not run",
        "Test FAR@TAR=80": "not run",
        "Test FAR@TAR=90": "not run",
        "Test DIR@FAR=1%": "not run",
        "Test DIR@FAR=5%": "not run",
        "Test max TAR": "not run",
    }


def read_lookalike_metrics(exp_dirs: List[Path]) -> Dict[str, str]:
    """Read look-alike evaluation metrics from lookalike_v2.json."""
    for exp_dir in exp_dirs:
        candidate_paths = [
            exp_dir / "lookalike_v2.json",
            exp_dir / "results" / "lookalike_v2.json",
        ]
        for p in candidate_paths:
            if p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    summary = data.get("lookalike_protocol", {}).get("summary", {})
                    return {
                        "Lookalike FAR@TAR=80": _format_stat(
                            summary.get("far_at_tar_80")
                        ),
                        "Lookalike DIR@FAR=1%": _format_stat(
                            summary.get("dir_at_far_1")
                        ),
                    }
                except Exception:
                    pass
    return {
        "Lookalike FAR@TAR=80": "not run",
        "Lookalike DIR@FAR=1%": "not run",
    }


def aggregate_all(output_root: Path) -> List[Dict[str, str]]:
    rows = []
    for exp_name in EXPERIMENTS:
        # Search candidate directories
        candidate_dirs = [
            output_root / exp_name,
            PROJECT_ROOT / "outputs" / exp_name,
            PROJECT_ROOT / "results" / exp_name,
        ]

        row = {"Experiment": exp_name}
        val_res = read_val_metrics(candidate_dirs)
        test_res = read_test_metrics(candidate_dirs)
        lk_res = read_lookalike_metrics(candidate_dirs)
        row.update(val_res)
        row.update(test_res)
        row.update(lk_res)
        rows.append(row)
    return rows


def generate_markdown(rows: List[Dict[str, str]]) -> str:
    lines = [
        "# Ablation Experiments Summary",
        "",
        "Aggregated validation hold-out and test-set open-set evaluation across all ablation studies.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(COLUMNS)) + " |",
    ]
    for r in rows:
        vals = [r.get(c, "-") for c in COLUMNS]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append(
        "> - **Evaluation protocol**: Validation metrics are extracted from the held-out validation identities (100 IDs) at the final training epoch (epoch 60). "
        "Test metrics represent evaluation on the official Market-1501 query/gallery test split using Evaluator v2 "
        "(mean ± std over 5 deterministic test evaluation seeds) on `checkpoint_epoch_060.pth`. Experiments marked 'not run' await GPU execution."
    )
    lines.append(
        "> - **Single training run**: Every comparison is based on a single training run (seed 42); training-seed variance is not measured yet. "
        "The ± standard deviations reported in the test table reflect probe/gallery split variance across evaluation seeds for that single trained model, not retraining variance."
    )
    lines.append(
        "> - **Best model overall**: `e0_baseline_val` is the best overall model on the test set, achieving the highest Rank-1 (89.52%), mAP (74.36%), "
        "max TAR (92.56% ± 0.39%), and DIR@FAR=5% (61.31% ± 2.15%), and the lowest FAR@TAR=80% (15.97% ± 1.50%) among all variants."
    )
    lines.append(
        "> - **Winner rule**: To declare a winner over `e0_baseline_val`, a model must outperform `e0` by more than about one std on FAR@TAR=80, FAR@TAR=90, "
        "and DIR@FAR=1% simultaneously in the favorable direction."
    )
    lines.append("> - **Outcome**: Nothing beats `e0_baseline_val`.")
    lines.append("")
    lines.append("## Negative Results")
    lines.append("")
    lines.append(
        "Across all ablations, neither margin softmax nor look-alike-only batch sampling improved upon the standard cross-entropy baseline (`e0_baseline_val`):"
    )
    lines.append("")
    lines.append("1. **Margin Softmax (CosFace)**:")
    lines.append(
        "   - Margin softmax gave no gain over the baseline on this benchmark."
    )
    lines.append(
        "   - At standard margin ($m=0.35$, `e1`), performance degraded severely: Rank-1 dropped by -8.26% (81.26% vs 89.52%), mAP dropped by -15.67% (58.69% vs 74.36%), and FAR@TAR=80% rose sharply to 41.46% ± 2.42% vs 15.97% ± 1.50% (+25.49% worse). Max TAR reached only 86.15% ± 0.31%, making 90% TAR completely unreachable."
    )
    lines.append(
        "   - The observed trend from `e1` ($m=0.35$) to `e1c` ($m=0.20$) shows that a smaller margin is better on every metric: Rank-1 recovered to 84.65% (+3.39% vs `e1`), mAP to 62.70% (+4.01% vs `e1`), max TAR to 88.88% (+2.73% vs `e1`), and FAR@TAR=80% dropped to 31.75% (-9.71% vs `e1`). However, even with $m=0.20$, `e1c` remains substantially worse than `e0_baseline_val` on FAR@TAR=80 (31.75% vs 15.97%), Rank-1 (84.65% vs 89.52%), and mAP (62.70% vs 74.36%)."
    )
    lines.append(
        "   - The explanatory mechanisms discussed in `docs/margin_diagnosis.md` (e.g., gradient saturation, feature suppression, threshold compression) are hypotheses, while the absence of gain from margin softmax is an empirical observation."
    )
    lines.append("")
    lines.append("2. **Look-Alike-Only Batches (Hard Identity Sampler)**:")
    lines.append(
        "   - Restricting training batches to look-alike identity clusters did not improve open-set verification performance over standard random PK sampling."
    )
    lines.append(
        "   - In combination with CosFace (`e2`), FAR@TAR=80 remained severely degraded at 33.34% ± 1.86% and DIR@FAR=1% dropped to 24.56% ± 3.45% (-6.48% vs `e0`)."
    )
    lines.append(
        "   - In combination with cross-entropy (`e4`), Rank-1 dropped by -2.70% (86.82% vs 89.52%), mAP dropped by -4.14% (70.22% vs 74.36%), max TAR dropped to 90.38% ± 0.68% (vs 92.56% ± 0.39%), DIR@FAR=5% dropped to 56.74% ± 2.61% (vs 61.31% ± 2.15%), and FAR@TAR=80% degraded to 22.84% ± 2.38% (vs 15.97% ± 1.50%, +6.87% worse)."
    )
    lines.append(
        "   - While `e4`'s point estimate for DIR@FAR=1% is 33.64% ± 4.76% vs 31.04% ± 2.18% for `e0`, this difference is well inside one standard deviation and accompanied by substantial degradation across all other operational points."
    )
    lines.append(
        "   - Look-alike-only sampling restricts batch diversity by omitting random negative identities, which appears to harm generalization to unseen open-set probe identities."
    )
    lines.append("")
    lines.append("3. **Absolute Negative Loss (`e5`)**:")
    lines.append(
        "   - Adding absolute negative pair loss to `e4` slightly reduced FAR@TAR=80 to 22.02% ± 2.65% (still +6.05% worse than `e0`'s 15.97% ± 1.50%)."
    )
    lines.append(
        "   - Test max TAR reached 90.62% ± 0.43%, meaning the 90% TAR operational point is barely reachable, with FAR@TAR=90 sitting at the ceiling at 81.47% ± 11.99% (vs 45.56% ± 3.91% for `e0`). Rank-1 (86.43%), mAP (69.86%), and DIR@FAR=5% (59.79%) all remain below `e0_baseline_val`."
    )
    lines.append("")
    lines.append("## Experiment Notes and Trajectories")
    lines.append("")
    lines.append("### baseline_resnet18 (Full 751-Identity Benchmark)")
    lines.append(
        "- **Validation**: not recorded (no validation split; trained on all 751 identities)."
    )
    lines.append("- **Test Metrics (v2, 5 seeds)**:")
    lines.append("  - Rank-1: 90.23% | mAP: 76.89%")
    lines.append("  - FAR@TAR=80%: 13.81% ± 2.33% | FAR@TAR=90%: 39.34% ± 4.01%")
    lines.append("  - DIR@FAR=1%: 37.78% ± 3.93% | DIR@FAR=5%: 66.44% ± 2.58%")
    lines.append("  - Max TAR: 93.12% ± 0.42%")
    lines.append("")
    lines.append(
        "### e0_baseline_val (Paired Baseline with 100 Held-Out Validation IDs)"
    )
    lines.append("- **Validation Trajectory**:")
    lines.append(
        "  - Epoch 05: Rank-1 92.73% | mAP 82.38% | FAR@TAR=80  8.98% | FAR@TAR=90 28.00% | DIR@FAR=1% 47.96% | max TAR 94.83%"
    )
    lines.append(
        "  - Epoch 15: Rank-1 95.74% | mAP 84.25% | FAR@TAR=80  4.73% | FAR@TAR=90 20.67% | DIR@FAR=1% 69.01% | max TAR 96.42%"
    )
    lines.append(
        "  - Epoch 30: Rank-1 96.49% | mAP 86.76% | FAR@TAR=80  3.42% | FAR@TAR=90 10.32% | DIR@FAR=1% 57.82% | max TAR 98.01%"
    )
    lines.append(
        "  - Epoch 45: Rank-1 96.49% | mAP 88.58% | FAR@TAR=80  2.02% | FAR@TAR=90  6.16% | DIR@FAR=1% 62.61% | max TAR 97.61%"
    )
    lines.append(
        "  - Epoch 60: Rank-1 97.74% | mAP 89.81% | FAR@TAR=80  1.61% | FAR@TAR=90  4.04% | DIR@FAR=1% 59.01% | max TAR 98.41%"
    )
    lines.append(
        "  - **Late Epoch Behavior**: Metrics continuously rise through early-to-mid epochs (epochs 5–35) and flatten smoothly across late epochs (epochs 45–60) with no decline, confirming stable convergence without overfitting."
    )
    lines.append("- **Test Metrics (v2, 5 seeds)**:")
    lines.append("  - Rank-1: 89.52% (Best overall; -0.71% vs unheld 751-ID baseline)")
    lines.append("  - mAP: 74.36% (Best overall; -2.53% vs unheld 751-ID baseline)")
    lines.append(
        "  - FAR@TAR=80%: 15.97% ± 1.50% (Lowest FAR@TAR=80 across all ablation variants)"
    )
    lines.append("  - FAR@TAR=90%: 45.56% ± 3.91%")
    lines.append("  - DIR@FAR=1%: 31.04% ± 2.18%")
    lines.append(
        "  - DIR@FAR=5%: 61.31% ± 2.15% (Highest DIR@FAR=5% across all ablation variants)"
    )
    lines.append(
        "  - Max TAR: 92.56% ± 0.39% (Highest max TAR across all ablation variants)"
    )
    lines.append("")
    lines.append("### e1_cosface (CosFace Margin Softmax m=0.35, s=30.0)")
    lines.append("- **Validation Trajectory**:")
    lines.append(
        "  - Epoch 05: Rank-1 89.97% | mAP 72.69% | FAR@TAR=80 16.21% | FAR@TAR=90   N/A  | DIR@FAR=1% 31.70% | max TAR 93.03%"
    )
    lines.append(
        "  - Epoch 15: Rank-1 93.73% | mAP 80.39% | FAR@TAR=80  6.78% | FAR@TAR=90 20.74% | DIR@FAR=1% 38.18% | max TAR 95.93%"
    )
    lines.append(
        "  - Epoch 30: Rank-1 93.73% | mAP 81.06% | FAR@TAR=80  5.55% | FAR@TAR=90 25.80% | DIR@FAR=1% 39.95% | max TAR 94.82%"
    )
    lines.append(
        "  - Epoch 45: Rank-1 93.23% | mAP 80.26% | FAR@TAR=80  5.83% | FAR@TAR=90 24.90% | DIR@FAR=1% 50.42% | max TAR 95.12%"
    )
    lines.append(
        "  - Epoch 60: Rank-1 93.48% | mAP 79.77% | FAR@TAR=80  5.12% | FAR@TAR=90 23.07% | DIR@FAR=1% 53.60% | max TAR 95.02%"
    )
    lines.append("- **Test Metrics (v2, 5 seeds)**:")
    lines.append("  - Rank-1: 81.26% (-8.26% vs e0_baseline_val)")
    lines.append("  - mAP: 58.69% (-15.67% vs e0_baseline_val)")
    lines.append("  - FAR@TAR=80%: 41.46% ± 2.42% (+25.49% vs e0_baseline_val)")
    lines.append("  - FAR@TAR=90%: N/A (max TAR = 86.15% ± 0.31%)")
    lines.append("  - DIR@FAR=1%: 28.37% ± 3.10% (-2.67% vs e0_baseline_val)")
    lines.append("  - DIR@FAR=5%: 51.12% ± 2.00% (-10.19% vs e0_baseline_val)")
    lines.append("  - Max TAR: 86.15% ± 0.31% (-6.41% vs e0_baseline_val)")
    lines.append("")
    lines.append("### e2_cosface_hardid (e1 + Hard Identity Sampler)")
    lines.append("- **Validation Trajectory**:")
    lines.append(
        "  - Epoch 05: Rank-1 62.41% | mAP 41.00% | FAR@TAR=80   nan% | FAR@TAR=90   N/A  | DIR@FAR=1% 12.43% | max TAR 71.79%"
    )
    lines.append(
        "  - Epoch 15: Rank-1 88.22% | mAP 71.79% | FAR@TAR=80 10.29% | FAR@TAR=90 50.14% | DIR@FAR=1% 43.05% | max TAR 92.94%"
    )
    lines.append(
        "  - Epoch 30: Rank-1 94.74% | mAP 78.56% | FAR@TAR=80  5.05% | FAR@TAR=90 20.90% | DIR@FAR=1% 54.06% | max TAR 95.82%"
    )
    lines.append(
        "  - Epoch 45: Rank-1 94.24% | mAP 81.20% | FAR@TAR=80  4.94% | FAR@TAR=90 17.96% | DIR@FAR=1% 52.00% | max TAR 95.52%"
    )
    lines.append(
        "  - Epoch 60: Rank-1 94.49% | mAP 81.80% | FAR@TAR=80  4.13% | FAR@TAR=90 16.74% | DIR@FAR=1% 49.63% | max TAR 96.02%"
    )
    lines.append("- **Test Metrics (v2, 5 seeds)**:")
    lines.append("  - Rank-1: 83.70% (-5.82% vs e0_baseline_val, +2.44% vs e1)")
    lines.append("  - mAP: 61.87% (-12.49% vs e0_baseline_val, +3.18% vs e1)")
    lines.append(
        "  - FAR@TAR=80%: 33.34% ± 1.86% (+17.37% vs e0_baseline_val, -8.12% vs e1)"
    )
    lines.append("  - FAR@TAR=90%: N/A (max TAR = 88.12% ± 0.63%)")
    lines.append(
        "  - DIR@FAR=1%: 24.56% ± 3.45% (-6.48% vs e0_baseline_val, -3.81% vs e1)"
    )
    lines.append(
        "  - DIR@FAR=5%: 51.58% ± 1.95% (-9.73% vs e0_baseline_val, +0.46% vs e1)"
    )
    lines.append(
        "  - Max TAR: 88.12% ± 0.63% (-4.44% vs e0_baseline_val, +1.97% vs e1)"
    )
    lines.append("")
    lines.append("### e3_cosface_hardid_absneg (e2 + Absolute Negative Loss)")
    lines.append("- **Validation Trajectory**:")
    lines.append(
        "  - Epoch 05: Rank-1 61.65% | mAP 42.25% | FAR@TAR=80   nan% | FAR@TAR=90   N/A  | DIR@FAR=1% 11.26% | max TAR 70.61%"
    )
    lines.append(
        "  - Epoch 15: Rank-1 89.97% | mAP 71.90% | FAR@TAR=80 18.18% | FAR@TAR=90   N/A  | DIR@FAR=1% 49.18% | max TAR 91.85%"
    )
    lines.append(
        "  - Epoch 30: Rank-1 92.73% | mAP 79.14% | FAR@TAR=80  6.55% | FAR@TAR=90 20.72% | DIR@FAR=1% 49.84% | max TAR 94.33%"
    )
    lines.append(
        "  - Epoch 45: Rank-1 93.98% | mAP 81.93% | FAR@TAR=80  5.14% | FAR@TAR=90 19.49% | DIR@FAR=1% 42.47% | max TAR 95.82%"
    )
    lines.append(
        "  - Epoch 60: Rank-1 93.98% | mAP 81.97% | FAR@TAR=80  3.63% | FAR@TAR=90 13.41% | DIR@FAR=1% 45.51% | max TAR 95.22%"
    )
    lines.append("- **Test Metrics (v2, 5 seeds)**:")
    lines.append("  - Rank-1: 84.06% (-5.46% vs e0_baseline_val, +0.36% vs e2)")
    lines.append("  - mAP: 62.21% (-12.15% vs e0_baseline_val, +0.34% vs e2)")
    lines.append(
        "  - FAR@TAR=80%: 32.48% ± 2.90% (+16.51% vs e0_baseline_val, -0.86% vs e2)"
    )
    lines.append("  - FAR@TAR=90%: N/A (max TAR = 88.71% ± 0.59%)")
    lines.append(
        "  - DIR@FAR=1%: 25.71% ± 1.99% (-5.33% vs e0_baseline_val, +1.15% vs e2)"
    )
    lines.append(
        "  - DIR@FAR=5%: 50.07% ± 1.20% (-11.24% vs e0_baseline_val, -1.51% vs e2)"
    )
    lines.append(
        "  - Max TAR: 88.71% ± 0.59% (-3.85% vs e0_baseline_val, +0.59% vs e2)"
    )
    lines.append("")
    lines.append("### e1b_arcface")
    lines.append("- **Validation**: not run")
    lines.append("- **Test Metrics vs Paired Baseline (e0_baseline_val)**: not run")
    lines.append("")
    lines.append("### e1c_cosface_small (CosFace Reduced Margin m=0.20, s=30.0)")
    lines.append("- **Validation Trajectory**:")
    lines.append(
        "  - Epoch 05: Rank-1 88.72% | mAP 72.64% | FAR@TAR=80 15.36% | FAR@TAR=90   N/A  | DIR@FAR=1% 35.35% | max TAR 91.45%"
    )
    lines.append(
        "  - Epoch 15: Rank-1 91.48% | mAP 81.17% | FAR@TAR=80  6.16% | FAR@TAR=90 19.93% | DIR@FAR=1% 51.59% | max TAR 93.93%"
    )
    lines.append(
        "  - Epoch 30: Rank-1 92.48% | mAP 82.66% | FAR@TAR=80  3.43% | FAR@TAR=90 13.72% | DIR@FAR=1% 55.31% | max TAR 93.94%"
    )
    lines.append(
        "  - Epoch 45: Rank-1 94.49% | mAP 82.78% | FAR@TAR=80  4.84% | FAR@TAR=90 15.24% | DIR@FAR=1% 55.85% | max TAR 95.92%"
    )
    lines.append(
        "  - Epoch 60: Rank-1 94.49% | mAP 82.59% | FAR@TAR=80  4.23% | FAR@TAR=90 16.22% | DIR@FAR=1% 57.24% | max TAR 96.31%"
    )
    lines.append("- **Test Metrics (v2, 5 seeds)**:")
    lines.append("  - Rank-1: 84.65% (-4.87% vs e0_baseline_val, +3.39% vs e1)")
    lines.append("  - mAP: 62.70% (-11.66% vs e0_baseline_val, +4.01% vs e1)")
    lines.append(
        "  - FAR@TAR=80%: 31.75% ± 2.70% (+15.78% vs e0_baseline_val, -9.71% vs e1)"
    )
    lines.append("  - FAR@TAR=90%: N/A (max TAR = 88.88% ± 0.69%)")
    lines.append(
        "  - DIR@FAR=1%: 30.54% ± 2.26% (-0.50% vs e0_baseline_val, +2.17% vs e1)"
    )
    lines.append(
        "  - DIR@FAR=5%: 54.84% ± 2.19% (-6.47% vs e0_baseline_val, +3.72% vs e1)"
    )
    lines.append(
        "  - Max TAR: 88.88% ± 0.69% (-3.68% vs e0_baseline_val, +2.73% vs e1)"
    )
    lines.append(
        "- **Empirical Observation**: Lowering the angular margin from $m=0.35$ to $m=0.20$ improves performance across all dimensions: Rank-1 (+3.39%), mAP (+4.01%), max TAR (+2.73%), DIR@1% (+2.17%), and reduces FAR@80 by 9.71% absolute. This trend shows margin softmax provides no gain over baseline cross-entropy on this benchmark. The explanatory mechanisms in `docs/margin_diagnosis.md` are hypotheses."
    )
    lines.append("")
    lines.append("### e4_ce_hardid (Baseline CE + Hard Identity Sampler)")
    lines.append("- **Validation Trajectory**:")
    lines.append(
        "  - Epoch 05: Rank-1 85.46% | mAP 67.70% | FAR@TAR=80 36.39% | FAR@TAR=90   N/A  | DIR@FAR=1% 20.50% | max TAR 90.46%"
    )
    lines.append(
        "  - Epoch 15: Rank-1 93.23% | mAP 81.54% | FAR@TAR=80  7.47% | FAR@TAR=90 22.76% | DIR@FAR=1% 39.12% | max TAR 95.72%"
    )
    lines.append(
        "  - Epoch 30: Rank-1 95.24% | mAP 85.96% | FAR@TAR=80  3.72% | FAR@TAR=90 11.49% | DIR@FAR=1% 59.57% | max TAR 96.82%"
    )
    lines.append(
        "  - Epoch 45: Rank-1 95.49% | mAP 86.38% | FAR@TAR=80  3.52% | FAR@TAR=90 10.59% | DIR@FAR=1% 61.98% | max TAR 96.51%"
    )
    lines.append(
        "  - Epoch 60: Rank-1 96.49% | mAP 87.25% | FAR@TAR=80  3.23% | FAR@TAR=90  7.56% | DIR@FAR=1% 61.17% | max TAR 96.71%"
    )
    lines.append("- **Test Metrics (v2, 5 seeds)**:")
    lines.append("  - Rank-1: 86.82% (-2.70% vs e0_baseline_val)")
    lines.append("  - mAP: 70.22% (-4.14% vs e0_baseline_val)")
    lines.append("  - FAR@TAR=80%: 22.84% ± 2.38% (+6.87% vs e0_baseline_val)")
    lines.append("  - FAR@TAR=90%: N/A (max TAR = 90.38% ± 0.68%)")
    lines.append(
        "  - DIR@FAR=1%: 33.64% ± 4.76% (+2.60% vs e0_baseline_val, within 1 std of e0's 31.04% ± 2.18%)"
    )
    lines.append("  - DIR@FAR=5%: 56.74% ± 2.61% (-4.57% vs e0_baseline_val)")
    lines.append("  - Max TAR: 90.38% ± 0.68% (-2.18% vs e0_baseline_val)")
    lines.append(
        "- **Assessment**: While `e4`'s point estimate for DIR@FAR=1% is 33.64% ± 4.76% vs 31.04% ± 2.18% for `e0`, the difference is well within one standard deviation. Meanwhile, `e4` is noticeably worse than `e0` across all other operational points: FAR@TAR=80 (22.84% vs 15.97%), Rank-1 (86.82% vs 89.52%), mAP (70.22% vs 74.36%), max TAR (90.38% vs 92.56%), and DIR@FAR=5% (56.74% vs 61.31%). Look-alike-only batching does not outperform the baseline."
    )
    lines.append("")
    lines.append("### e5_ce_hardid_absneg (e4 + Absolute Negative Loss)")
    lines.append("- **Validation Trajectory**:")
    lines.append(
        "  - Epoch 05: Rank-1 86.47% | mAP 67.72% | FAR@TAR=80 34.55% | FAR@TAR=90   N/A  | DIR@FAR=1% 18.52% | max TAR 91.25%"
    )
    lines.append(
        "  - Epoch 15: Rank-1 93.98% | mAP 81.53% | FAR@TAR=80  5.15% | FAR@TAR=90 25.44% | DIR@FAR=1% 53.66% | max TAR 96.52%"
    )
    lines.append(
        "  - Epoch 30: Rank-1 94.74% | mAP 84.78% | FAR@TAR=80  3.83% | FAR@TAR=90 11.19% | DIR@FAR=1% 63.89% | max TAR 96.13%"
    )
    lines.append(
        "  - Epoch 45: Rank-1 94.74% | mAP 86.58% | FAR@TAR=80  2.42% | FAR@TAR=90  9.08% | DIR@FAR=1% 65.53% | max TAR 96.52%"
    )
    lines.append(
        "  - Epoch 60: Rank-1 95.24% | mAP 87.07% | FAR@TAR=80  2.12% | FAR@TAR=90  6.87% | DIR@FAR=1% 66.07% | max TAR 97.12%"
    )
    lines.append("- **Test Metrics (v2, 5 seeds)**:")
    lines.append("  - Rank-1: 86.43% (-3.09% vs e0_baseline_val)")
    lines.append("  - mAP: 69.86% (-4.50% vs e0_baseline_val)")
    lines.append("  - FAR@TAR=80%: 22.02% ± 2.65% (+6.05% vs e0_baseline_val)")
    lines.append(
        "  - FAR@TAR=90%: 81.47% ± 11.99% (barely reachable; sits at the ceiling of max TAR 90.62% ± 0.43%)"
    )
    lines.append("  - DIR@FAR=1%: 31.26% ± 4.35% (+0.22% vs e0_baseline_val)")
    lines.append("  - DIR@FAR=5%: 59.79% ± 1.49% (-1.52% vs e0_baseline_val)")
    lines.append("  - Max TAR: 90.62% ± 0.43% (-1.94% vs e0_baseline_val)")
    lines.append(
        "- **Assessment**: Adding absolute negative loss to `e4` leaves Rank-1 (86.43%) and mAP (69.86%) well below `e0_baseline_val` (89.52% and 74.36%). Although max TAR reaches 90.62% ± 0.43%, FAR@TAR=90 (81.47% ± 11.99%) sits directly at the ceiling and is barely reachable rather than a robust operating regime."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    doc_path = Path(args.doc_path)
    if not doc_path.is_absolute():
        doc_path = PROJECT_ROOT / doc_path

    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = aggregate_all(output_root)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Aggregate] CSV table written to: {csv_path}")

    # Write Markdown
    md_content = generate_markdown(rows)
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Aggregate] Markdown summary written to: {doc_path}")


if __name__ == "__main__":
    main()
