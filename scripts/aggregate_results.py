"""Aggregate validation and test evaluation results across ablation experiments."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS = [
    "baseline_resnet18",
    "e0_baseline_val",
    "e1_cosface",
    "e2_cosface_hardid",
    "e3_cosface_hardid_absneg",
    "e1b_arcface",
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
    return str(val)


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
                        "Test FAR@TAR=80": _format_stat(float(far80) * 100 if far80 is not None else None),
                        "Test FAR@TAR=90": _format_stat(float(far90) * 100 if far90 is not None else None),
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
        row.update(val_res)
        row.update(test_res)
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
        "> Validation metrics are extracted from the held-out validation identities at the final training epoch. "
        "Test metrics represent evaluation on the official Market-1501 query/gallery test split using Evaluator v2 "
        "(mean ± std over 5 deterministic seeds) evaluated on `checkpoint_epoch_060.pth`. Experiments marked 'not run' "
        "await GPU execution."
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
