"""Evaluation metrics and protocols for closed-set and open-set person ReID."""

from reid.eval.metrics import (
    compute_closed_set_cmc_and_map,
    compute_metrics_from_scores_v2,
    compute_openset_metrics_v2,
    compute_openset_tar_far,
    plot_roc_curve_v2,
)

__all__ = [
    "compute_closed_set_cmc_and_map",
    "compute_openset_tar_far",
    "compute_metrics_from_scores_v2",
    "compute_openset_metrics_v2",
    "plot_roc_curve_v2",
]
