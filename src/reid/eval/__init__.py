"""Evaluation metrics and protocols for closed-set and open-set person ReID."""

from reid.eval.metrics import compute_closed_set_cmc_and_map, compute_openset_tar_far

__all__ = ["compute_closed_set_cmc_and_map", "compute_openset_tar_far"]
