"""Loss functions for metric learning and open-set representation."""

from reid.losses.combined import BaselineReIDLoss
from reid.losses.margin import MarginSoftmaxHead
from reid.losses.pair import AbsoluteNegativeLoss
from reid.losses.triplet import BatchHardTripletLoss

__all__ = [
    "BatchHardTripletLoss",
    "BaselineReIDLoss",
    "MarginSoftmaxHead",
    "AbsoluteNegativeLoss",
]
