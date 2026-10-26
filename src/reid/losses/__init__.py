"""Loss functions for metric learning and open-set representation."""

from reid.losses.combined import BaselineReIDLoss
from reid.losses.triplet import BatchHardTripletLoss

__all__ = ["BatchHardTripletLoss", "BaselineReIDLoss"]
