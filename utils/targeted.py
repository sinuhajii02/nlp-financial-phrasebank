"""Targeted experiments: #2 (FinBERT embeddings frozen) and #4 (FinBERT embeddings fine-tuned).

Exact counterpart of utils.random_control (#3 / #5): same baseline, same fine-tuning phase, same
scoring and bookkeeping. The only difference is that the up-weighted subset is the hard set of the
training data instead of a random subset of the same size.
"""
from __future__ import annotations

import numpy as np

from utils.random_control import run_weighted_finetune

CONDITION = "targeted"


def run_targeted(baseline_model, data: dict, trainable: bool, seed: int,
                 hard_train_idx: np.ndarray, val_hard_idx: np.ndarray,
                 weight: float | None = None) -> dict:
    """One run of #2 (trainable=False) or #4 (trainable=True) for one seed.

    `baseline_model` must have been trained with the same `trainable` setting.
    """
    return run_weighted_finetune(baseline_model, data, trainable, seed, np.asarray(hard_train_idx),
                                 hard_train_idx, val_hard_idx, weight)
