# Old plan (P20 hard set): targeted runs. Same as the random control in random_control.py,
# except that the up-weighted subset is the hard set instead of random sentences.
import numpy as np

from utils.random_control import run_weighted_finetune

CONDITION = "targeted"


def run_targeted(baseline_model, data, trainable, seed, hard_train_idx, val_hard_idx, weight=None):
    # one targeted run for one seed
    return run_weighted_finetune(baseline_model, data, trainable, seed, np.array(hard_train_idx),
                                 hard_train_idx, val_hard_idx, weight)
