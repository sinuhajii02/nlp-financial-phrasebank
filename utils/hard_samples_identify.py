import numpy as np
import pandas as pd

DEFAULT_MARGIN_THRESHOLD = 0.2


def validate_probs(probs):
    probabilities = np.array(probs, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probs must be a 2D array with shape (samples, classes)")
    if probabilities.shape[1] < 2:
        raise ValueError("probs must contain at least two class probabilities")
    return probabilities


def calculate_top_two_margins(probs):
    # margin = highest probability - second highest probability
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def get_hard_sample_indices(probs, percentile=20.0):
    # rows with a margin at or below the given percentile of all margins.
    # ties at the cutoff are included, so you can get a few more rows than the percentile
    probabilities = validate_probs(probs)
    margins = calculate_top_two_margins(probabilities)
    margin_threshold = float(np.round(np.percentile(margins, percentile), 12))
    hard_indices = np.where(margins <= margin_threshold)[0]
    return hard_indices, margin_threshold


def get_hard_sample_indices_by_margin(probs, margin_threshold=DEFAULT_MARGIN_THRESHOLD):
    # rows with a margin at or below a fixed threshold
    probabilities = validate_probs(probs)
    if margin_threshold < 0 or margin_threshold > 1:
        raise ValueError("margin_threshold must be between 0 and 1")
    margins = calculate_top_two_margins(probabilities)
    return np.where(margins <= margin_threshold)[0]


def add_is_hard(df, probs, margin_threshold=DEFAULT_MARGIN_THRESHOLD):
    # copy of df with margin, pred_class and is_hard columns
    probabilities = validate_probs(probs)
    if len(df) != len(probabilities):
        raise ValueError("df and probs must contain the same number of samples")
    if margin_threshold < 0 or margin_threshold > 1:
        raise ValueError("margin_threshold must be between 0 and 1")

    margins = calculate_top_two_margins(probabilities)
    evaluated = df.copy()
    evaluated["margin"] = margins
    evaluated["pred_class"] = np.argmax(probabilities, axis=1)
    evaluated["is_hard"] = margins <= margin_threshold
    return evaluated


def split_dataset_by_margin(df, probs, percentile=20.0, margin_threshold=None):
    # split df into hard and easy rows, with a fixed threshold or a percentile
    if margin_threshold is None:
        probabilities = validate_probs(probs)
        if len(df) != len(probabilities):
            raise ValueError("df and probs must contain the same number of samples")
        margins = calculate_top_two_margins(probabilities)
        margin_threshold = float(np.round(np.percentile(margins, percentile), 12))

    evaluated = add_is_hard(df, probs, margin_threshold)
    hard_mask = evaluated["is_hard"].values
    hard_df = evaluated[hard_mask].reset_index(drop=True)
    easy_df = evaluated[~hard_mask].reset_index(drop=True)
    return hard_df, easy_df, float(margin_threshold)
