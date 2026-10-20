from typing import Any
import numpy as np
import pandas as pd



def get_hard_sample_indices(probs: Any, percentile: float = 20.0) -> tuple[np.ndarray, float]:
    # Return indices at or below a bottom-percentile margin cutoff.
    # Ties at the threshold are included, so the result can contain slightly more
    # than the requested percentage when multiple samples share a margin.

    probabilities = validate_probs(probs)
    margins = calculate_top_two_margins(probabilities)
    margin_threshold = float(np.percentile(margins, percentile))
    hard_indices = np.flatnonzero(margins <= margin_threshold)
    return hard_indices, margin_threshold


def split_dataset_by_margin(df: pd.DataFrame, probs: Any, percentile: float = 20.0):
    #Return the data split for a DataFrame, giving hard and easy and the margin threshold

    probabilities = validate_probs(probs)
    if len(df) != len(probabilities):
        raise ValueError("df and probs must contain the same number of samples")

    margins = calculate_top_two_margins(probabilities)
    margin_threshold = float(np.percentile(margins, percentile))

    evaluated = df.copy()
    evaluated["margin"] = margins
    evaluated["pred_class"] = np.argmax(probabilities, axis=1)
    hard_mask = margins <= margin_threshold

    hard_df = evaluated.loc[hard_mask].reset_index(drop=True)
    easy_df = evaluated.loc[~hard_mask].reset_index(drop=True)
    return hard_df, easy_df, margin_threshold


#Utils
def validate_probs(probs: Any) -> np.ndarray:
    probabilities = np.asarray(probs, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probs must be a 2D array with shape (samples, classes)")
    if probabilities.shape[1] < 2:
        raise ValueError("probs must contain at least two class probabilities")
    if probabilities.shape[0] == 0:
        raise ValueError("probs must contain at least one sample")
    if not np.isfinite(probabilities).all():
        raise ValueError("probs must contain only finite values")
    return probabilities


def calculate_top_two_margins(probs: Any) -> np.ndarray:
    # calculates the vector: M = P_highest - P_second_highest
    top_two = np.sort(np.partition(probs, -2, axis=1)[:, -2:], axis=1)
    return top_two[:, 1] - top_two[:, 0]
