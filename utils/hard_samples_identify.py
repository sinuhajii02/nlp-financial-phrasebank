from typing import Any
import numpy as np
import pandas as pd



DEFAULT_MARGIN_THRESHOLD = 0.2


def get_hard_sample_indices(
    probs: Any, percentile: float = 20.0
) -> tuple[np.ndarray, float]:
    # Return indices at or below a bottom-percentile margin cutoff.
    # Ties at the threshold are included, so the result can contain slightly more
    # than the requested percentage when multiple samples share a margin.

    probabilities = validate_probs(probs)
    margins = calculate_top_two_margins(probabilities)
    margin_threshold = float(np.round(np.percentile(margins, percentile), 12))
    hard_indices = np.flatnonzero(margins <= margin_threshold)
    return hard_indices, margin_threshold


def get_hard_sample_indices_by_margin(
    probs: Any, margin_threshold: float = DEFAULT_MARGIN_THRESHOLD
) -> np.ndarray:
    """Return indices whose top-two probability margin is below a fixed threshold."""
    probabilities = validate_probs(probs)
    margins = calculate_top_two_margins(probabilities)
    if not 0 <= margin_threshold <= 1:
        raise ValueError("margin_threshold must be between 0 and 1")
    return np.flatnonzero(margins <= margin_threshold)


def add_is_hard(
    df: pd.DataFrame,
    probs: Any,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> pd.DataFrame:
    """Return a copy of ``df`` with baseline margin and ``is_hard`` columns."""
    probabilities = validate_probs(probs)
    if len(df) != len(probabilities):
        raise ValueError("df and probs must contain the same number of samples")
    if not 0 <= margin_threshold <= 1:
        raise ValueError("margin_threshold must be between 0 and 1")

    margins = calculate_top_two_margins(probabilities)
    evaluated = df.copy()
    evaluated["margin"] = margins
    evaluated["pred_class"] = np.argmax(probabilities, axis=1)
    evaluated["is_hard"] = margins <= margin_threshold
    return evaluated


def split_dataset_by_margin(
    df: pd.DataFrame,
    probs: Any,
    percentile: float = 20.0,
    margin_threshold: float | None = None,
):
    """Split a dataframe by a fixed threshold or, for compatibility, a percentile."""

    if margin_threshold is None:
        probabilities = validate_probs(probs)
        if len(df) != len(probabilities):
            raise ValueError("df and probs must contain the same number of samples")
        margins = calculate_top_two_margins(probabilities)
        margin_threshold = float(np.round(np.percentile(margins, percentile), 12))
        evaluated = add_is_hard(df, probabilities, margin_threshold)
    else:
        evaluated = add_is_hard(df, probs, margin_threshold)
    hard_mask = evaluated["is_hard"].to_numpy()

    hard_df = evaluated.loc[hard_mask].reset_index(drop=True)
    easy_df = evaluated.loc[~hard_mask].reset_index(drop=True)
    return hard_df, easy_df, float(margin_threshold)


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
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probs must be between 0 and 1")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-3):
        raise ValueError("probs must sum to 1")
    return probabilities


def calculate_top_two_margins(probs: Any) -> np.ndarray:
    # calculates the vector: M = P_highest - P_second_highest
    top_two = np.sort(np.partition(probs, -2, axis=1)[:, -2:], axis=1)
    return top_two[:, 1] - top_two[:, 0]
