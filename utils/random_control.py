"""Random-control experiments: #3 (FinBERT embeddings frozen) and #5 (FinBERT embeddings fine-tuned).

Same configuration and method as the random control of experiment 7 (utils.finbert_cnn):
    1. Baseline CNN on FinBERT word-piece embeddings, trained on the full training set.
    2. Fine-tuning phase: a copy of the baseline is trained further on the full training set with a
       RANDOM subset up-weighted. The subset has exactly as many sentences as the hard subset used by
       the targeted runs, and is drawn with the same seed -> the same indices as experiment 7.
    3. Score on the validation set: overall, on the hard subset and on the non-hard subset.

The only thing that differs from the targeted runs is WHICH sentences get up-weighted.

Heavy imports (keras) are done inside the functions that need them, so the pure-logic functions
(sampling, scoring, metrics bookkeeping) can be imported and tested without them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------
ARCHITECTURE = "cnn_64_finbert"  # Conv1D(64, 3) on FinBERT word pieces, as in experiment 7
CONDITION = "random_control"
HARD_PERCENTILE = 20  # percentile of experiment 7 used for the team metrics row

# Exactly the columns of results/metrics.csv, in the team's order.
TEAM_COLUMNS = [
    "condition", "embedding_config", "architecture",
    "accuracy_overall", "f1_overall",
    "accuracy_hard", "f1_hard",
    "accuracy_nonhard", "f1_nonhard",
    "f1_negative", "f1_neutral", "f1_positive",
]
METRIC_COLUMNS = TEAM_COLUMNS[3:]


def embedding_name(trainable: bool) -> str:
    return "finbert_finetuned" if trainable else "finbert_frozen"


# ---------------------------------------------------------------------------------------------
# Random subset
# ---------------------------------------------------------------------------------------------
def sample_random_subset(n_train: int, n_extra: int, seed: int) -> np.ndarray:
    """Indices of `n_extra` distinct random training sentences (reproducible per seed).

    Same draw as utils.finbert_cnn.random_indices, so a seed picks the same sentences as experiment 7.
    """
    if not 0 < n_extra <= n_train:
        raise ValueError(f"n_extra must be between 1 and n_train={n_train}, got {n_extra}")
    return np.random.default_rng(seed).choice(n_train, size=n_extra, replace=False)


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------
def score_predictions(y_true: np.ndarray, probs: np.ndarray, val_hard_idx: np.ndarray) -> dict:
    """Accuracy and macro F1 overall / hard / non-hard, plus per-class F1 (overall).

    The hard subset is given by `val_hard_idx` (positions in the validation set) and must be the
    same list for every experiment.
    """
    y_true = np.asarray(y_true)
    pred = np.asarray(probs).argmax(axis=1)
    hard = np.zeros(len(y_true), dtype=bool)
    hard[np.asarray(val_hard_idx, dtype=int)] = True

    def acc_f1(mask):
        return (accuracy_score(y_true[mask], pred[mask]),
                f1_score(y_true[mask], pred[mask], average="macro", zero_division=0))

    everyone = np.ones(len(y_true), dtype=bool)
    acc_all, f1_all = acc_f1(everyone)
    acc_hard, f1_hard = acc_f1(hard)
    acc_easy, f1_easy = acc_f1(~hard)
    f1_neg, f1_neu, f1_pos = f1_score(y_true, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "accuracy_overall": acc_all, "f1_overall": f1_all,
        "accuracy_hard": acc_hard, "f1_hard": f1_hard,
        "accuracy_nonhard": acc_easy, "f1_nonhard": f1_easy,
        "f1_negative": f1_neg, "f1_neutral": f1_neu, "f1_positive": f1_pos,
    }


# ---------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------
def get_embedding_weights(model) -> np.ndarray:
    layer = next(l for l in model.layers if l.__class__.__name__ == "Embedding")
    return layer.get_weights()[0]


def run_random_control(baseline_model, data: dict, trainable: bool, seed: int,
                       hard_train_idx: np.ndarray, val_hard_idx: np.ndarray,
                       weight: float | None = None) -> dict:
    """One run of #3 (trainable=False) or #5 (trainable=True) for one seed.

    `baseline_model` must have been trained with the same `trainable` setting.
    `data` holds X_train, y_train, X_val, y_val (FinBERT word-piece ids, label ids).
    """
    from utils.finbert_cnn import DEFAULT_WEIGHT, fine_tune

    weight = DEFAULT_WEIGHT if weight is None else weight
    subset_idx = sample_random_subset(len(data["y_train"]), len(hard_train_idx), seed)
    model = fine_tune(baseline_model, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                      subset_idx, seed, weight)

    embedding_change = float(np.abs(get_embedding_weights(model) - get_embedding_weights(baseline_model)).mean())
    if not trainable and embedding_change != 0.0:
        raise RuntimeError("Embeddings are supposed to be frozen but their weights changed")

    probs = model.predict(data["X_val"], batch_size=32, verbose=0)
    return {
        "seed": seed,
        "embedding_config": embedding_name(trainable),
        "n_extra": len(subset_idx),
        "weight": weight,
        "overlap_with_hard": len(set(subset_idx) & set(hard_train_idx)) / len(subset_idx),
        "embedding_change": embedding_change,
        **score_predictions(data["y_val"], probs, val_hard_idx),
    }


# ---------------------------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------------------------
def summarize_seeds(seed_df: pd.DataFrame):
    """Mean and standard deviation over seeds, per embedding configuration."""
    grouped = seed_df.groupby("embedding_config")[METRIC_COLUMNS]
    return grouped.mean(), grouped.std(ddof=1)


def to_metrics_rows(mean_df: pd.DataFrame) -> pd.DataFrame:
    rows = mean_df.reset_index()
    rows.insert(0, "condition", CONDITION)
    rows.insert(2, "architecture", ARCHITECTURE)
    return rows[TEAM_COLUMNS]


def update_metrics_csv(path: str, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Add/replace rows in results/metrics.csv without touching the other teammates' rows."""
    key = ["condition", "embedding_config", "architecture"]
    path = Path(path)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=TEAM_COLUMNS)

    if existing.empty:
        out = new_rows.copy()
    else:
        merged = existing.merge(new_rows[key].drop_duplicates(), on=key, how="left", indicator=True)
        kept = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
        out = pd.concat([kept, new_rows], ignore_index=True)

    out = out[TEAM_COLUMNS]
    out.to_csv(path, index=False)
    return out
