from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Two definitions of "hard", each with its own matched controls:
#   uncertain (band):   clean_hard   vs random       / clean_random
#   misclassified (JTT): clean_error vs error_random / error_clean_random
ARMS = ["no_weight", "clean_hard", "random", "clean_random", "variant",
        "clean_error", "error_random", "error_clean_random"]
AMBIGUOUS_WEIGHT = 0.5
SUBSETS = ["overall", "hard", "nonhard", "allagree", "not_allagree"]
RUN_KEY = ["arm", "k", "embedding_config", "seed"]


def build_row_sets(band_train: pd.DataFrame, seeds) -> pd.DataFrame:
    """One row per (row_set, seed, train_idx). Fixed sets are repeated for every seed so lookups are simple."""
    hard = band_train.is_hard.to_numpy()
    agree = band_train.all_agree.to_numpy()
    clean_hard = np.flatnonzero(hard & agree)
    ambiguous_hard = np.flatnonzero(hard & ~agree)
    clean_easy = np.flatnonzero(~hard & agree)
    wrong = (band_train.pred != band_train.label).to_numpy()   # out-of-fold errors (Just Train Twice)
    clean_error = np.flatnonzero(wrong & agree)
    clean_right = np.flatnonzero(~wrong & agree)

    parts = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sets = {
            "clean_hard": clean_hard,
            "ambiguous_hard": ambiguous_hard,
            "random": rng.choice(len(band_train), len(clean_hard), replace=False),
            "clean_random": rng.choice(clean_easy, len(clean_hard), replace=False),
            "clean_error": clean_error,
            "error_random": rng.choice(len(band_train), len(clean_error), replace=False),
            "error_clean_random": rng.choice(clean_right, len(clean_error), replace=False),
        }
        for name, idx in sets.items():
            parts.append(pd.DataFrame({"row_set": name, "seed": seed, "train_idx": np.sort(idx)}))
    return pd.concat(parts, ignore_index=True)


def rows_for(row_sets: pd.DataFrame, name: str, seed: int) -> np.ndarray:
    mask = (row_sets.row_set == name) & (row_sets.seed == seed)
    return row_sets.train_idx[mask].to_numpy()


def make_weights(row_sets: pd.DataFrame, arm: str, k: float, seed: int, n_train: int) -> np.ndarray:
    weights = np.ones(n_train)
    if arm == "no_weight":
        return weights
    if arm == "variant":
        weights[rows_for(row_sets, "ambiguous_hard", seed)] = AMBIGUOUS_WEIGHT
        weights[rows_for(row_sets, "clean_hard", seed)] = k
        return weights
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    weights[rows_for(row_sets, arm, seed)] = k
    return weights


def score_subsets(y_true, probs, band: pd.DataFrame) -> dict:
    """Accuracy and macro F1 overall, on hard / non-hard and on all-agree / not-all-agree rows."""
    y_true = np.asarray(y_true)
    pred = np.asarray(probs).argmax(axis=1)
    hard = band.is_hard.to_numpy()
    agree = band.all_agree.to_numpy()
    masks = {"overall": np.ones(len(y_true), bool), "hard": hard, "nonhard": ~hard,
             "allagree": agree, "not_allagree": ~agree}

    scores = {}
    for name, mask in masks.items():
        scores[f"accuracy_{name}"] = accuracy_score(y_true[mask], pred[mask])
        scores[f"f1_{name}"] = f1_score(y_true[mask], pred[mask], average="macro", zero_division=0)
    f1_neg, f1_neu, f1_pos = f1_score(y_true, pred, labels=[0, 1, 2], average=None, zero_division=0)
    scores.update(f1_negative=f1_neg, f1_neutral=f1_neu, f1_positive=f1_pos)
    return scores


def run_arm(arm: str, k: float, seed: int, data: dict, row_sets: pd.DataFrame,
            trainable: bool = False, epochs: int = 20):
 
    from tensorflow import keras
    from utils.finbert_cnn import train_baseline

    weights = make_weights(row_sets, arm, k, seed, len(data["y_train"]))
    keras.utils.set_random_seed(seed)
    model = train_baseline(data["embedding_matrix"], data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                           trainable=trainable, verbose=0, epochs=epochs, sample_weight=weights)
    probs = model.predict(data["X_val"], batch_size=32, verbose=0)
    row = {"arm": arm, "k": 1 if arm == "no_weight" else k,
           "embedding_config": "finbert_finetuned" if trainable else "finbert_frozen", "seed": seed,
           **score_subsets(data["y_val"], probs, data["band_val"])}
    return row, probs


def save_runs(path: str, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Add runs to the shared per-seed file, replacing any earlier run with the same arm/k/embedding/seed."""
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path)
        old = old.merge(new_rows[RUN_KEY], on=RUN_KEY, how="left", indicator=True)
        old = old[old._merge == "left_only"].drop(columns="_merge")
        new_rows = pd.concat([old, new_rows], ignore_index=True)
    new_rows = new_rows.sort_values(RUN_KEY).reset_index(drop=True)
    new_rows.to_csv(path, index=False)
    return new_rows
