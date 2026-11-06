# scoring and the shared results/metrics.csv
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ARCHITECTURE = "cnn_64"  # Conv1D(64, 3) on FinBERT word pieces; embedding_config says finbert or random

# columns of results/metrics.csv, in the team's order
TEAM_COLUMNS = [
    "condition", "embedding_config", "architecture",
    "accuracy_overall", "f1_overall",
    "accuracy_hard", "f1_hard",
    "accuracy_nonhard", "f1_nonhard",
    "f1_negative", "f1_neutral", "f1_positive",
]
METRIC_COLUMNS = TEAM_COLUMNS[3:]


def score_predictions(y_true, probs, val_hard_idx):
    # accuracy and macro F1 overall / hard / non-hard, plus F1 per class
    y_true = np.array(y_true)
    pred = np.array(probs).argmax(axis=1)
    hard = np.zeros(len(y_true), dtype=bool)
    hard[np.array(val_hard_idx, dtype=int)] = True
    everyone = np.ones(len(y_true), dtype=bool)

    scores = {}
    for name, mask in [("overall", everyone), ("hard", hard), ("nonhard", ~hard)]:
        scores["accuracy_" + name] = accuracy_score(y_true[mask], pred[mask])
        scores["f1_" + name] = f1_score(y_true[mask], pred[mask], average="macro", zero_division=0)

    per_class = f1_score(y_true, pred, labels=[0, 1, 2], average=None, zero_division=0)
    scores["f1_negative"] = per_class[0]
    scores["f1_neutral"] = per_class[1]
    scores["f1_positive"] = per_class[2]
    return scores


def to_metrics_rows(mean_df, condition):
    # mean_df: one row per embedding_config with the metric columns
    rows = mean_df.reset_index()
    rows["condition"] = condition
    rows["architecture"] = ARCHITECTURE
    return rows[TEAM_COLUMNS]


def update_metrics_csv(path, new_rows):
    # adds rows to metrics.csv; a row with the same condition/embedding/architecture is replaced
    key = ["condition", "embedding_config", "architecture"]
    if os.path.exists(path):
        old = pd.read_csv(path)
        out = pd.concat([old, new_rows], ignore_index=True)
    else:
        out = new_rows.copy()
    out = out.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    out = out[TEAM_COLUMNS]
    out.to_csv(path, index=False)
    return out
