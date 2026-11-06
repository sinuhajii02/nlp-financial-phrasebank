# Old plan (P20 hard set): random control and scoring / metrics.csv helpers.
# A copy of the baseline is trained a bit more with a RANDOM subset up-weighted. The subset is as big
# as the hard set, so the only difference with the targeted runs is which sentences get the weight.
# keras is only imported inside the functions that need it, so the tests run without it.
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ARCHITECTURE = "cnn_64"  # Conv1D(64, 3) on FinBERT word pieces; embedding_config says finbert or random
CONDITION = "random_control"
HARD_PERCENTILE = 20

# columns of results/metrics.csv, in the team's order
TEAM_COLUMNS = [
    "condition", "embedding_config", "architecture",
    "accuracy_overall", "f1_overall",
    "accuracy_hard", "f1_hard",
    "accuracy_nonhard", "f1_nonhard",
    "f1_negative", "f1_neutral", "f1_positive",
]
METRIC_COLUMNS = TEAM_COLUMNS[3:]


def embedding_name(trainable):
    if trainable:
        return "finbert_finetuned"
    return "finbert_frozen"


def sample_random_subset(n_train, n_extra, seed):
    # same draw as finbert_cnn.random_indices, so a seed always picks the same sentences
    if n_extra <= 0 or n_extra > n_train:
        raise ValueError(f"n_extra must be between 1 and {n_train}, got {n_extra}")
    return np.random.default_rng(seed).choice(n_train, size=n_extra, replace=False)


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


def get_embedding_weights(model):
    for layer in model.layers:
        if layer.__class__.__name__ == "Embedding":
            return layer.get_weights()[0]


def run_random_control(baseline_model, data, trainable, seed, hard_train_idx, val_hard_idx, weight=None):
    # one random-control run for one seed. data has X_train, y_train, X_val, y_val
    subset_idx = sample_random_subset(len(data["y_train"]), len(hard_train_idx), seed)
    return run_weighted_finetune(baseline_model, data, trainable, seed, subset_idx,
                                 hard_train_idx, val_hard_idx, weight)


def run_weighted_finetune(baseline_model, data, trainable, seed, subset_idx, hard_train_idx, val_hard_idx,
                          weight=None):
    # used by both the random control and the targeted runs
    from utils.finbert_cnn import DEFAULT_WEIGHT, fine_tune

    if weight is None:
        weight = DEFAULT_WEIGHT
    model = fine_tune(baseline_model, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                      subset_idx, seed, weight)

    # check that frozen embeddings really did not change
    embedding_change = float(np.abs(get_embedding_weights(model) - get_embedding_weights(baseline_model)).mean())
    if not trainable and embedding_change != 0.0:
        raise RuntimeError("Embeddings are supposed to be frozen but their weights changed")

    probs = model.predict(data["X_val"], batch_size=32, verbose=0)
    overlap = len(set(subset_idx) & set(hard_train_idx)) / len(subset_idx)

    row = {"seed": seed, "embedding_config": embedding_name(trainable), "n_extra": len(subset_idx),
           "weight": weight, "overlap_with_hard": overlap, "embedding_change": embedding_change}
    row.update(score_predictions(data["y_val"], probs, val_hard_idx))
    return row


def summarize_seeds(seed_df):
    # mean and std over the seeds, for each embedding config
    grouped = seed_df.groupby("embedding_config")[METRIC_COLUMNS]
    return grouped.mean(), grouped.std(ddof=1)


def to_metrics_rows(mean_df, condition=CONDITION):
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
