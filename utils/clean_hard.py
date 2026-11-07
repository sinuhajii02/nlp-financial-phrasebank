import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# arms we compare. hard = unsure (band) or wrong (JTT), each with its own controls
ARMS = ["no_weight", "clean_hard", "random", "clean_random", "variant",
        "clean_error", "error_random", "error_clean_random"]
AMBIGUOUS_WEIGHT = 0.5
RUN_KEY = ["arm", "k", "embedding_config", "seed"]


def build_row_sets(band_train, seeds):
    # which train rows get the extra weight in each arm
    hard = band_train["is_hard"].values
    agree = band_train["all_agree"].values
    wrong = (band_train["pred"] != band_train["label"]).values

    clean_hard = np.where(hard & agree)[0]
    ambiguous_hard = np.where(hard & ~agree)[0]
    clean_easy = np.where(~hard & agree)[0]
    clean_error = np.where(wrong & agree)[0]
    clean_right = np.where(~wrong & agree)[0]

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sets = {}
        sets["clean_hard"] = clean_hard
        sets["ambiguous_hard"] = ambiguous_hard
        sets["random"] = rng.choice(len(band_train), len(clean_hard), replace=False)
        sets["clean_random"] = rng.choice(clean_easy, len(clean_hard), replace=False)
        sets["clean_error"] = clean_error
        sets["error_random"] = rng.choice(len(band_train), len(clean_error), replace=False)
        sets["error_clean_random"] = rng.choice(clean_right, len(clean_error), replace=False)

        for name in sets:
            for i in sorted(sets[name]):
                rows.append({"row_set": name, "seed": seed, "train_idx": i})
    return pd.DataFrame(rows)


def rows_for(row_sets, name, seed):
    picked = row_sets[(row_sets["row_set"] == name) & (row_sets["seed"] == seed)]
    return picked["train_idx"].values


def make_weights(row_sets, arm, k, seed, n_train):
    if arm not in ARMS:
        raise ValueError("unknown arm: " + arm)

    weights = np.ones(n_train)
    if arm == "no_weight":
        return weights
    if arm == "variant":
        weights[rows_for(row_sets, "ambiguous_hard", seed)] = AMBIGUOUS_WEIGHT
        weights[rows_for(row_sets, "clean_hard", seed)] = k
    else:
        weights[rows_for(row_sets, arm, seed)] = k
    return weights


def score_subsets(y_true, probs, band):
    y_true = np.array(y_true)
    pred = np.array(probs).argmax(axis=1)
    hard = band["is_hard"].values
    agree = band["all_agree"].values

    subsets = {
        "overall": np.ones(len(y_true), dtype=bool),
        "hard": hard,
        "nonhard": ~hard,
        "allagree": agree,
        "not_allagree": ~agree,
    }
    scores = {}
    for name in subsets:
        mask = subsets[name]
        scores["accuracy_" + name] = accuracy_score(y_true[mask], pred[mask])
        scores["f1_" + name] = f1_score(y_true[mask], pred[mask], average="macro", zero_division=0)

    per_class = f1_score(y_true, pred, labels=[0, 1, 2], average=None, zero_division=0)
    scores["f1_negative"] = per_class[0]
    scores["f1_neutral"] = per_class[1]
    scores["f1_positive"] = per_class[2]
    return scores


def run_arm(arm, k, seed, data, row_sets, trainable=False, epochs=20):
    # trains one model from scratch with the weights of this arm, scores it on val
    from tensorflow import keras
    from utils.finbert_cnn import train_baseline

    weights = make_weights(row_sets, arm, k, seed, len(data["y_train"]))
    keras.utils.set_random_seed(seed)
    model = train_baseline(data["embedding_matrix"], data["X_train"], data["y_train"],
                           data["X_val"], data["y_val"], trainable=trainable, verbose=0,
                           epochs=epochs, sample_weight=weights)
    probs = model.predict(data["X_val"], batch_size=32, verbose=0)

    if arm == "no_weight":
        k = 1
    if trainable:
        embedding_config = "finbert_finetuned"
    else:
        embedding_config = "finbert_frozen"

    row = {"arm": arm, "k": k, "embedding_config": embedding_config, "seed": seed}
    row.update(score_subsets(data["y_val"], probs, data["band_val"]))
    return row, probs


def save_runs(path, new_rows):
    # if a run is done again, the new result replaces the old one
    if os.path.exists(path):
        old = pd.read_csv(path)
        new_rows = pd.concat([old, new_rows], ignore_index=True)
    new_rows = new_rows.drop_duplicates(subset=RUN_KEY, keep="last")
    new_rows = new_rows.sort_values(RUN_KEY).reset_index(drop=True)
    new_rows.to_csv(path, index=False)
    return new_rows
