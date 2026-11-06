import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# train with annotator agreement (50 / 66 / 75 / 100 % of annotators gave the gold label)
ARMS = ["agreement_weight", "soft_label"]
LEVELS = [100, 75, 66, 50]


def agreement_weights(agreement):
    # 100% agreement -> weight 1, 50% -> weight 0.5
    return np.array(agreement) / 100


def soft_labels(y, agreement):
    # gold label gets the agreement share, the rest is split evenly over the other two labels
    # (the files don't say which label the other annotators picked)
    share = np.array(agreement) / 100
    targets = np.zeros((len(y), 3))
    for i in range(len(y)):
        targets[i, :] = (1 - share[i]) / 2
        targets[i, y[i]] = share[i]
    return targets


def score_by_agreement(y_true, probs, agreement):
    # macro F1 overall, on all-agree / not-all-agree rows and per agreement level,
    # plus the mean top probability per level (how sure the model is)
    y_true = np.array(y_true)
    probs = np.array(probs)
    pred = probs.argmax(axis=1)
    agreement = np.array(agreement)

    subsets = {"overall": np.ones(len(y_true), dtype=bool),
               "allagree": agreement == 100,
               "not_allagree": agreement < 100}
    for level in LEVELS:
        subsets[f"agree{level}"] = agreement == level

    scores = {}
    for name in subsets:
        mask = subsets[name]
        scores["accuracy_" + name] = accuracy_score(y_true[mask], pred[mask])
        scores["f1_" + name] = f1_score(y_true[mask], pred[mask], average="macro", zero_division=0)
    for level in LEVELS:
        scores[f"confidence_agree{level}"] = probs[agreement == level].max(axis=1).mean()
    return scores


def run_agreement_arm(arm, seed, data, agreement_train):
    # trains one model from scratch (frozen FinBERT, same recipe as notebook 06) and scores it on val
    from tensorflow import keras
    from utils.finbert_cnn import train_baseline

    weights = None
    targets = None
    if arm == "agreement_weight":
        weights = agreement_weights(agreement_train)
    elif arm == "soft_label":
        targets = soft_labels(data["y_train"], agreement_train)
    else:
        raise ValueError("unknown arm: " + arm)

    keras.utils.set_random_seed(seed)
    model = train_baseline(data["embedding_matrix"], data["X_train"], data["y_train"],
                           data["X_val"], data["y_val"], verbose=0,
                           sample_weight=weights, train_targets=targets)
    probs = model.predict(data["X_val"], batch_size=32, verbose=0)

    row = {"arm": arm, "seed": seed}
    row.update(score_by_agreement(data["y_val"], probs, data["agreement_val"]))
    return row, probs
