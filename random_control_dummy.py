"""
Random-control experiments  (#3 = FinBERT embeddings FROZEN, #5 = FinBERT embeddings FINE-TUNED)

Idea: take the trained baseline LSTM, give it the SAME amount of extra training
as the targeted run, but on RANDOMLY chosen training sentences instead of the
hard ones. Repeat over several seeds and average.

Everything marked  # MATCH TARGETED  must be identical to what the targeted
experiments (#2 / #4) use, otherwise the comparison is not controlled.

Assumptions (adapt if your baseline differs):
  - the baseline is a saved Keras model whose first layer is the Embedding layer
  - labels are integer ids (see utils/preprocessing.py -> encode_labels)
  - the baseline was compiled with sparse_categorical_crossentropy
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras

# ----------------------------------------------------------------- config
BASELINE_PATH = "../models/baseline_lstm.keras"  # the trained #1 baseline
MARGIN_THRESHOLD = 0.2                           # hard if (top1 - top2) < 0.2
METHOD = "finetune"                              # MATCH TARGETED: finetune | oversample | weighted
EXTRA_EPOCHS = 3                                 # MATCH TARGETED
BATCH_SIZE = 32                                  # MATCH TARGETED
LEARNING_RATE = 1e-4                             # MATCH TARGETED
HARD_WEIGHT = 3.0                                # MATCH TARGETED (only used if METHOD == "weighted")
SEEDS = [0, 1, 2, 3, 4]
ARCHITECTURE = "LSTM"

# #3 -> embeddings frozen, #5 -> embeddings fine-tuned
CONDITIONS = {
    "finbert_frozen": False,      
    "finbert_finetuned": True,
}


# ----------------------------------------------------------------- helpers
def margin(probs):
    """P(highest) - P(second highest) for every row of a softmax output."""
    top2 = np.sort(probs, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def hard_mask(model, X, thr=MARGIN_THRESHOLD):
    return margin(model.predict(X, verbose=0)) < thr


def load_start_model(embedding_trainable):
    """Fresh copy of the baseline, with the embedding layer frozen/unfrozen."""
    model = keras.models.load_model(BASELINE_PATH)
    model.layers[0].trainable = embedding_trainable   # layer 0 = Embedding
    # trainable flags only take effect after (re)compiling
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def extra_training(model, X, y, idx):
    """Apply the extra emphasis to the samples in `idx`, same protocol as targeted."""
    if METHOD == "finetune":            # continued training on the subset only
        model.fit(X[idx], y[idx], epochs=EXTRA_EPOCHS,
                  batch_size=BATCH_SIZE, verbose=0)
    elif METHOD == "oversample":        # subset appended to the full train set
        X2, y2 = np.concatenate([X, X[idx]]), np.concatenate([y, y[idx]])
        model.fit(X2, y2, epochs=EXTRA_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    elif METHOD == "weighted":          # subset gets a higher loss weight
        w = np.ones(len(X))
        w[idx] = HARD_WEIGHT
        model.fit(X, y, sample_weight=w, epochs=EXTRA_EPOCHS,
                  batch_size=BATCH_SIZE, verbose=0)
    else:
        raise ValueError(METHOD)


def scores(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan, np.nan
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0))


def evaluate(model, X_test, y_test, test_hard):
    """Overall / hard / non-hard accuracy and macro-F1.
    `test_hard` is FIXED (defined once by the baseline) so every experiment
    is scored on exactly the same hard / non-hard test sentences."""
    pred = model.predict(X_test, verbose=0).argmax(axis=1)
    out = {}
    for name, m in [("overall", np.ones(len(y_test), bool)),
                    ("hard", test_hard), ("nonhard", ~test_hard)]:
        out[f"accuracy_{name}"], out[f"f1_{name}"] = scores(y_test[m], pred[m])
    return out


# ----------------------------------------------------------------- main
def run(X_train, y_train, X_test, y_test, metrics_path="../results/metrics.csv",
        seeds_path="../results/random_control_seeds.csv"):
    """X_* = padded FinBERT input_ids (np.ndarray), y_* = integer labels (np.ndarray)."""
    baseline = keras.models.load_model(BASELINE_PATH)

    # K = number of hard training samples found by the baseline.
    # The random control must use exactly the same number.
    K = int(hard_mask(baseline, X_train).sum())
    test_hard = hard_mask(baseline, X_test)          # fixed test split: hard vs non-hard
    print(f"K (matched to targeted) = {K}, hard test samples = {test_hard.sum()}")

    per_seed = []
    for cond, trainable in CONDITIONS.items():
        for seed in SEEDS:
            keras.utils.set_random_seed(seed)
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X_train), size=K, replace=False)   # random, train set only

            model = load_start_model(trainable)
            extra_training(model, X_train, y_train, idx)

            res = evaluate(model, X_test, y_test, test_hard)
            res.update(condition="random_control", embedding_config=cond,
                       architecture=ARCHITECTURE, seed=seed)
            per_seed.append(res)
            print(cond, seed, {k: round(v, 3) for k, v in res.items()
                               if isinstance(v, float)})

    df = pd.DataFrame(per_seed)
    df.to_csv(seeds_path, index=False)               # keep every seed for the report

    cols = ["accuracy_overall", "f1_overall", "accuracy_hard", "f1_hard",
            "accuracy_nonhard", "f1_nonhard"]
    mean = (df.groupby(["condition", "embedding_config", "architecture"])[cols]
              .mean().reset_index())
    old = pd.read_csv(metrics_path)
    pd.concat([old, mean], ignore_index=True).to_csv(metrics_path, index=False)
    print(df.groupby("embedding_config")[cols].agg(["mean", "std"]).round(3))
    return df