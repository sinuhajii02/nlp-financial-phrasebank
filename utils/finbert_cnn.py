"""Shared FinBERT-embedding CNN pipeline (experiment 7 recipe).

Used by notebook/07_cnn_finbert_targeted.ipynb (targeted vs random control) and
notebook/exp3_exp5_random_control.ipynb (#3 frozen / #5 fine-tuned embeddings), so both run the
exact same tokenization, model, hard-sample selection and fine-tuning phase.

Call `keras.utils.set_random_seed(BASELINE_SEED)` once before `train_baseline`, then train the
baseline and the out-of-fold models in that order: the weight initialisations depend on it.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras import layers

from utils.hard_samples_identify import get_hard_sample_indices, get_hard_sample_indices_by_margin
from utils.preprocessing import LABELS

FINBERT_NAME = "yiyanghkust/finbert-pretrain"
MAX_LEN = 64
BASELINE_SEED = 42
PERCENTILES = [10, 20, 30, 40, 50]
SEEDS = [42, 43, 44, 45, 46]

# Fine-tuning phase: continue from the baseline on the full train set, up-weighting a subset.
FINE_TUNE_LR = 1e-4
FINE_TUNE_EPOCHS = 10
FINE_TUNE_PATIENCE = 1
DEFAULT_WEIGHT = 1.5


def load_finbert_inputs(*dfs, model_name: str = FINBERT_NAME):
    """FinBERT word-piece ids for every dataframe, plus FinBERT's word-piece embedding matrix."""
    from transformers import BertModel, BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    finbert = BertModel.from_pretrained(model_name)
    embedding_matrix = finbert.embeddings.word_embeddings.weight.detach().cpu().numpy()
    del finbert

    encoded = [tokenizer(df.sentence.tolist(), padding="max_length", truncation=True,
                         max_length=MAX_LEN, return_tensors="np")["input_ids"] for df in dfs]
    return (*encoded, embedding_matrix)


def build_cnn(embedding_matrix: np.ndarray, trainable: bool = False):
    inputs = keras.Input(shape=(MAX_LEN,), dtype="int32")
    # Weights are copied in with set_weights below, not a Constant initializer: clone_model would
    # serialize the whole FinBERT matrix into the config and run out of memory. "zeros" draws no
    # random numbers, so the other layers get the same initialisation as with Constant.
    embedding = layers.Embedding(
        embedding_matrix.shape[0],
        embedding_matrix.shape[1],
        embeddings_initializer="zeros",
        trainable=trainable, mask_zero=False,
    )
    x = embedding(inputs)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.Conv1D(64, 3, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(LABELS), activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    embedding.set_weights([embedding_matrix])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


class MacroF1Callback(keras.callbacks.Callback):
    def __init__(self, X, y):
        super().__init__(); self.X = X; self.y = y
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        predictions = self.model.predict(self.X, batch_size=32, verbose=0).argmax(axis=1)
        logs["val_macro_f1"] = f1_score(self.y, predictions, average="macro")
        print(f" - val_macro_f1: {logs['val_macro_f1']:.4f}")


def train_baseline(embedding_matrix, X_train, y_train, X_val, y_val, trainable: bool = False, verbose="auto"):
    model = build_cnn(embedding_matrix, trainable)
    model.fit(
        X_train, keras.utils.to_categorical(y_train, len(LABELS)),
        validation_data=(X_val, keras.utils.to_categorical(y_val, len(LABELS))),
        epochs=20, batch_size=32, verbose=verbose,
        callbacks=[MacroF1Callback(X_val, y_val), keras.callbacks.EarlyStopping(
            monitor="val_macro_f1", mode="max", patience=3, restore_best_weights=True
        )],
    )
    return model


def oof_probabilities(embedding_matrix, X_train, y_train, trainable: bool = False):
    """Out-of-fold class probabilities for every training sentence (5-fold, same recipe as baseline)."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(y_train), len(LABELS)))
    for fold_train, fold_held_out in folds.split(X_train, y_train):
        fold_model = build_cnn(embedding_matrix, trainable)
        fold_model.fit(
            X_train[fold_train], keras.utils.to_categorical(y_train[fold_train], len(LABELS)),
            validation_split=0.1, epochs=20, batch_size=32, verbose=0,   # same recipe as baseline: early stopping
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        )
        oof[fold_held_out] = fold_model.predict(X_train[fold_held_out], batch_size=32, verbose=0)
    return oof


def select_hard(oof_probs, val_probs, percentiles=PERCENTILES) -> dict:
    """One cutoff per percentile, computed on train only, applied unchanged to val."""
    hard_selection = {}
    for percentile in percentiles:
        train_indices, train_cutoff = get_hard_sample_indices(oof_probs, percentile)
        val_indices = get_hard_sample_indices_by_margin(val_probs, train_cutoff)
        hard_selection[percentile] = {"train_indices": train_indices, "val_indices": val_indices,
                                      "train_cutoff": train_cutoff}
    return hard_selection


def load_hard_selection(percentile: int, hard_dir: str = "../results/hard_samples"):
    """(train_indices, val_indices) of the hard sets saved by notebook 07 (exp7_{train,val}_hard.csv).

    Re-training the out-of-fold models does not give bit-identical hard sets, so every experiment
    loads the saved ones to work on exactly the same sentences.
    """
    import pandas as pd

    return tuple(np.flatnonzero(pd.read_csv(f"{hard_dir}/exp7_{split}_hard.csv")[f"is_hard_p{percentile}"])
                 for split in ["train", "val"])


def fine_tune(baseline_model, X_train, y_train, X_val, y_val, weighted_indices, seed,
              weight: float = DEFAULT_WEIGHT):
    """Copy of the baseline, trained further on the full train set with `weighted_indices` up-weighted."""
    keras.utils.set_random_seed(seed)
    model = keras.models.clone_model(baseline_model)
    model.set_weights(baseline_model.get_weights())
    model.compile(optimizer=keras.optimizers.Adam(FINE_TUNE_LR), loss="categorical_crossentropy", metrics=["accuracy"])
    weights = np.ones(len(y_train))
    weights[weighted_indices] = weight
    model.fit(
        X_train, keras.utils.to_categorical(y_train, len(LABELS)), sample_weight=weights,
        validation_data=(X_val, keras.utils.to_categorical(y_val, len(LABELS))),
        epochs=FINE_TUNE_EPOCHS, batch_size=32, verbose=0,
        callbacks=[MacroF1Callback(X_val, y_val), keras.callbacks.EarlyStopping(
            monitor="val_macro_f1", mode="max", patience=FINE_TUNE_PATIENCE, restore_best_weights=True)],
    )
    return model


def random_indices(n_train: int, n: int, seed: int) -> np.ndarray:
    """Random-control subset: `n` distinct training indices, reproducible per seed."""
    return np.random.default_rng(seed).choice(n_train, size=n, replace=False)


def score_model(model, X, y, hard_indices) -> dict:
    probabilities = model.predict(X, batch_size=32, verbose=0)
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro"),
        "hard_accuracy": accuracy_score(y[hard_indices], predictions[hard_indices]),
        "hard_macro_f1": f1_score(y[hard_indices], predictions[hard_indices], average="macro"),
    }
