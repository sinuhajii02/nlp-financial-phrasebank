# CNN on FinBERT word-piece embeddings, shared by all the CNN notebooks.
# Set keras.utils.set_random_seed(BASELINE_SEED) before training so results can be reproduced.
import numpy as np
import pandas as pd
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

# settings for the old fine-tuning phase (continue training a copy of the baseline)
FINE_TUNE_LR = 1e-4
FINE_TUNE_EPOCHS = 10
FINE_TUNE_PATIENCE = 1
DEFAULT_WEIGHT = 1.5


def load_finbert_inputs(*dfs, model_name=FINBERT_NAME):
    # returns the token ids for every dataframe given, and FinBERT's embedding matrix at the end
    from transformers import BertModel, BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    finbert = BertModel.from_pretrained(model_name)
    embedding_matrix = finbert.embeddings.word_embeddings.weight.detach().cpu().numpy()
    del finbert

    outputs = []
    for df in dfs:
        tokens = tokenizer(list(df["sentence"]), padding="max_length", truncation=True,
                           max_length=MAX_LEN, return_tensors="np")
        outputs.append(tokens["input_ids"])
    outputs.append(embedding_matrix)
    return outputs


def build_cnn(embedding_matrix, trainable=False):
    inputs = keras.Input(shape=(MAX_LEN,), dtype="int32")
    # we start with zeros and copy FinBERT in with set_weights below. Using a Constant initializer
    # instead makes clone_model run out of memory (it saves the whole matrix in the config)
    embedding = layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                 embeddings_initializer="zeros", trainable=trainable, mask_zero=False)
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
    # computes macro F1 on val after every epoch, so early stopping can use it
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        predictions = self.model.predict(self.X, batch_size=32, verbose=0).argmax(axis=1)
        logs["val_macro_f1"] = f1_score(self.y, predictions, average="macro")
        print(f" - val_macro_f1: {logs['val_macro_f1']:.4f}")


def train_baseline(embedding_matrix, X_train, y_train, X_val, y_val, trainable=False, verbose="auto",
                   epochs=20, sample_weight=None, train_targets=None):
    # train_targets: optional soft labels (one row of 3 probabilities per sentence) instead of one-hot
    model = build_cnn(embedding_matrix, trainable)
    if train_targets is None:
        train_targets = keras.utils.to_categorical(y_train, len(LABELS))
    early_stop = keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=5,
                                               restore_best_weights=True)
    model.fit(X_train, train_targets, sample_weight=sample_weight,
              validation_data=(X_val, keras.utils.to_categorical(y_val, len(LABELS))),
              epochs=epochs, batch_size=32, verbose=verbose,
              callbacks=[MacroF1Callback(X_val, y_val), early_stop])
    return model


def oof_probabilities(embedding_matrix, X_train, y_train, trainable=False):
    # 5-fold: every train sentence is predicted by a model that did not see it
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(y_train), len(LABELS)))
    for fold_train, fold_held_out in folds.split(X_train, y_train):
        fold_model = build_cnn(embedding_matrix, trainable)
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        fold_model.fit(X_train[fold_train], keras.utils.to_categorical(y_train[fold_train], len(LABELS)),
                       validation_split=0.1, epochs=20, batch_size=32, verbose=0, callbacks=[early_stop])
        oof[fold_held_out] = fold_model.predict(X_train[fold_held_out], batch_size=32, verbose=0)
    return oof


def select_hard(oof_probs, val_probs, percentiles=PERCENTILES):
    # one margin cutoff per percentile, computed on train and used as is on val
    hard_selection = {}
    for percentile in percentiles:
        train_indices, train_cutoff = get_hard_sample_indices(oof_probs, percentile)
        val_indices = get_hard_sample_indices_by_margin(val_probs, train_cutoff)
        hard_selection[percentile] = {"train_indices": train_indices, "val_indices": val_indices,
                                      "train_cutoff": train_cutoff}
    return hard_selection


def load_hard_selection(percentile, hard_dir="../results/hard_samples"):
    # load the saved P10-P50 hard sets (retraining would give slightly different ones)
    train = pd.read_csv(f"{hard_dir}/train_hard.csv")
    val = pd.read_csv(f"{hard_dir}/val_hard.csv")
    column = f"is_hard_p{percentile}"
    return np.where(train[column])[0], np.where(val[column])[0]


def fine_tune(baseline_model, X_train, y_train, X_val, y_val, weighted_indices, seed, weight=DEFAULT_WEIGHT):
    # copy of the baseline, trained a bit more with weighted_indices up-weighted
    keras.utils.set_random_seed(seed)
    model = keras.models.clone_model(baseline_model)
    model.set_weights(baseline_model.get_weights())
    model.compile(optimizer=keras.optimizers.Adam(FINE_TUNE_LR), loss="categorical_crossentropy", metrics=["accuracy"])

    weights = np.ones(len(y_train))
    weights[weighted_indices] = weight
    early_stop = keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=FINE_TUNE_PATIENCE,
                                               restore_best_weights=True)
    model.fit(X_train, keras.utils.to_categorical(y_train, len(LABELS)), sample_weight=weights,
              validation_data=(X_val, keras.utils.to_categorical(y_val, len(LABELS))),
              epochs=FINE_TUNE_EPOCHS, batch_size=32, verbose=0,
              callbacks=[MacroF1Callback(X_val, y_val), early_stop])
    return model


def random_indices(n_train, n, seed):
    return np.random.default_rng(seed).choice(n_train, size=n, replace=False)


def score_model(model, X, y, hard_indices):
    probabilities = model.predict(X, batch_size=32, verbose=0)
    predictions = probabilities.argmax(axis=1)
    scores = {}
    scores["accuracy"] = accuracy_score(y, predictions)
    scores["macro_f1"] = f1_score(y, predictions, average="macro")
    scores["hard_accuracy"] = accuracy_score(y[hard_indices], predictions[hard_indices])
    scores["hard_macro_f1"] = f1_score(y[hard_indices], predictions[hard_indices], average="macro")
    return scores
