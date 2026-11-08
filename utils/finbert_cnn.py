# CNN on FinBERT word-piece embeddings, shared by all the CNN notebooks.
# Set keras.utils.set_random_seed(BASELINE_SEED) before training so results can be reproduced.
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertModel, BertTokenizerFast


from utils.preprocessing import LABELS

FINBERT_NAME = "yiyanghkust/finbert-pretrain"
MAX_LEN = 64
BASELINE_SEED = 42
SEEDS = [42, 43, 44]


def load_finbert_inputs(*dfs, model_name=FINBERT_NAME):
    # returns the token ids for every dataframe given, and FinBERT's embedding matrix at the end
    
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    finbert = BertModel.from_pretrained(model_name)
    embedding_matrix = finbert.embeddings.word_embeddings.weight.detach().cpu().numpy()
    del finbert

    outputs = []
    for df in dfs:
        tokens = tokenizer(list(df["sentence"]), padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="np")
        outputs.append(tokens["input_ids"])
    outputs.append(embedding_matrix)
    return outputs


def build_cnn(embedding_matrix, trainable=False):
    inputs = keras.Input(shape=(MAX_LEN,), dtype="int32")
    # we start with zeros and copy FinBERT in with set_weights below. Using a Constant initializer
    # instead makes clone_model run out of memory (it saves the whole matrix in the config)
    embedding = layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], embeddings_initializer="zeros", trainable=trainable, mask_zero=False)
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



def load_hard_selection(percentile, hard_dir="../results/hard_samples"):
    # load the saved P10-P50 hard sets (retraining would give slightly different ones)
    train = pd.read_csv(f"{hard_dir}/train_hard.csv")
    val = pd.read_csv(f"{hard_dir}/val_hard.csv")
    column = f"is_hard_p{percentile}"
    return np.where(train[column])[0], np.where(val[column])[0]
