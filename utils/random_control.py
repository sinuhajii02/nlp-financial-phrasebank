"""Random-control experiments: #3 (FinBERT embeddings frozen) and #5 (FinBERT embeddings fine-tuned).

Pipeline per (embedding mode, seed):
    1. Train the LSTM on the full training set (stage 1, same recipe as the baseline LSTM).
    2. Extra fine-tuning phase on a RANDOM subset of the training set. The subset has exactly as
       many sentences as the hard subset used by the targeted runs (#2 / #4).
    3. Score on the validation set: overall, on the hard subset and on the non-hard subset.

The only thing that differs from the targeted runs is WHICH sentences get the extra phase. Everything
else (embeddings, architecture, learning rate, number of extra epochs) must be identical.

Heavy imports (keras, transformers) are done inside the functions that need them, so the
pure-logic functions (sampling, scoring, metrics bookkeeping) can be imported and tested without them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from utils.hard_samples_identify import get_hard_sample_indices

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------
NUM_CLASSES = 3
OOV_TOKEN = "<OOV>"  # must match utils.preprocessing.fit_tokenizer
SPECIAL_WORDS = {"<num>": "number"}  # placeholder -> a real word FinBERT knows
ARCHITECTURE = "lstm_64_dropout"  # same name as the baseline LSTM row in metrics.csv
CONDITION = "random_control"
HARD_PERCENTILE = 20.0  # default of utils.hard_samples_identify

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
# FinBERT embedding matrix on the shared (word-level) vocabulary
# ---------------------------------------------------------------------------------------------
def average_wordpiece_embeddings(word_index: dict, wordpiece_matrix: np.ndarray, tokenize):
    """Build a (vocab_size, dim) matrix for the Keras vocabulary.

    FinBERT works on word pieces, our tokenizer works on whole words. For every word we split it
    into FinBERT word pieces and average their embedding vectors.
      - row 0 (padding) stays all zeros;
      - "<num>" uses the vector of the word "number";
      - "<OOV>" and words FinBERT cannot split get the mean vector of all other words.
    `tokenize(word) -> list[int]` returns the word-piece ids of a word.
    """
    vocab_size = len(word_index) + 1
    dim = wordpiece_matrix.shape[1]
    matrix = np.zeros((vocab_size, dim), dtype=np.float32)
    has_vector = np.zeros(vocab_size, dtype=bool)
    n_single = n_multi = 0

    for word, idx in word_index.items():
        if word == OOV_TOKEN:
            continue
        ids = list(tokenize(SPECIAL_WORDS.get(word, word)))
        if not ids:
            continue
        matrix[idx] = wordpiece_matrix[ids].mean(axis=0)
        has_vector[idx] = True
        if len(ids) == 1:
            n_single += 1
        else:
            n_multi += 1

    fallback = matrix[has_vector].mean(axis=0)
    missing = ~has_vector
    missing[0] = False  # padding stays zero
    matrix[missing] = fallback

    stats = {
        "n_words": vocab_size - 1,
        "n_single_piece": n_single,
        "n_multi_piece": n_multi,
        "n_fallback": int(missing.sum()),
    }
    return matrix, stats


def build_finbert_embedding_matrix(word_index: dict, model_name: str = "ProsusAI/finbert",
                                   cache_path: str | None = None):
    """Download FinBERT once and map its word-piece embeddings onto our vocabulary.

    Needs `pip install transformers torch` and internet access. The result is cached in
    `cache_path` (.npz) together with the word list, so it is only rebuilt when the vocabulary changes.
    """
    words = [w for w, _ in sorted(word_index.items(), key=lambda kv: kv[1])]

    if cache_path and Path(cache_path).exists():
        cached = np.load(cache_path, allow_pickle=False)
        if list(cached["words"]) == words:
            return cached["matrix"], None

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    wordpiece_matrix = model.get_input_embeddings().weight.detach().cpu().numpy()
    matrix, stats = average_wordpiece_embeddings(
        word_index, wordpiece_matrix,
        lambda w: tokenizer(w, add_special_tokens=False)["input_ids"],
    )

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, matrix=matrix, words=np.array(words))
    return matrix, stats


def make_dummy_embedding_matrix(vocab_size: int, dim: int = 768, seed: int = 0) -> np.ndarray:
    """Random stand-in for FinBERT. ONLY for testing the pipeline, never for real results."""
    matrix = np.random.default_rng(seed).normal(0.0, 0.1, size=(vocab_size, dim)).astype(np.float32)
    matrix[0] = 0.0
    return matrix


# ---------------------------------------------------------------------------------------------
# Hard samples and random subset
# ---------------------------------------------------------------------------------------------
def oof_logreg_probs(train_df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Out-of-fold class probabilities for every training sentence (TF-IDF + LogReg baseline).

    Each sentence is predicted by a model that never saw it (5-fold cross_val_predict), as required
    by the team rule for hard-sample flags. Same model settings as the main baseline.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    from utils.preprocessing import clean_text, encode_labels

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, tokenizer=str.split, token_pattern=None,
                                  ngram_range=(1, 2), sublinear_tf=True)),
        ("lr", LogisticRegression(C=10, class_weight="balanced", max_iter=2000)),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, train_df["sentence"], encode_labels(train_df["label"]),
                             cv=cv, method="predict_proba")


def sample_random_subset(n_train: int, n_extra: int, seed: int) -> np.ndarray:
    """Indices of `n_extra` distinct random training sentences (sorted, reproducible per seed)."""
    if not 0 < n_extra <= n_train:
        raise ValueError(f"n_extra must be between 1 and n_train={n_train}, got {n_extra}")
    return np.sort(np.random.default_rng(seed).choice(n_train, size=n_extra, replace=False))


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
# Model (Keras) - same recipe as the baseline LSTM in notebook/01_baseline.ipynb
# ---------------------------------------------------------------------------------------------
def prepare_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Shared preprocessing: tokenizer fitted on train only, padded sequences, label ids."""
    from utils.preprocessing import encode_labels, encode_texts, fit_tokenizer

    tok = fit_tokenizer(train_df["sentence"])
    return {
        "tok": tok,
        "X_train": encode_texts(tok, train_df["sentence"]),
        "y_train": encode_labels(train_df["label"]),
        "X_val": encode_texts(tok, val_df["sentence"]),
        "y_val": encode_labels(val_df["label"]),
        "vocab_size": len(tok.word_index) + 1,
    }


def build_lstm(vocab_size: int, embedding_matrix: np.ndarray, trainable: bool,
               lstm_units: int = 64, dropout: float = 0.3, learning_rate: float = 1e-3):
    from keras.layers import LSTM, Dense, Dropout, Embedding, Input
    from keras.models import Sequential
    from keras.optimizers import Adam

    from utils.preprocessing import MAX_LEN

    if embedding_matrix.shape[0] != vocab_size:
        raise ValueError(f"embedding matrix has {embedding_matrix.shape[0]} rows, vocabulary has {vocab_size}")

    embedding = Embedding(vocab_size, embedding_matrix.shape[1], mask_zero=True, trainable=trainable)
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        embedding,
        LSTM(lstm_units),
        Dropout(dropout),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    embedding.set_weights([embedding_matrix])
    model.compile(optimizer=Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def get_embedding_weights(model) -> np.ndarray:
    layer = next(l for l in model.layers if l.__class__.__name__ == "Embedding")
    return layer.get_weights()[0]


def train_stage1(model, data: dict, epochs: int = 30, batch_size: int = 32, patience: int = 3):
    """Full-training-set phase. Same settings as the baseline LSTM (early stopping on val loss)."""
    from keras.callbacks import EarlyStopping
    from keras.utils import to_categorical

    model.fit(
        data["X_train"], to_categorical(data["y_train"], NUM_CLASSES),
        validation_data=(data["X_val"], to_categorical(data["y_val"], NUM_CLASSES)),
        epochs=epochs, batch_size=batch_size, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)],
    )
    return model


def extra_finetune(model, data: dict, subset_idx: np.ndarray, epochs: int, learning_rate: float,
                   batch_size: int = 32):
    """Extra fine-tuning phase on a subset of the training set (fixed number of epochs, no val)."""
    from keras.optimizers import Adam
    from keras.utils import to_categorical

    model.compile(optimizer=Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        data["X_train"][subset_idx], to_categorical(data["y_train"][subset_idx], NUM_CLASSES),
        epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0,
    )
    return model


def run_random_control(data: dict, embedding_matrix: np.ndarray, trainable: bool, seed: int,
                       hard_train_idx: np.ndarray, val_hard_idx: np.ndarray,
                       extra_epochs: int = 3, extra_lr: float = 1e-4) -> list[dict]:
    """One run of #3 (trainable=False) or #5 (trainable=True) for one seed.

    Returns two result dicts: phase "stage1" (before the extra phase) and "after_extra".
    `extra_epochs` and `extra_lr` MUST equal the values used in the targeted runs.
    """
    import keras

    keras.utils.set_random_seed(seed)
    model = build_lstm(data["vocab_size"], embedding_matrix, trainable)
    emb_before = get_embedding_weights(model).copy()

    train_stage1(model, data)
    stage1_probs = model.predict(data["X_val"], verbose=0)

    subset_idx = sample_random_subset(len(data["y_train"]), len(hard_train_idx), seed)
    extra_finetune(model, data, subset_idx, extra_epochs, extra_lr)
    final_probs = model.predict(data["X_val"], verbose=0)

    embedding_change = float(np.abs(get_embedding_weights(model) - emb_before).mean())
    if not trainable and embedding_change != 0.0:
        raise RuntimeError("Embeddings are supposed to be frozen but their weights changed")

    info = {
        "seed": seed,
        "embedding_config": embedding_name(trainable),
        "n_extra": len(subset_idx),
        "extra_epochs": extra_epochs,
        "extra_lr": extra_lr,
        "overlap_with_hard": len(set(subset_idx) & set(hard_train_idx)) / len(subset_idx),
        "embedding_change": embedding_change,
    }
    return [
        {**info, "phase": "stage1", **score_predictions(data["y_val"], stage1_probs, val_hard_idx)},
        {**info, "phase": "after_extra", **score_predictions(data["y_val"], final_probs, val_hard_idx)},
    ]


# ---------------------------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------------------------
def summarize_seeds(seed_df: pd.DataFrame, phase: str = "after_extra"):
    """Mean and standard deviation over seeds, per embedding configuration."""
    grouped = seed_df[seed_df["phase"] == phase].groupby("embedding_config")[METRIC_COLUMNS]
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
