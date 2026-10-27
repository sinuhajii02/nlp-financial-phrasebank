from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.hard_samples_identify import get_hard_sample_indices
from utils.random_control import (
    ARCHITECTURE, CONDITION, METRIC_COLUMNS, TEAM_COLUMNS,
    average_wordpiece_embeddings, oof_logreg_probs, sample_random_subset,
    score_predictions, summarize_seeds, to_metrics_rows, update_metrics_csv,
)

ROOT = Path(__file__).resolve().parents[1]
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}


# ---------- random subset ----------
def test_random_subset_basic():
    idx = sample_random_subset(n_train=100, n_extra=20, seed=42)
    assert len(idx) == 20
    assert len(set(idx)) == 20  # no repeats
    assert idx.min() >= 0 and idx.max() < 100


def test_random_subset_reproducible_and_seed_dependent():
    a = sample_random_subset(1000, 200, seed=42)
    b = sample_random_subset(1000, 200, seed=42)
    c = sample_random_subset(1000, 200, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("n_extra", [0, -1, 101])
def test_random_subset_invalid_size(n_extra):
    with pytest.raises(ValueError):
        sample_random_subset(100, n_extra, seed=0)


# ---------- FinBERT -> vocabulary mapping ----------
def test_average_wordpiece_embeddings():
    # word pieces: 0 "profit", 1 "##able", 2 "number", 3 "loss"
    pieces = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [2.0, 2.0]])
    ids = {"profit": [0], "profitable": [0, 1], "number": [2], "loss": [3], "zzz": []}
    word_index = {"<OOV>": 1, "profit": 2, "profitable": 3, "<num>": 4, "loss": 5, "zzz": 6}

    matrix, stats = average_wordpiece_embeddings(word_index, pieces, lambda w: ids[w])

    assert matrix.shape == (7, 2)
    assert np.all(matrix[0] == 0)                      # padding row is zero
    assert np.allclose(matrix[2], [1.0, 0.0])          # single piece
    assert np.allclose(matrix[3], [0.5, 0.5])          # mean of two pieces
    assert np.allclose(matrix[4], [5.0, 5.0])          # <num> -> "number"
    expected_fallback = matrix[[2, 3, 4, 5]].mean(axis=0)
    assert np.allclose(matrix[1], expected_fallback)   # <OOV> -> mean vector
    assert np.allclose(matrix[6], expected_fallback)   # word with no pieces -> mean vector
    assert stats == {"n_words": 6, "n_single_piece": 3, "n_multi_piece": 1, "n_fallback": 2}


# ---------- scoring ----------
def test_score_predictions_hard_nonhard_split():
    y = np.array([0, 1, 2, 1])
    probs = np.array([[0.8, 0.1, 0.1],   # correct
                      [0.1, 0.8, 0.1],   # correct
                      [0.8, 0.1, 0.1],   # wrong (true 2)
                      [0.1, 0.1, 0.8]])  # wrong (true 1)
    scores = score_predictions(y, probs, val_hard_idx=np.array([2, 3]))
    assert scores["accuracy_overall"] == 0.5
    assert scores["accuracy_hard"] == 0.0
    assert scores["accuracy_nonhard"] == 1.0
    assert list(scores) == METRIC_COLUMNS


def test_score_matches_teammate_baseline():
    """Our scoring must reproduce the numbers the baseline notebook stored in metrics.csv."""
    val = pd.read_csv(ROOT / "data" / "val.csv")
    probs_df = pd.read_csv(ROOT / "results" / "probs" / "baseline_logreg_val.csv")
    assert (val["sentence"].values == probs_df["sentence"].values).all()  # same row order

    y = val["label"].map(LABEL_TO_ID).to_numpy()
    probs = probs_df[["p_negative", "p_neutral", "p_positive"]].to_numpy()
    hard_idx, _ = get_hard_sample_indices(probs)
    scores = score_predictions(y, probs, hard_idx)

    metrics = pd.read_csv(ROOT / "results" / "metrics.csv")
    ref = metrics[(metrics["condition"] == "baseline") & (metrics["architecture"] == "logreg")].iloc[0]
    for col in ["accuracy_overall", "f1_overall", "f1_negative", "f1_neutral", "f1_positive"]:
        assert scores[col] == pytest.approx(ref[col])

    assert 0.19 <= len(hard_idx) / len(val) <= 0.25  # bottom ~20% by margin (ties may add a few)


# ---------- out-of-fold hard flags ----------
def test_oof_hard_flags_on_train():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    probs = oof_logreg_probs(train)
    assert probs.shape == (len(train), 3)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-6)
    hard_idx, _ = get_hard_sample_indices(probs)
    assert 0.19 <= len(hard_idx) / len(train) <= 0.25


# ---------- bookkeeping ----------
def _seed_df():
    rows = []
    for emb in ["finbert_frozen", "finbert_finetuned"]:
        for seed, f1 in [(42, 0.6), (43, 0.8)]:
            row = {"seed": seed, "embedding_config": emb, "phase": "after_extra"}
            row.update({c: f1 for c in METRIC_COLUMNS})
            rows.append(row)
            rows.append({**row, "phase": "stage1", **{c: 0.0 for c in METRIC_COLUMNS}})
    return pd.DataFrame(rows)


def test_summary_and_metrics_rows():
    mean, std = summarize_seeds(_seed_df())  # only the "after_extra" phase
    assert mean.loc["finbert_frozen", "f1_overall"] == pytest.approx(0.7)
    assert std.loc["finbert_frozen", "f1_overall"] == pytest.approx(np.std([0.6, 0.8], ddof=1))

    rows = to_metrics_rows(mean)
    assert list(rows.columns) == TEAM_COLUMNS
    assert set(rows["condition"]) == {CONDITION}
    assert set(rows["architecture"]) == {ARCHITECTURE}


def test_update_metrics_csv_keeps_other_rows_and_is_idempotent(tmp_path):
    path = tmp_path / "metrics.csv"
    other = {c: 0.5 for c in METRIC_COLUMNS}
    pd.DataFrame([{"condition": "baseline", "embedding_config": "tfidf_1-2gram",
                   "architecture": "logreg", **other}])[TEAM_COLUMNS].to_csv(path, index=False)

    new_rows = to_metrics_rows(summarize_seeds(_seed_df())[0])
    update_metrics_csv(path, new_rows)
    out = update_metrics_csv(path, new_rows)  # second call must not duplicate

    assert len(out) == 3
    assert list(pd.read_csv(path).columns) == TEAM_COLUMNS
    assert (out["condition"] == "baseline").sum() == 1
