from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.hard_samples_identify import get_hard_sample_indices
from utils.random_control import (
    ARCHITECTURE, CONDITION, METRIC_COLUMNS, TEAM_COLUMNS,
    sample_random_subset, score_predictions, summarize_seeds, to_metrics_rows, update_metrics_csv,
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


def test_random_subset_matches_experiment_7():
    # experiment 7 draws np.random.default_rng(seed).choice(len(y_train), size=n, replace=False)
    for seed in [42, 43, 44, 45, 46]:
        expected = np.random.default_rng(seed).choice(2902, size=580, replace=False)
        assert np.array_equal(sample_random_subset(2902, 580, seed), expected)


@pytest.mark.parametrize("n_extra", [0, -1, 101])
def test_random_subset_invalid_size(n_extra):
    with pytest.raises(ValueError):
        sample_random_subset(100, n_extra, seed=0)


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


# ---------- bookkeeping ----------
def _seed_df():
    rows = []
    for emb in ["finbert_frozen", "finbert_finetuned"]:
        for seed, f1 in [(42, 0.6), (43, 0.8)]:
            row = {"seed": seed, "embedding_config": emb}
            row.update({c: f1 for c in METRIC_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_summary_and_metrics_rows():
    mean, std = summarize_seeds(_seed_df())
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


# ---------- shared hard sets ----------
def test_load_hard_selection_matches_saved_exp7_sets():
    from utils.finbert_cnn import load_hard_selection

    train_idx, val_idx = load_hard_selection(20, hard_dir=ROOT / "results" / "hard_samples")
    assert len(train_idx) == 580 and len(val_idx) == 97
    assert np.all(np.diff(train_idx) > 0) and np.all(np.diff(val_idx) > 0)
