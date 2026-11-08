from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.hard_samples_identify import get_hard_sample_indices
from utils.metrics import ARCHITECTURE, METRIC_COLUMNS, TEAM_COLUMNS, score_predictions, to_metrics_rows, update_metrics_csv

ROOT = Path(__file__).resolve().parents[1]
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}


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
    # our scoring must give the same numbers the baseline notebook stored in metrics.csv
    val = pd.read_csv(ROOT / "data" / "val.csv")
    probs_df = pd.read_csv(ROOT / "results" / "predictions" / "baseline_logreg.csv")
    assert (val["sentence"].values == probs_df["sentence"].values).all()  # same row order

    y = val["label"].map(LABEL_TO_ID).values
    probs = probs_df[["p_negative", "p_neutral", "p_positive"]].values
    hard_idx, _ = get_hard_sample_indices(probs)
    scores = score_predictions(y, probs, hard_idx)

    metrics = pd.read_csv(ROOT / "results" / "metrics.csv")
    ref = metrics[(metrics["condition"] == "baseline") & (metrics["architecture"] == "logreg")].iloc[0]
    for col in ["accuracy_overall", "f1_overall", "f1_negative", "f1_neutral", "f1_positive"]:
        assert scores[col] == pytest.approx(ref[col])


# ---------- metrics.csv ----------
def mean_rows():
    df = pd.DataFrame({"embedding_config": ["finbert_frozen", "finbert_finetuned"]})
    for c in METRIC_COLUMNS:
        df[c] = 0.7
    return df.set_index("embedding_config")


def test_to_metrics_rows():
    rows = to_metrics_rows(mean_rows(), "targeted")
    assert list(rows.columns) == TEAM_COLUMNS
    assert set(rows["condition"]) == {"targeted"}
    assert set(rows["architecture"]) == {ARCHITECTURE}


def test_update_metrics_csv_keeps_other_rows_and_does_not_duplicate(tmp_path):
    path = tmp_path / "metrics.csv"
    other = {c: 0.5 for c in METRIC_COLUMNS}
    pd.DataFrame([{"condition": "baseline", "embedding_config": "tfidf_1-2gram",
                   "architecture": "logreg", **other}])[TEAM_COLUMNS].to_csv(path, index=False)

    new_rows = to_metrics_rows(mean_rows(), "targeted")
    update_metrics_csv(path, new_rows)
    out = update_metrics_csv(path, new_rows)  # second call must not add the rows again

    assert len(out) == 3
    assert list(pd.read_csv(path).columns) == TEAM_COLUMNS
    assert (out["condition"] == "baseline").sum() == 1


# ---------- saved hard sets ----------
def test_load_hard_selection_matches_saved_sets():
    from utils.finbert_cnn import load_hard_selection

    train_idx, val_idx = load_hard_selection(20, hard_dir=ROOT / "results" / "hard_samples")
    assert len(train_idx) == 580 and len(val_idx) == 97
    assert np.all(np.diff(train_idx) > 0) and np.all(np.diff(val_idx) > 0)
