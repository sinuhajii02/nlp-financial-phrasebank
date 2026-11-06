import numpy as np
import pandas as pd
import pytest

from utils.clean_hard import AMBIGUOUS_WEIGHT, build_row_sets, make_weights, rows_for, save_runs, score_subsets


@pytest.fixture
def band():
    rng = np.random.default_rng(0)
    labels = np.array(["negative", "neutral", "positive"])
    label = labels[rng.integers(0, 3, 500)]
    pred = np.where(rng.random(500) < 0.2, labels[rng.integers(0, 3, 500)], label)   # mostly right, like the CNN
    return pd.DataFrame({"is_hard": rng.random(500) < 0.15, "all_agree": rng.random(500) < 0.5,
                         "label": label, "pred": pred})


def test_row_sets_sizes_and_pools(band):
    sets = build_row_sets(band, seeds=[1, 2])
    clean_hard = rows_for(sets, "clean_hard", 1)
    assert (band.is_hard & band.all_agree).iloc[clean_hard].all()
    for seed in [1, 2]:
        assert len(rows_for(sets, "random", seed)) == len(clean_hard)
        clean_random = rows_for(sets, "clean_random", seed)
        assert len(clean_random) == len(clean_hard)
        assert (~band.is_hard & band.all_agree).iloc[clean_random].all()
    assert not np.array_equal(rows_for(sets, "random", 1), rows_for(sets, "random", 2))
    assert np.array_equal(rows_for(sets, "clean_hard", 1), rows_for(sets, "clean_hard", 2))


def test_weights(band):
    sets = build_row_sets(band, seeds=[1])
    clean_hard = rows_for(sets, "clean_hard", 1)
    ambiguous = rows_for(sets, "ambiguous_hard", 1)

    assert (make_weights(sets, "no_weight", 5, 1, len(band)) == 1).all()

    w = make_weights(sets, "clean_hard", 5, 1, len(band))
    assert (w[clean_hard] == 5).all() and (w == 5).sum() == len(clean_hard) and (w == 1).sum() == len(band) - len(clean_hard)

    w = make_weights(sets, "variant", 5, 1, len(band))
    assert (w[clean_hard] == 5).all() and (w[ambiguous] == AMBIGUOUS_WEIGHT).all()

    with pytest.raises(ValueError):
        make_weights(sets, "nope", 5, 1, len(band))


def test_score_subsets_perfect_and_keys(band):
    y = np.arange(len(band)) % 3
    probs = np.eye(3)[y]
    scores = score_subsets(y, probs, band)
    assert scores["f1_overall"] == 1.0 and scores["accuracy_hard"] == 1.0 and scores["f1_not_allagree"] == 1.0


def test_save_runs_replaces_same_key(tmp_path):
    path = tmp_path / "runs.csv"
    row = {"arm": "random", "k": 3, "embedding_config": "finbert_frozen", "seed": 42, "f1_overall": 0.5}
    save_runs(path, pd.DataFrame([row]))
    save_runs(path, pd.DataFrame([{**row, "f1_overall": 0.7}, {**row, "seed": 43}]))
    out = pd.read_csv(path)
    assert len(out) == 2
    assert out.loc[out.seed == 42, "f1_overall"].item() == 0.7


def test_error_row_sets(band):
    sets = build_row_sets(band, seeds=[1, 2])
    wrong = band.pred != band.label
    clean_error = rows_for(sets, "clean_error", 1)
    assert len(clean_error) == (wrong & band.all_agree).sum()
    assert (wrong & band.all_agree).iloc[clean_error].all()
    for seed in [1, 2]:
        assert len(rows_for(sets, "error_random", seed)) == len(clean_error)
        control = rows_for(sets, "error_clean_random", seed)
        assert len(control) == len(clean_error)
        assert (~wrong & band.all_agree).iloc[control].all()
    w = make_weights(sets, "clean_error", 10, 1, len(band))
    assert (w[clean_error] == 10).all() and (w == 10).sum() == len(clean_error)
