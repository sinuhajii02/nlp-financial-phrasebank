import numpy as np
import pandas as pd

from utils.taxonomy_eval import score_groups, summarise_runs


def test_score_groups_counts_and_empty_group():
    y = np.array([0, 1, 2, 0])
    probs = np.eye(3)[[0, 1, 1, 0]]                 # one mistake, on row 2
    groups = {"overall": np.ones(4, bool), "hard": np.array([0, 0, 1, 0], bool), "domain": np.zeros(4, bool)}
    row = score_groups(y, probs, groups)
    assert (row["n_overall"], row["n_hard"], row["n_domain"]) == (4, 1, 0)
    assert row["f1_hard"] == 0.0
    assert np.isnan(row["f1_domain"])


def test_summarise_runs_mean_std_and_paired_difference():
    runs = pd.DataFrame({
        "arm": ["targeted", "targeted", "random", "random"], "seed": [42, 43, 42, 43],
        "embedding_config": "finbert_frozen", "learning_rate": 1e-5,
        "n_overall": 10, "f1_overall": [0.8, 0.6, 0.7, 0.7],
    })
    table = summarise_runs(runs, groups=["overall"]).set_index("arm")
    assert table.loc["targeted", "f1_mean"] == 0.7
    assert np.isclose(table.loc["targeted", "f1_std"], np.std([0.8, 0.6], ddof=1))
    assert table.loc["targeted", "n"] == 10 and table.loc["targeted", "n_seeds"] == 2
    diff = table.loc["targeted - random"]                 # per seed: +0.1, -0.1
    assert np.isclose(diff.f1_mean, 0.0) and np.isclose(diff.f1_std, np.std([0.1, -0.1], ddof=1))
