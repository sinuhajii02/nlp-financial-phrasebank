import numpy as np
import pandas as pd
import pytest
from utils.hard_samples_identify import (
    add_is_hard,
    get_hard_sample_indices,
    get_hard_sample_indices_by_margin,
    split_dataset_by_margin,
)

# To run use pytest hard_sample_test.py on /test

# UNit Testing
def test_basic():
    probs = np.array([
        [0.90, 0.05, 0.05],
        [0.50, 0.30, 0.20],
        [0.40, 0.40, 0.20],
        [0.60, 0.30, 0.10]
    ])
    
    indices, thresh = get_hard_sample_indices(probs, percentile=25.0)
    print(thresh)
    
    assert len(indices) == 1
    assert indices[0] == 2
    assert thresh == 0.15

def test_get_hard_sample_indices_ties():
    probs = np.array([
        [0.4, 0.4, 0.2],  # margin 0.0
        [0.4, 0.4, 0.2],  # margin 0.0
        [0.9, 0.1, 0.0]   # margin 0.8
    ])

    indices, thresh = get_hard_sample_indices(probs, percentile=10.0)
    assert len(indices) == 2
    assert set(indices) == {0, 1}


# Test on datasets

@pytest.fixture
def train_dataset():
    df = pd.read_csv("../data/train.csv")
    
    if 'id' not in df.columns:
        df['id'] = range(len(df))
        
    num_samples = len(df)
    num_classes = 3
    
    # Scaled mock probabilities 
    raw_probs = np.random.uniform(0.1, 0.9, (num_samples, num_classes))
    probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)
    
    return df, probs


def test_split_dataset_invariants(train_dataset):
    df, probs = train_dataset
    percentile = 20.0
    
    hard_df, easy_df, thresh = split_dataset_by_margin(df, probs, percentile)
    
    assert len(hard_df) + len(easy_df) == len(df)
    
    hard_ids = set(hard_df['id'])
    easy_ids = set(easy_df['id'])
    assert len(hard_ids.intersection(easy_ids)) == 0
    
    if len(hard_df) > 0:
        assert hard_df['margin'].max() <= thresh
    if len(easy_df) > 0:
        assert easy_df['margin'].min() > thresh


def test_split_dataset_mismatch(train_dataset):
    df, probs = train_dataset
    probs_short = probs[:-1]
    
    with pytest.raises(ValueError, match="same number of samples"):
        split_dataset_by_margin(df, probs_short)


def test_add_is_hard_uses_fixed_margin_threshold():
    df = pd.DataFrame({"sentence": ["uncertain", "confident"]})
    probs = np.array([
        [0.45, 0.45, 0.10],
        [0.90, 0.05, 0.05],
    ])

    evaluated = add_is_hard(df, probs, margin_threshold=0.2)

    assert evaluated["is_hard"].tolist() == [True, False]
    assert evaluated["margin"].tolist() == [0.0, 0.85]
    np.testing.assert_array_equal(
        get_hard_sample_indices_by_margin(probs, margin_threshold=0.2),
        np.array([0]),
    )