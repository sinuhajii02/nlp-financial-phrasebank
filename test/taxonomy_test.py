import numpy as np
import pandas as pd

from utils.taxonomy import CATEGORIES, load_tagged_splits, make_weights, rule_tag


def _tagged(rows):
    # rows: list of (coded, {category: 1}) -> dataframe shaped like load_tagged_splits output
    df = pd.DataFrame({"coded": [c for c, _ in rows]})
    for cat in CATEGORIES:
        df[f"manual_{cat}"] = pd.array([t.get(cat, 0) if c else pd.NA for c, t in rows], dtype="Int64")
    df["rule_numerical"] = [t.get("rule_numerical", 0) for _, t in rows]
    df["rule_mixed"] = [t.get("rule_mixed", 0) for _, t in rows]
    return df


def test_make_weights():
    df = _tagged([
        (False, {}),                                    # untagged -> 1.0
        (False, {"rule_numerical": 1}),                 # rule tag on uncoded row -> 2.0
        (True, {"company_event": 1}),                   # 1.5
        (True, {"domain": 1, "numerical": 1}),          # largest of up-weights -> 2.0
        (True, {"ambiguous": 1, "mixed": 1}),           # down-weight wins -> 0.5
        (True, {"rule_numerical": 1}),                  # coded and not manually numerical -> 1.0
    ])
    w = make_weights(df, {"ambiguous": 0.5, "numerical": 2.0, "mixed": 2.0, "company_event": 1.5, "domain": 1.5})
    assert w.dtype == np.float32
    np.testing.assert_array_equal(w, [1.0, 2.0, 1.5, 2.0, 0.5, 1.0])
    # categories left out of the dict count as 1.0
    np.testing.assert_array_equal(make_weights(df, {"ambiguous": 0.5}), [1, 1, 1, 1, 0.5, 1])


def test_rule_tag_numerical():
    assert "numerical" in rule_tag("EPS stood at EUR0 .07 versus EUR0 .11 .")
    assert "numerical" in rule_tag("Operating profit fell to EUR 23.26 mn from EUR 32.86 mn .")
    # a year alone is not a figure to compare
    assert "numerical" not in rule_tag("The deal was compared to the one signed in 2008 .")
    assert "numerical" not in rule_tag("The company employs 200 people .")


def test_rule_tag_mixed():
    assert "mixed" in rule_tag("Profit rose but sales declined .")
    assert "mixed" not in rule_tag("Operating loss increased to EUR 17mn .")
    assert rule_tag("The company is based in Helsinki .") == []


def test_load_tagged_splits_merges_manual_tags_on_train_only():
    train, val = load_tagged_splits()
    assert len(train) == 2901 and len(val) == 967
    assert train.coded.sum() == 120          # 2 sheets x 70, 20 shared
    assert not val.coded.any()
    for cat in CATEGORIES:
        assert train.loc[~train.coded, f"manual_{cat}"].isna().all()
        assert train.loc[train.coded, f"manual_{cat}"].notna().all()
