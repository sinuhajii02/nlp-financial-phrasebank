import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_TAG_PATHS = [ROOT / "results/taxonomy/coding_sheet_B.csv",
                     ROOT / "results/taxonomy/coding_sheet_D.csv"]

CATEGORIES = ["ambiguous", "domain", "mixed", "company_event", "numerical"]
RULE_CATEGORIES = ["numerical", "mixed"]
# coding-sheet column -> taxonomy category
SHEET_COLUMNS = {"ambiguous_label": "ambiguous", "domain_required": "domain", "mixed_sentiment": "mixed",
                 "corporate_action": "company_event", "numerical_comparison": "numerical"}


YEAR = re.compile(r"^(19|20)\d\d$")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
COMPARISON_WORDS = re.compile(
    r"\bcompared (?:to|with)\b|\bagainst\b|\bversus\b|\bvs\b\.?|\bthan\b"
    r"|\byear-ago\b|\byear-on-year\b|\by-o-y\b|\ba year (?:earlier|ago|before)\b"
    r"|\b(?:previous|prior|last|preceding) (?:year|quarter|period|financial year)\b"
    r"|\b(?:corresponding|same|comparable) (?:period|quarter)\b"
    r"|%\s+(?:over|on|from)\b",
    re.IGNORECASE)

# mixed: a positive and a negative word in the same sentence.
POSITIVE_WORDS = re.compile(
    r"\b(?:increas(?:e|ed|es|ing)|ris(?:e|es|ing|en)|rose|gr[eo]w(?:s|n|th|ing)?|improv(?:e|ed|es|ing|ement)"
    r"|higher|gain(?:s|ed)?|strong(?:er|est)?|boost(?:s|ed|ing)?|record|surg(?:e|ed)|jump(?:s|ed)?"
    r"|climb(?:s|ed)?|exceed(?:s|ed)?|lift(?:s|ed)?|recover(?:s|ed|y)?|up\s+(?:by\s+)?\d)\b",
    re.IGNORECASE)
NEGATIVE_WORDS = re.compile(
    r"\b(?:decreas(?:e|ed|es|ing)|declin(?:e|ed|es|ing)|f[ae]ll(?:s|en|ing)?|drop(?:s|ped)?|lower|weak(?:er|ened)?"
    r"|slid(?:e|es)?|cuts?|worse(?:ned)?|plung(?:e|ed)|tough|negative|below|down\s+(?:by\s+)?\d)\b",
    re.IGNORECASE)


def has_figure(sentence):
    return any(not YEAR.match(n) for n in NUMBER.findall(sentence))


def has_from_figure(sentence):
    tokens = sentence.split()
    for i, tok in enumerate(tokens):
        if tok.lower() == "from":
            for nxt in tokens[i + 1:i + 4]:
                if any(not YEAR.match(n) for n in NUMBER.findall(nxt)):
                    return True
    return False


def rule_tag(sentence):
    # returns [], ["numerical"], ["mixed"] or both
    tags = []
    if has_figure(sentence) and (COMPARISON_WORDS.search(sentence) or has_from_figure(sentence)):
        tags.append("numerical")
    if POSITIVE_WORDS.search(sentence) and NEGATIVE_WORDS.search(sentence):
        tags.append("mixed")
    return tags


def add_rule_tags(df):
    out = df.copy()
    tags = out.sentence.map(rule_tag)
    for cat in RULE_CATEGORIES:
        out[f"rule_{cat}"] = tags.map(lambda t: int(cat in t))
    return out


def load_manual_tags(tag_paths=DEFAULT_TAG_PATHS):
    # one row per coded train sentence, 0/1 per category.
    if isinstance(tag_paths, (str, Path)):
        tag_paths = [tag_paths]
    sheets = []
    for path in tag_paths:
        sheet = pd.read_csv(path)
        missing = set(SHEET_COLUMNS) - set(sheet.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        tags = sheet[["train_idx", "sentence", *SHEET_COLUMNS]].rename(columns=SHEET_COLUMNS)
        tags[CATEGORIES] = tags[CATEGORIES].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        sheets.append(tags)
    tags = pd.concat(sheets, ignore_index=True)
    return tags.groupby("train_idx").agg({"sentence": "first", **{c: "max" for c in CATEGORIES}})


def load_tagged_splits(tag_paths=DEFAULT_TAG_PATHS, data_dir=DATA_DIR):
    train = pd.read_csv(Path(data_dir) / "train.csv")
    val = pd.read_csv(Path(data_dir) / "val.csv")
    manual = load_manual_tags(tag_paths)

    if manual.index.max() >= len(train):
        raise ValueError("coding sheet has a train_idx outside train.csv")
    if not (train.sentence.iloc[manual.index].values == manual.sentence.values).all():
        raise ValueError("coding sheet sentences don't match train.csv at their train_idx")

    manual = manual[CATEGORIES].add_prefix("manual_").astype("Int64")
    train = add_rule_tags(train).join(manual)
    train["coded"] = train.index.isin(manual.index)
    val = add_rule_tags(val)
    val["coded"] = False
    for col in manual.columns:
        val[col] = pd.array([pd.NA] * len(val), dtype="Int64")
    return train, val


DEFAULT_CAT_WEIGHTS = {"ambiguous": 0.5, "numerical": 2.0, "mixed": 2.0, "company_event": 1.5, "domain": 1.5}


def category_matrix(df):
    # True/False per category
    out = pd.DataFrame(False, index=df.index, columns=CATEGORIES)
    coded = df["coded"].to_numpy(bool) if "coded" in df else np.zeros(len(df), bool)
    for cat in CATEGORIES:
        if f"rule_{cat}" in df:
            out[cat] = df[f"rule_{cat}"].to_numpy() == 1
        if f"manual_{cat}" in df:
            manual = df[f"manual_{cat}"].fillna(0).to_numpy() == 1
            out.loc[coded, cat] = manual[coded]
    return out


def make_weights(df, cat_weight=DEFAULT_CAT_WEIGHTS):
    # one weight per sentence. No category = 1.0.
    unknown = set(cat_weight) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"unknown categories {sorted(unknown)}, expected some of {CATEGORIES}")
    cats = category_matrix(df)
    weights = np.ones(len(df), dtype=np.float32)
    for i, row in enumerate(cats.to_numpy()):
        matched = [cat_weight[c] for c, on in zip(CATEGORIES, row) if on and c in cat_weight]
        if matched:
            weights[i] = min(matched) if min(matched) < 1 else max(matched)
    return weights


def category_counts(df):
    rows = {}
    for cat in CATEGORIES:
        rule_col = f"rule_{cat}"
        rows[cat] = {"rule": int(df[rule_col].sum()) if rule_col in df else pd.NA,
                     "manual": int(df[f"manual_{cat}"].sum())}
    return pd.DataFrame(rows).T.rename_axis("category")


if __name__ == "__main__":
    train, val = load_tagged_splits()
    for name, df in [("train", train), ("val", val)]:
        print(f"\n{name}: {len(df)} sentences, {df.coded.sum()} manually coded")
        print(category_counts(df).to_string())
        w = pd.Series(make_weights(df))
        print("weights:", w.value_counts().sort_index().to_dict(), f"mean={w.mean():.3f}")

    # How well the rules match the hand coding (coded train sentences only)
    coded = train[train.coded]
    print(f"\nrules vs manual tags on the {len(coded)} coded train sentences:")
    for cat in RULE_CATEGORIES:
        rule, manual = coded[f"rule_{cat}"] == 1, coded[f"manual_{cat}"] == 1
        tp = int((rule & manual).sum())
        print(f"  {cat:9s} rule={int(rule.sum()):3d} manual={int(manual.sum()):3d} both={tp:3d} "
              f"precision={tp / max(rule.sum(), 1):.2f} recall={tp / max(manual.sum(), 1):.2f}")
