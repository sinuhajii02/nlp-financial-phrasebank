# Scoring for the taxonomy-guided training: macro F1 per group (overall, hard, non-hard, each category)
# and the mean / std over seeds.
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from utils.taxonomy import CATEGORIES, category_matrix

GROUPS = ["overall", "hard", "nonhard", *CATEGORIES]
CONFIG_KEY = ["embedding_config", "learning_rate", "stop_on", "weighting"]


def fill_legacy_columns(runs):
    # runs saved before the weighting / stop_on options existed used "all" and val-loss stopping
    return runs.assign(**{c: runs[c].fillna(v) if c in runs else v
                          for c, v in [("weighting", "all"), ("stop_on", "val_loss")]})


def eval_groups(tagged, band):
    # True/False per group. hard = is_hard from band_val.csv, categories from category_matrix
    if not (band.sentence.values == tagged.sentence.values).all():
        raise ValueError("band and tagged rows are not the same sentences in the same order")
    hard = band.is_hard.to_numpy(bool)
    groups = {"overall": np.ones(len(tagged), bool), "hard": hard, "nonhard": ~hard}
    cats = category_matrix(tagged)
    groups.update({c: cats[c].to_numpy(bool) for c in CATEGORIES})
    return groups


def score_groups(y_true, probs, groups):
    # size and macro F1 of every group (NaN if the group is empty)
    y_true = np.asarray(y_true)
    pred = np.asarray(probs).argmax(axis=1)
    row = {}
    for name, mask in groups.items():
        row[f"n_{name}"] = int(mask.sum())
        row[f"f1_{name}"] = (f1_score(y_true[mask], pred[mask], labels=[0, 1, 2], average="macro", zero_division=0)
                             if mask.any() else np.nan)
    return row


def group_rows(runs, arm, groups):
    rows = []
    for name in groups:
        f1 = runs[f"f1_{name}"]
        rows.append({"arm": arm, "group": name, "n": int(runs[f"n_{name}"].iloc[0]),
                     "n_seeds": int(f1.notna().sum()), "f1_mean": f1.mean(), "f1_std": f1.std(ddof=1)})
    return rows


def summarise_runs(runs, groups=GROUPS, target="targeted", control="random"):
    # one row per setting, arm and group: mean and std of macro F1 over the seeds.
    # also adds "targeted - random": the difference for the same seed, averaged
    runs = fill_legacy_columns(runs)
    rows = []
    for config, runs_c in runs.groupby(CONFIG_KEY):
        prefix = dict(zip(CONFIG_KEY, config))
        for arm, runs_a in runs_c.groupby("arm"):
            rows += [{**prefix, **r} for r in group_rows(runs_a, arm, groups)]
        t = runs_c[runs_c.arm == target].set_index("seed")
        c = runs_c[runs_c.arm == control].set_index("seed")
        seeds = t.index.intersection(c.index)
        if len(seeds):
            diff = t.loc[seeds, [f"f1_{g}" for g in groups]] - c.loc[seeds, [f"f1_{g}" for g in groups]]
            diff[[f"n_{g}" for g in groups]] = t.loc[seeds, [f"n_{g}" for g in groups]]
            rows += [{**prefix, **r} for r in group_rows(diff, f"{target} - {control}", groups)]
    table = pd.DataFrame(rows)
    order = {g: i for i, g in enumerate(groups)}
    return table.sort_values([*CONFIG_KEY, "arm", "group"], key=lambda s: s.map(order) if s.name == "group" else s,
                             ignore_index=True)


def format_table(table):
    # table for printing: groups as rows, one column per arm
    cell = table.f1_mean.map("{:.4f}".format) + " +/- " + table.f1_std.map("{:.4f}".format)
    wide = table.assign(cell=cell.where(table.f1_mean.notna(), "-")).pivot_table(
        index=[*CONFIG_KEY, "group", "n"], columns="arm", values="cell", aggfunc="first", sort=False)
    return wide.reset_index(level="n")


FINAL_GROUPS = ["overall", "hard", "nonhard", "numerical", "mixed"]   # the groups val can score


def final_table(table, groups=FINAL_GROUPS):
    # the final table: one row per setting and arm
    sizes = table.drop_duplicates("group").set_index("group").n
    rows = []
    for key, part in table.groupby([*CONFIG_KEY, "arm"], sort=False):
        cfg = dict(zip([*CONFIG_KEY, "arm"], key))
        by_group = part.set_index("group")
        row = {"setting": f"lr={cfg['learning_rate']:g} stop={cfg['stop_on']} weighting={cfg['weighting']}",
               "arm": cfg["arm"], "seeds": int(by_group.n_seeds.max())}
        for g in groups:
            m, s = by_group.loc[g, "f1_mean"], by_group.loc[g, "f1_std"]
            row[f"{g} (n={sizes[g]})"] = f"{m:.4f}" if pd.isna(s) else f"{m:.4f} +/- {s:.4f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index(["setting", "arm"])


def ablation_view(table, arm="targeted - random"):
    # targeted - random per setting, groups as columns
    view = table[table.arm == arm].pivot_table(index=CONFIG_KEY, columns="group", values="f1_mean", sort=False)
    return view[[g for g in GROUPS if g in view.columns]]
