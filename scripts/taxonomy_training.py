# Taxonomy-guided training vs a random control with the same weights (report section 5.6).
# Both start from the same saved baseline and use the same seed. Scored on val only.
#
#   python scripts/taxonomy_training.py                       # lr 1e-5
#   python scripts/taxonomy_training.py --lr 3e-5 --seeds 42 43
#   python scripts/taxonomy_training.py --ablation            # all categories, then each one alone
#   python scripts/taxonomy_training.py --summary-only        # only rebuild the tables
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".cache/huggingface/hub"))

import numpy as np
import pandas as pd
from tensorflow import keras

from utils.continued_training import (DEFAULT_CHECKPOINT, STOP_MODES, continue_training,
                                      ensure_baseline_checkpoint, matched_random_weights)
from utils.finbert_cnn import BASELINE_SEED, SEEDS, load_finbert_inputs
from utils.metrics import score_predictions
from utils.preprocessing import encode_labels
from utils.taxonomy import CATEGORIES, DEFAULT_CAT_WEIGHTS, load_tagged_splits, make_weights
from utils.taxonomy_eval import (ablation_view, eval_groups, fill_legacy_columns, final_table, format_table,
                                 score_groups, summarise_runs)

RUN_KEY = ["weighting", "arm", "embedding_config", "learning_rate", "stop_on", "seed"]


def weighting_configs(only=None, ablation=False):
    # name -> {category: weight}. "all" = every category at its default weight,
    # "only_<cat>" = just that category at its default weight, every other sentence 1.0
    if ablation:
        names = ["all", *CATEGORIES]
    else:
        names = [only or "all"]
    return {("all" if n == "all" else f"only_{n}"):
            (dict(DEFAULT_CAT_WEIGHTS) if n == "all" else {n: DEFAULT_CAT_WEIGHTS[n]}) for n in names}


def score(y_val, probs, groups):
    # accuracy / per-class F1 as in metrics.csv, plus n and macro-F1 for every group
    row = score_predictions(y_val, probs, np.where(groups["hard"])[0])
    row.update(score_groups(y_val, probs, groups))
    return row


def write_summary(runs_path, summary_path):
    table = summarise_runs(pd.read_csv(runs_path))
    table.to_csv(summary_path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\nval macro-F1, mean +/- std across seeds:")
        print(format_table(table).to_string())
        print("\ntargeted - random (mean paired gain in val macro-F1), per setting:")
        print(ablation_view(table).round(4).to_string())
        print("\n=== FINAL: val macro-F1, mean +/- std across seeds (baseline = the checkpoint, 1 model) ===")
        print(final_table(table).to_string())
    print(f"\nruns: {runs_path}\nsummary: {summary_path}")


def save_runs(path, new_rows):
    # a repeated run replaces the old row
    if path.exists():
        new_rows = pd.concat([fill_legacy_columns(pd.read_csv(path)), new_rows], ignore_index=True)
    out = new_rows.drop_duplicates(subset=RUN_KEY, keep="last").sort_values(RUN_KEY).reset_index(drop=True)
    out.to_csv(path, index=False)
    return out


def main():
    parser = argparse.ArgumentParser(description="Taxonomy-guided training vs random control")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--stop-on", choices=list(STOP_MODES), default="val_loss",
                        help="early-stopping metric; the best epoch on it is restored")
    parser.add_argument("--finetuned", action="store_true", help="fine-tuned instead of frozen FinBERT embeddings")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results/taxonomy/weighted_runs.csv")
    parser.add_argument("--summary", type=Path, default=None,
                        help="results table (default: weighted_summary.csv next to --out)")
    parser.add_argument("--summary-only", action="store_true", help="only rebuild the table from --out")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--only", choices=CATEGORIES, help="up-/down-weight only this category")
    mode.add_argument("--ablation", action="store_true", help="run 'all', then every category on its own")
    args = parser.parse_args()
    summary_path = args.summary or args.out.with_name("weighted_summary.csv")
    if args.summary_only:
        write_summary(args.out, summary_path)
        return

    embedding_config = "finbert_finetuned" if args.finetuned else "finbert_frozen"
    checkpoint = args.checkpoint or (DEFAULT_CHECKPOINT if not args.finetuned
                                     else DEFAULT_CHECKPOINT.with_name("finbert_cnn_finetuned_baseline.keras"))
    probs_dir = args.out.parent / "probs"
    probs_dir.mkdir(parents=True, exist_ok=True)

    train_tagged, val_tagged = load_tagged_splits()
    y_train, y_val = encode_labels(train_tagged.label), encode_labels(val_tagged.label)
    X_train, X_val, embedding_matrix = load_finbert_inputs(train_tagged, val_tagged)
    groups = eval_groups(val_tagged, pd.read_csv(ROOT / "results/hard_samples/band_val.csv"))
    print("val group sizes:", {g: int(m.sum()) for g, m in groups.items()})

    ensure_baseline_checkpoint(checkpoint, embedding_matrix, X_train, y_train, X_val, y_val,
                               trainable=args.finetuned)
    print(f"checkpoint: {checkpoint}")

    def record(weighting, arm, seed, probs, weights=None, history=None):
        row = {"weighting": weighting, "arm": arm, "embedding_config": embedding_config,
               "learning_rate": args.lr, "stop_on": args.stop_on, "seed": seed,
               "n_weighted": int((weights != 1.0).sum()) if weights is not None else 0,
               "weight_sum": float(weights.sum()) if weights is not None else float(len(y_train)),
               "epochs_run": len(history.history["loss"]) if history else 0,
               "best_epoch": 0}
        if history:
            monitored = np.array(history.history[args.stop_on])
            best = monitored.argmin() if STOP_MODES[args.stop_on] == "min" else monitored.argmax()
            row["best_epoch"] = int(best) + 1
        row.update(score(y_val, probs, groups))
        stop_tag = "" if args.stop_on == "val_loss" else f"_{args.stop_on}"   # keeps the old file names
        np.save(probs_dir / f"{weighting}_{arm}_{embedding_config}_lr{args.lr:g}{stop_tag}_s{seed}.npy", probs)
        save_runs(args.out, pd.DataFrame([row]))
        print(f"{weighting:20s} {arm:9s} seed={seed}: f1={row['f1_overall']:.4f} hard={row['f1_hard']:.4f} "
              f"numerical={row['f1_numerical']:.4f} mixed={row['f1_mixed']:.4f} best_epoch={row['best_epoch']}")

    # the checkpoint itself, for reference (same model for every weighting)
    baseline_probs = keras.models.load_model(checkpoint, compile=False).predict(X_val, batch_size=32, verbose=0)

    for weighting, cat_weight in weighting_configs(args.only, args.ablation).items():
        target_weights = make_weights(train_tagged, cat_weight)
        print(f"\n== {weighting} {cat_weight}: {int((target_weights != 1.0).sum())} rows != 1.0, "
              f"{pd.Series(target_weights).value_counts().sort_index().to_dict()}")
        record(weighting, "baseline", BASELINE_SEED, baseline_probs)
        for seed in args.seeds:
            control_weights = matched_random_weights(target_weights, seed)
            assert np.array_equal(np.sort(control_weights), np.sort(target_weights))
            for arm, weights in [("targeted", target_weights), ("random", control_weights)]:
                keras.utils.set_random_seed(seed)
                model, history = continue_training(checkpoint, X_train, y_train, X_val, y_val,
                                                   sample_weight=weights, learning_rate=args.lr,
                                                   epochs=args.epochs, patience=args.patience,
                                                   stop_on=args.stop_on)
                record(weighting, arm, seed, model.predict(X_val, batch_size=32, verbose=0), weights, history)
                keras.backend.clear_session()

    write_summary(args.out, summary_path)


if __name__ == "__main__":
    main()
