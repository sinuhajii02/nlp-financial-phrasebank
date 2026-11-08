# NLP Financial PhraseBank
## Research Question
Can identifying and explicitly training on difficult financial-language phenomena improve sentiment classification more effectively than simply adding more (random) training data? Can targeted training on a small model even outperform an LLM's zero/few-shot performance?

## Taxonomy-guided training
Continued training of the frozen FinBERT CNN from one baseline checkpoint, with per-sentence weights from the
error taxonomy, compared with a matched random control (same weights, same count, same seed). Results and
caveats: [results/taxonomy/README.md](results/taxonomy/README.md).

```
utils/taxonomy.py             rule tags (numerical, mixed), manual-tag loader, make_weights
utils/continued_training.py   baseline checkpoint, continue_training, matched random control weights
utils/taxonomy_eval.py        macro-F1 per group (overall / hard / non-hard / category), results tables
scripts/taxonomy_training.py  runs targeted vs random over seeds, writes the results CSVs
notebook/08_error_taxonomy.ipynb   error pool, codebook and coding sheets
results/taxonomy/             coding sheets, weighted_runs.csv, weighted_summary.csv, README.md
results/taxonomy/probs/       per-run val probabilities (not in git)
results/archive/              archived runs, e.g. the stopped lr 1e-5 ablation (not in git)
models/                       baseline checkpoint, created on the first run (not in git)
test/taxonomy_test.py, test/taxonomy_eval_test.py   unit tests
```

Run from the repo root:

```bash
python scripts/taxonomy_training.py --lr 3e-5 --stop-on val_macro_f1   # targeted vs random, 5 seeds
python scripts/taxonomy_training.py --ablation                         # all categories, then each one alone
python scripts/taxonomy_training.py --summary-only                     # rebuild the tables, no training
```