# NLP Financial PhraseBank
## Research Question
Can identifying and explicitly training on difficult financial-language phenomena improve sentiment classification more effectively than simply adding more (random) training data? Can targeted training on a small model even outperform an LLM's zero/few-shot performance?

## Run order

00 split -> 01 baselines -> 02 FinBERT CNN and hard sets -> 06 agreement levels -> 03 Qwen benchmark -> 04 FinBERT transformer and targeted training -> 05 clean-hard on the CNN -> 07 error taxonomy.
06 only builds `results/hard_samples/agreement_*.csv`, which 04 and 07 read, so it runs before them.

## Taxonomy-guided training
Continued training of the frozen FinBERT CNN from one baseline checkpoint, with per-sentence weights from the
error taxonomy, compared with a matched random control (same weights, same count, same seed). Results and
caveats: [results/taxonomy/README.md](results/taxonomy/README.md).

```
utils/taxonomy.py             rule tags (numerical, mixed), manual-tag loader, make_weights
utils/continued_training.py   baseline checkpoint, continue_training, matched random control weights
utils/taxonomy_eval.py        macro-F1 per group (overall / hard / non-hard / category), results tables
scripts/taxonomy_training.py  runs targeted vs random over seeds, writes the results CSVs
notebook/07_error_taxonomy.ipynb   error pool, codebook, coder agreement and the numbers for report sections 5.4 and 5.5
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
## Use of AI tools

Two parts of the code were written with an AI coding assistant:
- the taxonomy-guided training: `utils/taxonomy.py`, `utils/taxonomy_eval.py`, `utils/continued_training.py` and
  `scripts/taxonomy_training.py`
- the targeted training on the transformer: the "Targeted training" part of `notebook/04_transformer_reference.ipynb`
  (from "Training with weights" to "Results"; the baseline transformer above it is our own)

We reviewed this code and ran the experiments ourselves.
For the rest of the project we used AI tools as an assistant (debugging, restructuring code, and checking that the numbers
in the report match the code). The experiment design, the runs and the analysis are our own.
