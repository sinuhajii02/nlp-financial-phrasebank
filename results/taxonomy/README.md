# Taxonomy-guided training results

Continued training of the frozen FinBERT CNN from one baseline checkpoint, with per-sentence weights from the
error taxonomy (`targeted`), against a matched random control (`random`: the same weights, same count, on
random train rows, same seed). All scores are validation macro-F1; nothing here uses the test set.

Weights (`weighting=all`): ambiguous 0.5, numerical 2.0, mixed 2.0, company_event 1.5, domain 1.5, else 1.0
(500 of 2,901 train sentences reweighted).

Files: `weighted_runs.csv` (one row per run), `weighted_summary.csv` (mean/std per setting, arm and group),
`coding_sheet_*` / `error_pool.csv` (manual coding, see notebook 08). Per-run probabilities in `probs/` and the
archived lr 1e-5 ablation in `results/archive/` are not tracked in git.

## Results (weighting = all)

`baseline` is the checkpoint itself (one model, no std). `targeted - random` is the mean of the per-seed
paired difference. Values are mean +/- std across seeds.

| lr | stopping | seeds | arm | overall (n=967) | hard (n=47) | numerical (n=129) |
|---|---|---|---|---|---|---|
| — | — | 1 | baseline | 0.7653 | 0.6058 | 0.7689 |
| 1e-5 | val loss | 5 | targeted | 0.7629 +/- 0.0011 | 0.5426 +/- 0.0000 | 0.7630 +/- 0.0000 |
| 1e-5 | val loss | 5 | random | 0.7636 +/- 0.0010 | 0.5426 +/- 0.0000 | 0.7630 +/- 0.0000 |
| 1e-5 | val loss | 5 | **targeted - random** | **-0.0006 +/- 0.0007** | **0.0000 +/- 0.0000** | **0.0000 +/- 0.0000** |
| 3e-5 *(preliminary)* | val macro-F1 | 2 | targeted | 0.7495 +/- 0.0035 | 0.4120 +/- 0.0169 | 0.7720 +/- 0.0097 |
| 3e-5 *(preliminary)* | val macro-F1 | 2 | random | 0.7469 +/- 0.0057 | 0.4000 +/- 0.0000 | 0.7753 +/- 0.0051 |
| 3e-5 *(preliminary)* | val macro-F1 | 2 | **targeted - random** | **+0.0026 +/- 0.0022** | **+0.0120 +/- 0.0169** | **-0.0033 +/- 0.0046** |
| 1e-4 *(preliminary)* | val loss | 2 | targeted | 0.7448 +/- 0.0049 | 0.4120 +/- 0.0169 | 0.7659 +/- 0.0010 |
| 1e-4 *(preliminary)* | val loss | 2 | random | 0.7409 +/- 0.0077 | 0.4120 +/- 0.0169 | 0.7535 +/- 0.0082 |
| 1e-4 *(preliminary)* | val loss | 2 | **targeted - random** | **+0.0039 +/- 0.0028** | **0.0000 +/- 0.0000** | **+0.0123 +/- 0.0093** |

The single-category ablation at lr 1e-5 (only ambiguous / company_event / domain / mixed) is in
`weighted_summary.csv`; every targeted - random difference there is within +/- 0.001 overall.

## Caveats

- **Few hand-coded sentences.** Only 120 train sentences are hand-coded (sheets B and D, 20 shared). Ambiguous,
  domain and company_event exist only on those 120; numerical and mixed also use rules on the other train rows.
- **Only numerical and mixed can be scored on validation.** Val has no manual tags, and only those two
  categories have rules (numerical n=129, mixed n=15). The other three categories have n=0 on val.
- **The lr 1e-5 ablation was stopped before finishing** (54 of 60 rows; `only_numerical` and
  `only_company_event` random seed 46 are missing). lr 1e-5 was too small to change predictions. Its
  probabilities are archived in `results/archive/stopped_lr1e-5_ablation/`; its scores remain in the CSVs.
- **Preliminary settings.** 3e-5 and 1e-4 use 2 seeds (42, 43) and different stopping rules, so they are not
  directly comparable with each other and the differences are not reliable yet. On 47 hard sentences one
  sentence moves F1 by about 0.02.
- **Both arms end below the baseline checkpoint** at every learning rate. The checkpoint was itself the
  best-of-16-epochs validation peak (early stopping on val macro-F1), so further training tends to regress
  from it. Read `targeted - random` as the result; the baseline row is a reference, not the level to beat.

## Commands

Run from the repo root. Each run writes to `weighted_runs.csv`, then rebuilds `weighted_summary.csv`.

```bash
# lr 1e-5 rows (default lr, val-loss stopping); stopped early, see caveats
python scripts/taxonomy_training.py --ablation

# lr 3e-5, preliminary
python scripts/taxonomy_training.py --lr 3e-5 --seeds 42 43 --stop-on val_macro_f1

# lr 1e-4, preliminary
python scripts/taxonomy_training.py --lr 1e-4 --seeds 42 43

# rebuild weighted_summary.csv and print the tables, no training
python scripts/taxonomy_training.py --summary-only
```

The first run trains the baseline checkpoint (`models/finbert_cnn_frozen_baseline.keras`, seed 42, not in git)
if it does not exist; later runs reuse it and never overwrite it.
