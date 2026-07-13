# Early Fraud Detection in Telecom to Minimize Revenue Loss Using Deep Learning

Experimental pipeline accompanying the MSc thesis of the same name (Imam Mohammad Ibn Saud
Islamic University, College of Computer and Information Sciences, MSc in Artificial Intelligence,
2026).

**Author:** Muneer Alnajdi
**Supervisor:** Dr. Fahman Saeed · **Co-supervisor:** Dr. Mohammad Alkhatib

---

## What this is

The thesis treats **observation length** — the number of events a model may see before issuing a
verdict — as the primary design variable of telecom fraud detection. Five model families are
evaluated under one uniform pipeline at three short observation windows (4, 6 and 16 earliest events
per subscriber), with anchor evaluations extending to the median history length of 1,645 events.

This repository contains the code that produced every reported result.

## Data

The experiments use the public telecom fraud dataset released for the **2020 Digital Sichuan
Innovation Competition** (6,106 subscribers, August 2019 – March 2020), organized by the Sichuan
Provincial Big Data Center and collected by China Mobile Sichuan.

**The dataset is not redistributed here.** Obtain it from the competition source and place it under
`dataset/`. The directory scaffolding is committed; the data is not.

Label distribution: **1,962 fraud / 6,106 subscribers (32.13%)**. Labels are read from the raw
`user.csv` registry — see *Key decisions* below.

---

## Pipeline at a glance

```
raw CSVs
  └─[1] eda_preprocessing ──────────► preprocessed CSVs
  └─[2] Clustring (figures only)
          │
          ▼
       [3] NAS × 12  (search per model × SL) ──► trial logs (results/NAS_v2)
          │
          ▼
       [4] NAS Analysis (assemble + select) ──► chosen configs
          │
          ▼
       [5] Best-of-each × 6 (retrain + evaluate) ──► summary / per-round tables
          │
          ▼
       [6] Excel: filtration + macro ──► final thesis numbers (Tables 5.1–5.8, 6.1–6.2)
```

## Pipeline stages

| # | Stage | Notebook(s) | Purpose | Inputs → Outputs |
|---|---|---|---|---|
| 1 | Preprocess + EDA | `eda_preprocessing.ipynb` | Clean the four raw tables (§4.3); profiling reports; t-SNE (Fig. 4.2) | raw `app/sms/user/voc.csv` → `preprocessed_*.csv` |
| 2 | Fraud-distribution analysis | `Clustring.ipynb` | Bubble chart per application (Fig. 4.1). Exploratory only — nothing downstream consumes it | raw CSVs → Fig. 4.1 |
| 3 | Architecture search | 12 notebooks: {TimesNet, LSTM, Transformer, RF+XGB} × WL_{4,6,16} | 100 Optuna TPE trials per (model, SL); objective = **validation** F1; test metrics logged, never used for selection. Manual seeds enqueued per study | preprocessed CSVs + label registry + frozen split → `results/NAS_v2/nas_*_{trial_log,results}_WL{sl}.csv` |
| 4 | Selection | `NAS Analysis.ipynb` | Assembles trial logs and applies the frozen validation-stability rule (band 0.01 → val AUC → smaller model → trial id) to pick one trial per (model, SL) per population | `results/NAS_v2/*.csv` → `NAS_RESULT_v2/*.txt` + selection CSVs |
| 5 | Consolidated evaluation | `Best of each_Base{4,6,16}` + `Manual Best of each` × 3 | Retrain each selected configuration under the unified protocol; validation-calibrated threshold (61-point grid over 0.20–0.80); test metrics, FLOPs, latency, per-round evaluation | selected configs + preprocessed CSVs + frozen split → summary and per-round tables, Figs. 5.1–5.9 |
| 6 | Winners + macro | `consolidated_results.xlsx` (no code) | Filtration trace and per-SL winner (§4.11, §6.3); macro metrics derived from the confusion matrix (test: 393 fraud / 829 normal) | stage-5 tables → Tables 5.5, 6.1, 6.2 |

## Repository layout

| Path | Contents | Thesis section |
|---|---|---|
| `notebooks/` | Preprocessing, EDA, NAS, consolidated evaluation, per-round replay | §4.3 – §4.5, §4.10, §4.12 |
| `src/`, `lib/` | Shared pipeline modules | §4.3 – §4.5 |
| `configs/baseline.yaml` | for paths: `root_path`, `ML.labels`, `ML.Events`, `output.results_dir`. All notebooks resolve paths through it | — |
| `splits/shared_user_split_v1/` | Frozen subscriber partition: **3,907 / 977 / 1,222** users (fraud **1,255 / 314 / 393**) | §4.4 |
| `models/` | Random Forest, XGBoost, LSTM and Transformer definitions | §4.7 |
| `Time-Series-Library/` | TimesNet, via a modified fork of the THUML library — see below | §4.7, §4.8 |
| `ExternalDataset/TimesNet/classification/` | Oldest-window `.ts` sequences for TimesNet (seven features padded to thirteen channels) | §4.5, §4.8 |
| `results/NAS_v2/` | Per-study Optuna trial logs — the raw search evidence | §4.10 |
| `results/`, `final_report_all_models.csv` | Consolidated test-set results and per-Round trajectories | §5 |
| `consolidated_results.xlsx` | Consolidated table, filtration trace, winners, macro sheet | §5, §6.3 |
| `ML_requirements-lock.txt` | Pinned environment | §5.2 |
| `run_experiment.py` | Entry point | — |

## Modified Time-Series-Library

`Time-Series-Library/` is a fork of
[thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library), retrieved at commit
`3846bfb` on 10 November 2025. The upstream classification pipeline is written around the UEA
benchmark archive, which ships a single train/test split; it therefore loads the test partition a
second time in place of a validation set and monitors it during training.

That convention is incompatible with the protocol of this thesis, which spends held-out data twice
— once selecting the architecture-search configuration and once calibrating the decision threshold.
The fork changes the following. The TimesNet architecture itself is **unmodified**.

| Component | Upstream | This fork |
|---|---|---|
| Validation partition | None — test reused as validation | Dedicated validation partition, disjoint from test |
| Selection criterion | Accuracy, on the test partition | Fraud-class F1, on the validation partition |
| Early stopping | Monitors accuracy | Monitors validation F1 |
| Decision rule | Arg-max over logits | Softmax → fraud probability → calibrated threshold |
| Threshold calibration | None | 61-point sweep over [0.20, 0.80] inside the validation loop |
| Validation ordering | Shuffled | Fixed order, as for test |
| Experiment classes | One (`Exp_Classification`) | Two, dispatched by `--mode`: Unfrozen, PartialUnfreeze |
| Seeding | Unseeded entry point | Python / NumPy / PyTorch seeded from one argument |
| Persisted artefacts | Accuracy only | Loss and F1 history; fraud probabilities and labels for train, validation and test |

Files touched: `run.py`, `data_provider/data_factory.py`,
`exp/exp_classification_save_full_train_opt_WL.py`,
`exp/exp_classification_save_partialUnfreeze.py`.

The TimesNet backbone is initialized from a checkpoint pre-trained on **SpokenArabicDigits** (THUML);
the seven per-event telecom features are padded to the checkpoint's thirteen input channels.

## Verification checkpoints

These must hold on every run. They are asserted in the notebooks, not merely documented.

- **Label merge.** `Fraud users: 1,962 / 6,106 (32.13%)` — labels read from the raw registry. The
  assertion fails the run if any label is missing.
- **Split.** Sizes `3907 / 977 / 1222`, fraud `1255 / 314 / 393`, and the log must read
  `Using shared user split files` — never `Creating`.
- **Window semantics.** Every model consumes each subscriber's **oldest** `max_seq_len` events
  (`selector_oldest` throughout; trees via `head(r)`; per-round via `selector_oldest(r)`).
- **Selection is validation-only.** Band 0.01 → validation AUC → smaller model → trial id. Test
  metrics are logged for reporting, never for choosing.
- **Threshold calibration.** Validation only, 61-point grid over 0.20–0.80 — identical in every
  notebook.
- **Study health.** 100 COMPLETE trials per study, all rows OK, sampler `TPE(seed=42,
  n_startup=10)`, no pruning — every trial runs to completion, giving a complete response surface.

## Key decisions

- **Labels come from the raw `user.csv` registry**, not from the preprocessed USER table.
  Preprocessing (§4.3) drops zero-ARPU months, which removes 177 subscribers from that table — 175
  of them fraudulent. Deriving labels downstream of that filter would silently relabel them as
  normal.
- **Oldest-window sequences.** §4.5 defines the model input as each subscriber's *first* SL events;
  the early-detection objective requires it. `selector_oldest` is used everywhere, for every family.
- **The partition is frozen in files** (`shared_user_split_v1`, stratified) so that every experiment
  ever run shares the same test subscribers (§4.4). It is never redrawn.
- **The selection rule was frozen before the final search ran.** The pipeline is pre-registered in
  order: rule → search → selection → consolidation → report. No rule was chosen after seeing results.
- **No pruner.** All 100 trials per study run to completion, so every configuration is fully
  evaluated.
- **Manual (enqueued) seeds** are re-run under identical conditions and form the Manual population
  for the §6.5 NAS-versus-manual ablation.

## Notes for anyone re-running this

- **Paths are not portable.** The notebooks were run in Google Colab and reference Google Drive
  paths. Repoint them via `configs/baseline.yaml` before running.
- **Seeds.** Headline runs use seed 2021 (TimesNet, via the Time-Series-Library entry point) and 42
  (LSTM, Transformer, Random Forest, XGBoost; also the Optuna TPE sampler). Seed sensitivity is
  reported in the thesis.
- **Do not regenerate the split.** `splits/shared_user_split_v1/` is written once and re-loaded by
  every experiment. Regenerating it will not reproduce the reported numbers.



## Citation

```bibtex
@mastersthesis{alnajdi2026earlyfraud,
  author = {Alnajdi, Muneer},
  title  = {Early Fraud Detection in Telecom to Minimize Revenue Loss Using Deep Learning},
  school = {Imam Mohammad Ibn Saud Islamic University},
  year   = {2026}
}
```

## Acknowledgements

Time-Series-Library and the TimesNet checkpoint are the work of the Tsinghua University Machine
Learning Lab (THUML). Optuna is used for the architecture search.
