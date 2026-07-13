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

## Repository layout

| Path | Contents | Thesis section |
|---|---|---|
| `notebooks/` | Preprocessing, EDA, feature engineering, snapshot construction, per-round replay | §4.3 – §4.5, §4.12 |
| `src/`, `lib/` | Shared pipeline modules | §4.3 – §4.5 |
| `splits/shared_user_split_v1/` | The frozen subscriber-level train / validation / test partition, reused unchanged by every experiment | §4.4 |
| `models/`, `configs/` | Random Forest, XGBoost, LSTM and Transformer definitions and selected configurations | §4.7 |
| `Time-Series-Library/` | TimesNet, via a modified fork of the THUML library — see below | §4.7, §4.8 |
| `nas_*_trial_log.csv`, `nas_*_results.csv` | Optuna trial logs and selections, one file per family | §4.10 |
| `results/`, `final_report_all_models.csv` | Consolidated test-set results and per-Round trajectories | §5 |
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

## Notes for anyone re-running this

- **Paths are not portable.** The notebooks were run in Google Colab and reference Google Drive
  paths. Repoint them to your own storage before running.
- **Seeds.** Headline runs use seed 2021 (TimesNet, via the Time-Series-Library entry point) and 42
  (LSTM, Transformer, Random Forest, XGBoost). Seed sensitivity is reported in the thesis.
- **The split is frozen.** `splits/shared_user_split_v1/` is written once and re-loaded by every
  experiment; it is never redrawn. Do not regenerate it if you want to reproduce the reported
  numbers.

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
