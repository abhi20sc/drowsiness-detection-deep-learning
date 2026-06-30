# Vision-Based Driver Drowsiness Detection

A subject-independent benchmark of baseline and deep-learning models for vision-based driver
drowsiness detection, evaluated across three public datasets with a unified preprocessing and
evaluation protocol. Final-year engineering research project (ES327).

The study compares two industry-standard baselines against three deep video architectures, with an
emphasis on **cross-domain robustness**, **threshold calibration**, **explainability (Grad-CAM)**,
and **real-time deployment feasibility** for Advanced Driver-Assistance Systems (ADAS).

## 📄 Publication

This work has been **accepted at ICECET 2026** (International Conference on Electrical, Computer
and Energy Technologies) and is **forthcoming in IEEE Xplore** (indexing expected within ~2 months
of acceptance).

> **Baseline vs Exploratory Deep Learning Video Models for Driver Drowsiness Detection:
> A Multi-Dataset Evaluation**
> Accepted at *ICECET 2026* — to appear in IEEE Xplore.

> _The full citation and DOI will be added here once the paper is indexed._

## Models

**Baselines**
- **PERCLOS** — percentage of eye closure over time
- **Rule-Based Fusion (RBF)** — fusion of eyelid and mouth (yawn) dynamics

**Deep learning**
- **CNN-GRU** — ResNet-18 per-frame features → GRU (strongest cross-domain robustness; best external/DROZY performance)
- **TSM-fast** — ResNet-18 with temporal averaging (lightweight, balanced, deployment-friendly)
- **R(2+1)D-18** — 3D-ResNet (best internal/UTA-RLDD F1, but computationally heavy)

## Datasets

| Dataset | Role | Notes |
|---|---|---|
| UTA-RLDD | Internal (train/val/test) | Primary dataset |
| YawDD | Internal (train/val/test) | Natural driving behaviour (talking, yawning) |
| DROZY | External (test only) | Low-light footage; labels from Karolinska Sleepiness Scale (KSS) |

All experiments use **subject-independent splits** and identical preprocessing: clips of **16 frames**
at **112×112**, **10 fps**, **stride 8**, stored as `.npz` arrays.

### Data availability & licensing

> **No dataset data is redistributed in this repository.** UTA-RLDD, YawDD, and DROZY are the property
> of their respective authors and remain governed by their own access terms and licences. This repo
> contains **only my own code** (preprocessing, training, and evaluation notebooks) — no raw videos,
> generated clips, manifests, or trained checkpoints are included (all excluded via `.gitignore`).
>
> To reproduce, obtain each dataset **directly from its original source** and comply with that source's
> terms:
> - **UTA-RLDD** — University of Texas at Arlington Real-Life Drowsiness Dataset (request access from the authors).
> - **YawDD** — Yawning Detection Dataset (available from its original distributor under its own terms).
> - **DROZY** — University of Liège ULg Multimodality Drowsiness Database (requires signing the providers' licence/EULA).
>
> Please cite the original dataset papers when using them. The MIT licence on this repository covers
> **the code only** — it does not grant any rights to the third-party datasets or to model weights
> derived from them.

## Pipeline

The notebooks run in order and form the full pipeline:

| Notebook | Stage |
|---|---|
| `01_preprocessing_uta_rldd.ipynb` | Clip generation + manifests — UTA-RLDD |
| `02_preprocessing_yawdd.ipynb` | Clip generation + labelling — YawDD |
| `03_preprocessing_drozy.ipynb` | Clip generation + KSS labelling — DROZY |
| `04_baseline_models_perclos_rbf.ipynb` | PERCLOS + RBF baselines (MediaPipe EAR/MAR) |
| `05_deep_models_training.ipynb` | Train TSM, CNN-GRU, R(2+1)D-18 (subject-independent) |
| `06_threshold_calibration.ipynb` | Dual-threshold selection on validation (precision ≥ 0.60 / recall ≥ 0.70) |
| `07_final_test_evaluation.ipynb` | Frozen-threshold evaluation on all test sets → results table |
| `08_gradcam_analysis.ipynb` | Grad-CAM saliency analysis (TP/FP/FN/TN, talking-FP, yawning-TP) |
| `09_computational_efficiency.ipynb` | Inference throughput / latency benchmark |

> **Methodology note:** decision thresholds are selected on validation data only, then **frozen** and
> applied unchanged to the test and external sets — preventing test leakage. This boundary sits
> between notebooks 06 and 07.

## Key Findings

- Deep models outperform the baselines across nearly all metrics.
- **R(2+1)D-18** achieves the best internal (UTA-RLDD) F1, but its throughput (~3 clips/sec) makes it impractical for real-time ADAS.
- **CNN-GRU** has the strongest external (DROZY) performance and PR-AUC at low computational cost, making it the best overall candidate for deployment — though it is highly threshold-sensitive.
- **TSM** is the most balanced and computationally efficient, generalising well despite lower peak performance.
- Grad-CAM reveals why models misclassify natural behaviours (e.g. talking) as drowsy, motivating dataset-specific threshold calibration.

## Repository Structure

```
.
├── notebooks/            # full pipeline, run in numbered order
├── experiments/          # exploratory work not used in final results
│   └── cnn_gru_mar_fusion.ipynb   # CNN-GRU + Mouth-Aspect-Ratio fusion variant
├── requirements.txt
└── README.md
```

## Running

The notebooks were developed in Google Colab (GPU runtime) with data on Google Drive. Paths point to
`/content/drive/MyDrive/ES327_Drowsiness/...`; update these to your own layout. Datasets must be
obtained from their original sources (UTA-RLDD, YawDD, DROZY) under their respective licences — they
are **not** included here.

```bash
pip install -r requirements.txt
```

### Expected data layout

The notebooks assume the following directory structure on Google Drive (root:
`MyDrive/ES327_Drowsiness/`):

```
ES327_Drowsiness/
├── Datasets_Preprocessed/     # source videos, before clip generation
├── Clips/                     # generated .npz clips (output of preprocessing)
├── splits/                    # train/val/test split CSVs (subject-independent)
├── Manifests/                 # per-dataset clip manifests
├── PythonScripts/             # extraction scripts written/run by preprocessing notebooks
├── models/                    # face_landmarker.task (MediaPipe, for baselines)
├── baseline_outputs/          # PERCLOS + RBF results (probs.csv, results.json)
└── outputs_all_3_models_v2/   # final deep-model checkpoints + evaluation outputs
    ├── tsm_fast/best.pt
    ├── cnn_gru/best.pt
    ├── r3d18/best.pt
    └── final_test_eval/       # per-clip predictions + summary metrics
```

Folders under `Clips/`, `outputs_all_3_models_v2/`, and the datasets are large and excluded from
version control (see `.gitignore`).

## Notes

- `experiments/cnn_gru_mar_fusion.ipynb` is an exploratory CNN-GRU variant that adds a Mouth-Aspect-Ratio
  feature stream. It is **not** the CNN-GRU reported in the final results and is kept for reference only.
- Raw datasets and trained checkpoints are excluded from version control (see `.gitignore`).
