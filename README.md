# EMBase: Benchmark and Unmixing Suite for Hyperspectral Imaging

Official code and reproduction suite for the paper comparing deep latent representations and classical dimensionality reduction methods for hyperspectral linear unmixing.

---

## 📋 Overview

This repository provides an autonomous, reproducible benchmark evaluating 6 representation learning methods across 3 standard ground-truth hyperspectral datasets:

- **Methods evaluated:**
  - `PCA` (Principal Component Analysis - Baseline)
  - `AE` (Standard Autoencoder)
  - `AE_PATCH` (Patch-based Autoencoder)
  - `ViT` (Vision Transformer autoencoder)
  - `MAE` (Masked Autoencoder)
  - `DINO` (Self-distillation with no labels)
- **Benchmark Datasets:**
  - `Samson` (95 × 95 px, 156 bands, 3 endmembers)
  - `Jasper Ridge` (100 × 100 px, 198 bands, 4 endmembers)
  - `Urban` (307 × 307 px, 162 bands, 4 endmembers)
- **Unmixing Engine:**
  - Endmember extraction: Vectorized `N-FINDR Turbo`
  - Abundance estimation: Vectorized `FCLS Turbo` (ANC + ASC constraints)
  - Hungarian algorithm pairing against Ground Truth (`scipy.optimize.linear_sum_assignment`)

---

## 🛠️ Repository Structure

```text
EMVisual-Paper/
├── config.yaml               # Centralized configuration (datasets & models)
├── run_experiments.py        # End-to-end benchmark orchestrator
├── environment.yml           # Conda environment definition
├── requirements.txt          # Python dependencies
├── src/                      # Core algorithm modules
│   ├── dataset_loaders_gt.py # Ground-truth loaders (Samson, Jasper, Urban)
│   ├── dim_reduction.py      # Dimensionality reduction architectures
│   ├── unmixing.py           # N-FINDR and FCLS implementations
│   ├── metrics.py            # Quantitative evaluation (mSAD, SAM, aRMSE)
│   ├── pipeline.py           # Pipeline runner & Hungarian matcher
│   ├── common.py             # Math helpers
│   ├── visuals.py            # Spectral and simplex plotting helpers
│   └── loaders.py            # Data loading interfaces
├── data/                     # Benchmark hyperspectral cubes (.mat)
│   ├── jasper/
│   ├── samson/
│   └── urban/
├── results_cl/               # Precomputed numerical results & model weights
│   ├── Global_Summary/       # Consolidated master_metrics.csv
│   ├── Unmixing/             # Extracted endmembers & abundances (.npy)
│   ├── Models/               # Trained model checkpoints (.weights.h5)
│   └── Reports/              # Quantitative metrics per run (.json / .csv)
└── paper_figures/            # Paper figures, tables, and generator scripts
    ├── generate_paper_plots.py      # Pareto front, heatmap, spectral comparison
    ├── generate_reconstructions.py  # Error heatmaps & reconstructed cubes
    ├── table/
    │   └── table_metrics.tex        # LaTeX quantitative results table
    └── *.png                        # Rendered publication figures (300 DPI)
```

---

## ⚙️ Installation

### Option 1: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate embase_paper
```

*Or use your existing environment (e.g. `pysptools_env`):*
```bash
conda activate pysptools_env
```

### Option 2: Pip

```bash
pip install -r requirements.txt
```

---

## 📊 Reproducing Paper Figures & Tables

All numerical outputs required to generate the paper figures are precomputed and included in `results_cl/`. You can reproduce all figures in seconds without needing a GPU:

### 1. Generate Efficiency Frontier, Heatmaps & Spectral Comparisons
```bash
python paper_figures/generate_paper_plots.py
```
Outputs generated in `paper_figures/`:
* `frontera_eficiencia.png` — Multi-objective Pareto frontier ($mSAD$ vs $aRMSE$).
* `heatmap_rendimiento.png` — Normalized performance matrix across all methods.
* `08_endmembers_vs_gt_samson.png` — Extracted endmembers vs Ground Truth (Samson).
* `08_endmembers_vs_gt_jasper.png` — Extracted endmembers vs Ground Truth (Jasper Ridge).
* `08_endmembers_vs_gt_urban.png` — Extracted endmembers vs Ground Truth (Urban).

### 2. Generate Spatial Reconstruction & Error Heatmaps
```bash
python paper_figures/generate_reconstructions.py
```
Outputs generated in `paper_figures/`:
* `reconstruccion_samson.png` — Error map comparison with unified color scale.
* `reconstruccion_jasper_ridge.png` — Error map comparison with unified color scale.
* `reconstruccion_urban.png` — Error map comparison with unified color scale.

### 3. LaTeX Table
The quantitative comparison table is available in:
* `paper_figures/table/table_metrics.tex`

---

## 🚀 Running End-to-End Experiments from Scratch

To retrain the models and re-execute the entire unmixing pipeline:

```bash
python run_experiments.py
```

Hyperparameters, epochs, and target datasets can be configured in `config.yaml`.
