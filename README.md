# Benchmark and Unmixing Suite for Hyperspectral Imaging

Official code and reproduction suite for the paper comparing deep latent representations and classical dimensionality reduction methods for hyperspectral linear unmixing.

---

## 📋 Overview

This repository provides an autonomous, reproducible benchmark evaluating 6 representation learning methods across 3 standard ground-truth hyperspectral datasets:

- **Methods evaluated:**
  - `PCA` (Principal Component Analysis - Baseline)
  - `AE` (Standard Autoencoder)
  - `AE-Patch` (Patch-based Autoencoder)
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
│   ├── plot_simplex_3d.py    # Interactive 2D/3D simplex renderer
│   └── loaders.py            # Data loading interfaces
├── data/                     # Benchmark hyperspectral cubes (.mat)
│   ├── jasper/
│   ├── samson/
│   └── urban/
├── results_cl/               # Precomputed numerical results & model weights
│   ├── Global_Summary/       # Consolidated master_metrics.csv
│   ├── Unmixing/             # Extracted endmembers & abundances (.npy)
│   ├── Models/               # Trained model checkpoints (.keras)
│   └── Reports/              # Quantitative metrics per run (.json / .csv)
├── in_deep_analysis/         # In-deep spatial analysis & reviewer response
│   └── generate_in_deep_analysis.py # ROI views and paired reconstruction RMSE maps
├── local_global/             # Compact global-to-material comparison
│   ├── analyze_local_global.py      # Reproduce the figure and supporting CSV
│   └── local_global_summary.tex     # Figure environment and short explanation
├── main.tex                  # Draft manuscript integrating both analysis fragments
└── paper_figures/            # Paper figures, tables, and generator scripts
    ├── generate_paper_plots.py      # Efficiency scatter, heatmap, spectral comparison
    ├── generate_reconstructions.py  # Reconstruction error maps
    ├── table/
    │   └── table_metrics.tex        # LaTeX quantitative results table
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

For figure reproduction only, the lightweight dependencies are sufficient:
```bash
python -m pip install numpy scipy pandas matplotlib pyyaml
```
Use the full environment above for experiments; it additionally includes
TensorFlow, scikit-learn, pysptools, and Plotly for the interactive simplex views.

### 1. Generate Efficiency Frontier, Heatmaps & Spectral Comparisons
```bash
python paper_figures/generate_paper_plots.py
```
Outputs generated in `paper_figures/`:
* `frontera_eficiencia.png` — Reconstruction RMSE versus total execution time.
* `heatmap_rendimiento.png` — Reconstruction RMSE across all methods and datasets.
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

### 3. Generate Selected ROI Comparisons
```bash
python in_deep_analysis/generate_in_deep_analysis.py
```
Outputs generated in `in_deep_analysis/`:
* `fig_spatial_error_analysis.png` — Selected reconstruction error comparisons, with shared color limits within each dataset's pair.

The accompanying `spatial_error_discussion.tex` describes the figure and explains
the competitor selection. These are system comparisons, not architectural ablations.

### 4. Generate the Compact Local–Global Summary
```bash
python local_global/analyze_local_global.py
```
This creates `local_global_summary.png`, its LaTeX figure and short explanatory
paragraph, and the small supporting `material_results.csv`.

### 5. LaTeX Tables & Discussion
* `paper_figures/table/table_metrics.tex` — Main quantitative benchmark table.
* `in_deep_analysis/spatial_error_discussion.tex` — Discussion text and figure environment for spatial error analysis.
* `local_global/local_global_summary.tex` — Material-specific winners compared with scene-level winners.

`main.tex` includes both discussion fragments and reads figures from
`paper_figures/`. Run the generators before compiling. Generated PNGs stay local;
`paper_figures/Met.png` is tracked because it is a source diagram without a generator.

The manuscript is a draft: its referenced `sample.bib` must be supplied.
Throughout the paper, `rRMSE` means reconstruction RMSE, not relative RMSE.
Full-cube, material-region, and ROI values use the same root-mean-square
aggregation; `aRMSE` remains the separate abundance metric with its existing definition.

---

## 🚀 Running End-to-End Experiments from Scratch

To retrain the models and re-execute the entire unmixing pipeline:

```bash
python run_experiments.py
```

Hyperparameters, epochs, and target datasets can be configured in `config.yaml`.
