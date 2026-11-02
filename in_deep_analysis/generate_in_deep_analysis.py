#!/usr/bin/env python3
"""
in_deep_analysis/plot_spatial_analysis.py

Generates the single, high-impact, publication-ready visual figure
    comparing selected ROI reconstruction errors across Samson, Jasper Ridge,
    and Urban.
"""

import os
import sys
import yaml
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import linear_sum_assignment

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300
})

DATASET_CONFIG = {
    'Samson': {
        'cfg_key': 'samson',
        'materials': ['Soil', 'Tree', 'Water'],
        'mat_colors': ['#d97706', '#16a34a', '#2563eb'], # Orange, Green, Blue
        'rgb_bands': (81, 48, 20),
        'crop_box': (20, 70, 20, 70), # y0, y1, x0, x1 (captures shore/water, canopy, and soil)
        'comp_model': 'PCA',
        'comp_label': 'PCA (Linear Baseline)'
    },
    'Jasper Ridge': {
        'cfg_key': 'jasper',
        'materials': ['Tree', 'Water', 'Soil', 'Road'],
        'mat_colors': ['#16a34a', '#2563eb', '#d97706', '#cbd5e1'], # Green, Blue, Orange, Light Gray (Road)
        'rgb_bands': (27, 15, 7),
        'crop_box': (25, 65, 25, 65), # y0, y1, x0, x1
        'comp_model': 'MAE',
        'comp_label': 'MAE (Pixel Reconstruction)'
    },
    'Urban': {
        'cfg_key': 'urban',
        'materials': ['Asphalt', 'Grass', 'Tree', 'Roof'],
        'mat_colors': ['#475569', '#eab308', '#16a34a', '#dc2626'], # Slate, Yellow (Pasto), Green (Árbol), Red
        'rgb_bands': (50, 27, 17),
        'crop_box': (140, 220, 80, 160), # y0, y1, x0, x1
        'comp_model': 'PCA',
        'comp_label': 'PCA (Linear Baseline)'
    }
}


def load_dataset(ds_name, cfg_path='config.yaml'):
    with open(os.path.join(PROJECT_ROOT, cfg_path), 'r') as f:
        config = yaml.safe_load(f)

    meta = DATASET_CONFIG[ds_name]
    ds_cfg = config['datasets'][meta['cfg_key']]

    rows, cols = ds_cfg['shape']['rows'], ds_cfg['shape']['cols']
    bands = ds_cfg['shape']['bands']

    img_mat = sio.loadmat(os.path.join(PROJECT_ROOT, ds_cfg['image_path']))
    Y_raw = img_mat[ds_cfg['mat_keys']['Y']].astype(np.float64)
    if Y_raw.ndim == 3:
        Y_raw = Y_raw.reshape(-1, bands).T
    y_max = float(Y_raw.max())
    Y = Y_raw / y_max if y_max > 1.0 else Y_raw.copy()
    original_cube = Y.T.reshape(rows, cols, bands)

    gt_mat = sio.loadmat(os.path.join(PROJECT_ROOT, ds_cfg['gt_path']))
    M_true = gt_mat[ds_cfg['mat_keys']['M']].astype(np.float64)
    A_true = gt_mat[ds_cfg['mat_keys']['A']].astype(np.float64)
    A_true_img = A_true.reshape(-1, rows, cols).transpose(1, 2, 0)

    # Pseudo-RGB
    rgb_idx = meta['rgb_bands']
    rgb_img = np.stack([
        original_cube[:, :, rgb_idx[0]],
        original_cube[:, :, rgb_idx[1]],
        original_cube[:, :, rgb_idx[2]]
    ], axis=-1)
    for b in range(3):
        p_low, p_high = np.percentile(rgb_img[:, :, b], (2, 98))
        if p_high > p_low:
            rgb_img[:, :, b] = np.clip((rgb_img[:, :, b] - p_low) / (p_high - p_low), 0, 1)

    return original_cube, M_true, A_true_img, rgb_img, rows, cols, bands, meta


def load_model_reconstruction_error(ds_name, method, original_cube):
    """
    Loads abundances and endmembers, computes linear cube reconstruction,
    and calculates conventional pixelwise spectral RMSE (not relative RMSE).
    """
    rows, cols, bands = original_cube.shape
    p_ab = os.path.join(PROJECT_ROOT, f'results_cl/Unmixing/{method}/{ds_name}/{ds_name}_FCLS_abundances.npy')
    p_em = os.path.join(PROJECT_ROOT, f'results_cl/Unmixing/{method}/{ds_name}/{ds_name}_FCLS_endmembers.npy')

    if not (os.path.exists(p_ab) and os.path.exists(p_em)):
        return None

    ab = np.load(p_ab)
    em = np.load(p_em)
    n_end = em.shape[0]

    reconstructed = np.dot(ab.reshape(-1, n_end), em).reshape(rows, cols, bands)
    rec_err_map = np.sqrt(np.mean((original_cube - reconstructed) ** 2, axis=2))
    return rec_err_map


def make_dominant_class_rgb(A_img, mat_colors):
    """
    Renders discrete dominant class thematic map: each pixel takes the solid,
    unblended color of the material endmember with the highest fractional abundance.
    """
    rows, cols, n_mat = A_img.shape
    dominant_idx = np.argmax(A_img, axis=-1)
    comp = np.zeros((rows, cols, 3), dtype=np.float64)
    for m in range(n_mat):
        hex_c = mat_colors[m].lstrip('#')
        rgb_val = np.array([int(hex_c[i:i+2], 16) / 255.0 for i in (0, 2, 4)])
        mask = (dominant_idx == m)
        comp[mask] = rgb_val
    return comp


def main():
    print("[*] Generating consolidated master visual figure: fig_spatial_error_analysis.png...")
    datasets = ['Samson', 'Jasper Ridge', 'Urban']

    # 3 rows x 4 columns:
    # Col 1: Scene Overview with ROI Box
    # Col 2: Ground Truth with Color Legend
    # Col 3: Competing Model Error Map (PCA / MAE)
    # Col 4: DINO Error Map (Same scale as Col 3)
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15.2, 11.2), facecolor='white')
    plt.subplots_adjust(hspace=0.28, wspace=0.22, left=0.04, right=0.96, top=0.95, bottom=0.04)

    row_letters = ['(a)', '(b)', '(c)']

    for row_idx, ds_name in enumerate(datasets):
        orig, M_true, A_true_img, rgb_img, rows, cols, bands, meta = load_dataset(ds_name)
        crop = meta['crop_box']
        y0, y1, x0, x1 = crop

        dino_err = load_model_reconstruction_error(ds_name, 'DINO', orig)
        comp_model = meta['comp_model']
        comp_err = load_model_reconstruction_error(ds_name, comp_model, orig)

        # Local crops of error maps
        comp_err_crop = comp_err[y0:y1, x0:x1]
        dino_err_crop = dino_err[y0:y1, x0:x1]
        vmax = max(float(np.max(comp_err_crop)), float(np.max(dino_err_crop)))

        # -------------------------------------------------------------
        # Col 1: Scene Overview with ROI Box
        # -------------------------------------------------------------
        ax1 = axes[row_idx, 0]
        ax1.imshow(rgb_img)
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2.0, edgecolor='#ef4444', facecolor='none')
        ax1.add_patch(rect)
        ax1.set_title(f"{row_letters[row_idx]}1. {ds_name} Scene\nROI [{y0}:{y1}, {x0}:{x1}]", fontweight='bold', pad=6)
        ax1.axis('off')

        # -------------------------------------------------------------
        # Col 2: Ground Truth Dominant Class + Color Legend
        # -------------------------------------------------------------
        ax2 = axes[row_idx, 1]
        gt_crop_rgb = make_dominant_class_rgb(A_true_img[y0:y1, x0:x1, :], meta['mat_colors'])
        ax2.imshow(gt_crop_rgb)
        ax2.set_title(f"{row_letters[row_idx]}2. Ground Truth (Dominant Class)", fontweight='bold', pad=6)
        ax2.axis('off')

        # Add explicit color legend below GT plot
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(facecolor=c, edgecolor='black', linewidth=0.6, label=m_name)
            for m_name, c in zip(meta['materials'], meta['mat_colors'])
        ]
        ax2.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.04),
            ncol=2 if len(meta['materials']) == 4 else 3,
            fontsize=8.5,
            frameon=True,
            facecolor='#f8fafc',
            edgecolor='#cbd5e1'
        )

        # -------------------------------------------------------------
        # Col 3: Competing Model Error Map (Zoom ROI)
        # -------------------------------------------------------------
        ax3 = axes[row_idx, 2]
        im3 = ax3.imshow(comp_err_crop, cmap='turbo', vmin=0, vmax=vmax)
        comp_mean = float(np.mean(comp_err_crop))
        ax3.set_title(f"{row_letters[row_idx]}3. {meta['comp_label']}\nMean pixel RMSE: {comp_mean:.3f}", fontweight='bold', pad=6)
        ax3.axis('off')
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.08)
        cbar3 = plt.colorbar(im3, cax=cax3)
        cbar3.set_label('Pixelwise RMSE', fontsize=8.5)
        cbar3.ax.tick_params(labelsize=8)

        # -------------------------------------------------------------
        # Col 4: DINO Error Map (Zoom ROI) - Same Scale
        # -------------------------------------------------------------
        ax4 = axes[row_idx, 3]
        im4 = ax4.imshow(dino_err_crop, cmap='turbo', vmin=0, vmax=vmax)
        dino_mean = float(np.mean(dino_err_crop))

        ax4.set_title(f"{row_letters[row_idx]}4. DINO\nMean pixel RMSE: {dino_mean:.3f}", fontweight='bold', color='#047857', pad=6)
        ax4.axis('off')
        divider4 = make_axes_locatable(ax4)
        cax4 = divider4.append_axes("right", size="5%", pad=0.08)
        cbar4 = plt.colorbar(im4, cax=cax4)
        cbar4.set_label('Pixelwise RMSE', fontsize=8.5)
        cbar4.ax.tick_params(labelsize=8)

    out_path = os.path.join(SCRIPT_DIR, 'fig_spatial_error_analysis.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Master figure saved successfully at: {out_path}")


if __name__ == '__main__':
    main()
