#!/usr/bin/env python3
"""Summarize global reconstruction, material-region rRMSE, and endmember SAD."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset_loaders_gt import GroundTruthDataset

METHODS = {'PCA': 'PCA', 'AE': 'AE', 'AE_PATCH': 'AE-Patch',
           'VIT': 'ViT', 'MAE': 'MAE', 'DINO': 'DINO'}
MATERIALS = {'samson': ['Soil', 'Tree', 'Water'],
             'jasper': ['Tree', 'Water', 'Soil', 'Road'],
             'urban': ['Asphalt', 'Grass', 'Tree', 'Roof']}


def analyze():
    config = yaml.safe_load((ROOT / 'config.yaml').read_text())
    summary = pd.read_csv(ROOT / 'results_cl/Global_Summary/master_metrics.csv')
    summary = summary.drop_duplicates(['Dataset', 'Method_DimRed'], keep='last')
    records = []
    for key, materials in MATERIALS.items():
        cfg = dict(config['datasets'][key])
        for field in ('image_path', 'gt_path'):
            cfg[field] = str(ROOT / cfg[field])
        spectra, reference, truth, _ = GroundTruthDataset(cfg).load()
        original, reference = spectra.T, reference.T
        dominant = truth.argmax(axis=0)
        dataset, q = cfg['name'], len(materials)
        for method, label in METHODS.items():
            folder = ROOT / 'results_cl/Unmixing' / method / dataset
            endmembers = np.load(folder / (dataset + '_FCLS_endmembers.npy'))
            abundance = np.load(folder / (dataset + '_FCLS_abundances.npy')).reshape(-1, q)
            if endmembers.shape != reference.shape or abundance.shape != (len(original), q):
                raise ValueError('Unexpected saved output shape: {} / {}'.format(dataset, label))
            mse = np.empty(len(original))
            for start in range(0, len(original), 8192):
                stop = start + 8192
                residual = original[start:stop] - abundance[start:stop] @ endmembers
                mse[start:stop] = np.mean(residual ** 2, axis=1)
            global_rmse = float(np.sqrt(mse.mean()))
            norms = np.linalg.norm(endmembers, axis=1)[:, None] * np.linalg.norm(reference, axis=1)[None, :]
            angles = np.degrees(np.arccos(np.clip((endmembers @ reference.T) / (norms + 1e-10), -1, 1)))
            extracted, matched = linear_sum_assignment(angles)
            sad = np.empty(q)
            sad[matched] = angles[extracted, matched]
            saved = summary[(summary.Dataset == dataset) & (summary.Method_DimRed == method)].iloc[0]
            if not np.allclose([global_rmse, sad.mean()], [saved.rRMSE, saved.mSAD], rtol=1e-6, atol=1e-7):
                raise ValueError('Source metric mismatch: {} / {}'.format(dataset, label))
            for index, material in enumerate(materials):
                mask = dominant == index
                if not mask.any():
                    raise ValueError('Empty material region: {} / {}'.format(dataset, material))
                records.append(dict(dataset=dataset, method=label, material=material, pixels=int(mask.sum()),
                                    global_rmse=global_rmse, local_rmse=float(np.sqrt(mse[mask].mean())),
                                    sad_deg=float(sad[index])))
    return pd.DataFrame(records)


def write_outputs(results, output):
    rows = []
    for dataset, part in results.groupby('dataset', sort=False):
        winner = part.groupby('method').global_rmse.first().idxmin()
        chosen = part[part.method == winner]
        best_local = part.groupby('material').local_rmse.min()
        count = sum(np.isclose(row.local_rmse, best_local[row.material], rtol=1e-10, atol=1e-12)
                    for row in chosen.itertuples())
        best_signatures = part.groupby('method').sad_deg.mean().idxmin()
        regions = []
        for material, values in part.groupby('material', sort=False):
            winners = values.loc[np.isclose(values.local_rmse, values.local_rmse.min(),
                                           rtol=1e-10, atol=1e-12), 'method'].tolist()
            regions.append(dict(material=material, methods=winners, label=' / '.join(winners),
                                matches_global=winner in winners))
        rows.append(dict(dataset=dataset, winner=winner, matches=int(count), total=len(chosen),
                         spectral_winner=best_signatures, regions=regions,
                         mean_sad=float(part[part.method == best_signatures].sad_deg.mean())))

    colors = {'PCA': '#475569', 'AE': '#3b82f6', 'AE-Patch': '#06b6d4',
              'ViT': '#8b5cf6', 'MAE': '#ec4899', 'DINO': '#10b981'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.96)
    ax.set_xlim(-2.25, 5.75)
    ax.set_ylim(-1.0, 2.95)
    ax.axis('off')
    ax.text(-0.25, 2.65, 'Scene / global rRMSE winner', ha='right', weight='bold', fontsize=9.5)
    ax.text(1.95, 2.65, 'Lowest local rRMSE by dominant material', ha='center', weight='bold', fontsize=9.5)
    ax.text(4.95, 2.65, 'Lowest mean SAD', ha='center', weight='bold', fontsize=9.5)
    for index, row in enumerate(rows):
        y = 2 - index
        ax.text(-0.25, y+0.17, row['dataset'], ha='right', va='center', weight='bold', fontsize=11)
        ax.text(-0.25, y-0.06, 'Global: ' + row['winner'], ha='right', va='center', fontsize=10)
        ax.text(-0.25, y-0.29, '{matches}/{total} local matches'.format(**row),
                ha='right', va='center', fontsize=8.5, color='#475569')
        for column, region in enumerate(row['regions']):
            color = matplotlib.colors.to_rgba(colors[region['methods'][0]], alpha=0.14)
            ax.add_patch(Rectangle((column, y-0.34), 0.92, 0.68, facecolor=color,
                edgecolor='#0f172a' if region['matches_global'] else '#cbd5e1',
                linewidth=2.2 if region['matches_global'] else 0.7))
            ax.text(column+0.46, y+0.13, region['material'], ha='center', va='center', fontsize=10)
            ax.text(column+0.46, y-0.12, region['label'], ha='center', va='center', fontsize=9.5,
                    weight='bold' if region['matches_global'] else 'normal', color='#0f172a')
        ax.scatter(4.43, y+0.10, s=40, color=colors[row['spectral_winner']])
        ax.text(4.60, y+0.10, row['spectral_winner'], va='center', fontsize=10.5)
        ax.text(4.60, y-0.16, '{mean_sad:.2f}°'.format(**row), va='center', fontsize=9.5, color='#475569')
    ax.add_patch(Rectangle((0, -0.85), 0.20, 0.16, fill=False, edgecolor='#0f172a', linewidth=2.2))
    ax.text(0.32, -0.77, 'Bold outline: regional winner matches the global rRMSE winner.',
            va='center', fontsize=9, color='#475569')
    fig.savefig(output / 'local_global_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    exceptions = [(row, region) for row in reversed(rows) for region in reversed(row['regions'])
                  if not region['matches_global']]
    local_example = ''
    if exceptions:
        row, region = exceptions[0]
        local_example = ('In {}, {} leads globally, whereas {} achieves the lowest local rRMSE '
                         'in {}-dominant pixels. ').format(
                             row['dataset'], row['winner'], region['label'], region['material'].lower())
    differences = [row for row in rows if row['winner'] != row['spectral_winner']]
    if differences:
        example = ('In {dataset}, {spectral_winner} has the lowest mean endmember SAD, '
                   'despite {winner} achieving the lowest global rRMSE. ').format(**differences[0])
    else:
        example = 'Global rRMSE and mean SAD winners coincide across these scenes. '
    paragraph = ('Figure~\\ref{fig:local_global_summary} contrasts scene-level reconstruction '
                 'with material-specific preservation. ' + local_example + example +
                 'Thus, global reconstruction quality alone does not establish uniform '
                 'preservation of every material region and endmember.')
    tex = r"""% Include from the project root; requires \usepackage{graphicx}.
% \input{local_global/local_global_summary.tex}
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{local_global/local_global_summary.png}
    \caption{Agreement between global reconstruction, local reconstruction, and endmember fidelity.
    Each block names a material and the method attaining the lowest rRMSE in its
    dominant ground-truth region. Bold outlines identify agreement with the global
    rRMSE winner. The prefix $r$ denotes reconstruction, not relative normalization. The right column reports the lowest mean endmember SAD.}
    \label{fig:local_global_summary}
\end{figure}

""" + paragraph + '\n'
    (output / 'local_global_summary.tex').write_text(tex, encoding='utf-8')
    tick = chr(96)
    (output / 'README.md').write_text(
        '# Resumen local–global\n\n'
        'Ejecutar: ' + tick + 'python local_global/analyze_local_global.py' + tick + '.\n\n'
        'Entregables: código, ' + tick + 'local_global_summary.png' + tick + ' y ' +
        tick + 'local_global_summary.tex' + tick + ' (figura y un párrafo corto en inglés). '
        'El CSV conserva los 66 resultados de respaldo.\n\n'
        'Para incluirlo desde la raíz del proyecto: ' + tick +
        '\\input{local_global/local_global_summary.tex}' + tick + '; requiere graphicx.\n\n'
        'Cada bloque identifica un material y su ganador local; el contorno grueso señala '
        'coincidencia con el ganador global. Los conteos se refieren a materiales, no píxeles; '
        'los empates locales también cuentan como coincidencias. '
        'rRMSE significa error de reconstrucción, no relativo; aRMSE se reserva para abundancias. '
        'SAD compara forma espectral. La revisión previa de escalas del GT sigue pendiente.\n\n'
        'Dependencias: NumPy, pandas, SciPy, PyYAML y Matplotlib. '
        'El script verifica rRMSE global y mSAD contra el resumen original.\n', encoding='utf-8')


if __name__ == '__main__':
    output = Path(__file__).resolve().parent
    results = analyze()
    results.to_csv(output / 'material_results.csv', index=False)
    write_outputs(results, output)
    print('Updated summary figure, LaTeX paragraph, and supporting CSV ({} rows).'.format(len(results)))
