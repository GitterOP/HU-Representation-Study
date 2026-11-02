#!/usr/bin/env python3
"""Summarize global reconstruction, material-region RMSE, and endmember SAD."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
        rows.append(dict(dataset=dataset, winner=winner, matches=int(count), total=len(chosen),
                         spectral_winner=best_signatures,
                         mean_sad=float(part[part.method == best_signatures].sad_deg.mean())))

    colors = {'PCA': '#475569', 'AE': '#3b82f6', 'AE-Patch': '#06b6d4',
              'ViT': '#8b5cf6', 'MAE': '#ec4899', 'DINO': '#10b981'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    fig.subplots_adjust(left=0.24, right=0.72, bottom=0.27, top=0.83)
    for index, row in enumerate(rows):
        fraction = row['matches'] / row['total']
        ax.barh(index, fraction, height=0.5, color=colors[row['winner']])
        ax.text(max(fraction / 2, 0.03), index, '{matches}/{total}'.format(**row),
                ha='center', va='center', weight='bold', color='#0f172a')
        transform = ax.get_yaxis_transform()
        ax.scatter(1.06, index, s=45, color=colors[row['spectral_winner']],
                   transform=transform, clip_on=False)
        ax.text(1.10, index, '{spectral_winner} ({mean_sad:.2f}°)'.format(**row),
                va='center', transform=transform)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(['{dataset}\nGlobal winner: {winner}'.format(**row) for row in rows])
    ax.set_ylim(len(rows)-0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_xlabel('Share of material regions with lowest local RMSE', labelpad=9)
    ax.text(1.04, 1.12, 'Lowest mean SAD', weight='bold', transform=ax.transAxes)
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=0.2)
    ax.tick_params(axis='y', length=0, pad=12)
    for side in ('left', 'right', 'top'):
        ax.spines[side].set_visible(False)
    fig.text(0.24, 0.04, 'Counts refer to GT-dominant material regions, not pixel area.',
             fontsize=9, color='#475569')
    fig.savefig(output / 'local_global_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    agreements = ['{matches}/{total} in {dataset}'.format(**row) for row in rows]
    agreements = ', '.join(agreements[:-1]) + ', and ' + agreements[-1]
    differences = [row for row in rows if row['winner'] != row['spectral_winner']]
    if differences:
        example = ('In {dataset}, {spectral_winner} has the lowest mean endmember SAD, '
                   'despite {winner} achieving the lowest global RMSE. ').format(**differences[0])
    else:
        example = 'Global RMSE and mean SAD winners coincide across these scenes. '
    paragraph = ('Figure~\\ref{fig:local_global_summary} contrasts scene-level reconstruction '
                 'with material-level preservation. The fractions of material regions where '
                 'the global RMSE winner also minimizes local RMSE are ' + agreements + '. ' + example +
                 'Thus, global reconstruction quality alone does not establish uniform '
                 'preservation of every material region and endmember.')
    tex = r"""% Include from the project root; requires \usepackage{graphicx}.
% \input{local_global/local_global_summary.tex}
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{local_global/local_global_summary.png}
    \caption{Agreement between global reconstruction, local reconstruction, and endmember fidelity.
    Bar labels report matching regions out of all material regions; each region comprises
    pixels sharing the dominant ground-truth material. The right column identifies the
    method with the lowest mean endmember SAD.}
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
        'Las fracciones cuentan materiales, no píxeles; los empates locales cuentan como coincidencias. '
        'RMSE es convencional, aunque el resumen original lo denomina rRMSE. '
        'SAD compara forma espectral. La revisión previa de escalas del GT sigue pendiente.\n\n'
        'Dependencias: NumPy, pandas, SciPy, PyYAML y Matplotlib. '
        'El script verifica RMSE global y mSAD contra el resumen original.\n', encoding='utf-8')


if __name__ == '__main__':
    output = Path(__file__).resolve().parent
    results = analyze()
    results.to_csv(output / 'material_results.csv', index=False)
    write_outputs(results, output)
    print('Updated summary figure, LaTeX paragraph, and supporting CSV ({} rows).'.format(len(results)))
