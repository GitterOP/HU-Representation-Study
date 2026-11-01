#!/usr/bin/env python3
"""Connect existing global, spatial, abundance, and endmember results; no training."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.text import Text
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import yaml


METHODS = ['PCA', 'AE', 'AE_PATCH', 'VIT', 'MAE', 'DINO']
LABELS = dict(zip(METHODS, ['PCA', 'AE', 'AE-Patch', 'ViT', 'MAE', 'DINO']))
COLORS = dict(zip(METHODS, ['#475569', '#3b82f6', '#06b6d4', '#8b5cf6', '#ec4899', '#10b981']))
MATERIAL_COLORS = {'Soil': '#b45309', 'Tree': '#15803d', 'Water': '#2563eb',
                   'Road': '#6b7280', 'Asphalt': '#475569', 'Grass': '#84cc16', 'Roof': '#dc2626'}
DATASETS = {
    'samson': ('Samson', ['Soil', 'Tree', 'Water']),
    'jasper': ('Jasper Ridge', ['Tree', 'Water', 'Soil', 'Road']),
    'urban': ('Urban', ['Asphalt', 'Grass', 'Tree', 'Roof']),
}
REGION_LABELS = {
    'all': 'Whole scene', 'boundary_r1': 'Boundary band', 'mixed_t080': 'Mixed pixels',
    'critical': 'Boundary or mixed', 'pure_interior': 'Pure interior',
    'low_signal': 'Low-signal pixels',
}


def slug(name):
    return name.lower().replace(' ', '_')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_scene(root, cfg):
    rows, cols, bands = (cfg['shape'][key] for key in ('rows', 'cols', 'bands'))
    pixels, q = rows * cols, cfg['shape']['materials']
    raw = np.asarray(loadmat(root / cfg['image_path'])[cfg['mat_keys']['Y']], dtype=float)
    if raw.ndim == 3:
        raw = raw.reshape(pixels, bands).T
    require(raw.shape == (bands, pixels), 'Unexpected hyperspectral data shape')
    divisor = float(raw.max()) if raw.max() > 1 else 1.0
    spectra = (raw / divisor).T.copy()
    gt = loadmat(root / cfg['gt_path'])
    signatures = np.asarray(gt[cfg['mat_keys']['M']], dtype=float).T.copy()
    abundances = np.asarray(gt[cfg['mat_keys']['A']], dtype=float).T.copy()
    require(signatures.shape == (q, bands), 'Unexpected reference endmember shape')
    require(abundances.shape == (pixels, q), 'Unexpected reference abundance shape')
    for values in (spectra, signatures, abundances):
        require(np.isfinite(values).all(), 'Non-finite ground-truth data')
    require(np.allclose(abundances.sum(1), 1, atol=1e-8), 'GT abundances do not sum to one')
    require(abundances.min() >= -1e-8, 'Negative GT abundance')
    return spectra, signatures, abundances, (rows, cols), divisor


def make_masks(gt_abundances, spectra, shape, materials):
    dominant = gt_abundances.argmax(1).reshape(shape)
    max_abundance = gt_abundances.max(1).reshape(shape)
    boundary = np.zeros(shape, dtype=bool)
    vertical = dominant[1:] != dominant[:-1]
    horizontal = dominant[:, 1:] != dominant[:, :-1]
    boundary[1:] |= vertical
    boundary[:-1] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    masks = {'all': np.ones(shape, dtype=bool)}
    for radius in (0, 1, 2):
        masks['boundary_r{}'.format(radius)] = (
            boundary.copy() if radius == 0 else
            binary_dilation(boundary, structure=np.ones((3, 3)), iterations=radius)
        )
    for threshold in (0.7, 0.8, 0.9):
        masks['mixed_t{:03d}'.format(round(threshold * 100))] = max_abundance < threshold
    masks['critical'] = masks['boundary_r1'] | masks['mixed_t080']
    masks['noncritical'] = ~masks['critical']
    masks['pure_interior'] = ~masks['boundary_r1'] & (max_abundance >= 0.9)
    brightness = spectra.mean(1).reshape(shape)
    masks['low_signal'] = brightness <= np.quantile(brightness, 0.1)
    for index, material in enumerate(materials):
        masks['material_' + slug(material)] = dominant == index
    require(np.all(masks['critical'] | masks['noncritical']), 'Incomplete spatial partition')
    require(not np.any(masks['critical'] & masks['noncritical']), 'Overlapping partition')
    return masks, dominant, max_abundance


def find_arrays(root, dataset, method):
    base = root / 'results_cl' / 'Unmixing' / method / dataset
    for subdir in ('', 'k_1.0', 'k_1', 'k_1.00'):
        folder = base / subdir
        abundance = folder / (dataset + '_FCLS_abundances.npy')
        endmember = folder / (dataset + '_FCLS_endmembers.npy')
        if abundance.exists() and endmember.exists():
            return abundance, endmember
    raise FileNotFoundError('Missing saved arrays: {} / {}'.format(dataset, method))


def match_signatures(estimated, reference):
    dots = estimated @ reference.T
    norms = np.linalg.norm(estimated, axis=1)[:, None] * np.linalg.norm(reference, axis=1)[None, :]
    cost = np.degrees(np.arccos(np.clip(dots / (norms + 1e-10), -1, 1)))
    extracted, truth = linear_sum_assignment(cost)
    require(len(truth) == len(reference), 'Incomplete endmember matching')
    permutation = np.empty(len(reference), dtype=int)
    permutation[truth] = extracted
    require(len(np.unique(permutation)) == len(reference), 'Non-unique endmember matching')
    return permutation, cost[permutation, np.arange(len(reference))]


def reconstruct_errors(original, abundance, endmember):
    mse = np.empty(len(original))
    angular = np.empty(len(original))
    for start in range(0, len(original), 8192):
        stop = min(start + 8192, len(original))
        observed = original[start:stop]
        reconstructed = abundance[start:stop] @ endmember
        mse[start:stop] = np.mean((observed - reconstructed) ** 2, axis=1)
        onorm = np.linalg.norm(observed, axis=1)
        rnorm = np.linalg.norm(reconstructed, axis=1)
        onorm[onorm == 0] = 1e-10
        rnorm[rnorm == 0] = 1e-10
        angular[start:stop] = np.degrees(np.arccos(np.clip(
            np.sum(observed * reconstructed, axis=1) / (onorm * rnorm), -1, 1)))
    require(np.isfinite(mse).all() and np.isfinite(angular).all(), 'Invalid reconstruction errors')
    return mse, angular


def markdown_table(frame, columns, decimals=4):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                value = '{:.{precision}f}'.format(value, precision=decimals) if np.isfinite(value) else 'NA'
            cells.append(str(value))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def best_labels(frame, metric):
    best = frame[metric].min()
    return ' / '.join(frame.loc[np.isclose(frame[metric], best, rtol=1e-10, atol=1e-12), 'method'].tolist())


def summarize_rankings(global_df, region_df, endmember_df):
    ranks, correlations, comparisons = [], [], []
    targets = {
        'Global reconstruction RMSE': ('global', 'reconstruction_rmse'),
        'Boundary reconstruction RMSE': ('boundary_r1', 'reconstruction_rmse'),
        'Mixed-pixel reconstruction RMSE': ('mixed_t080', 'reconstruction_rmse'),
        'Critical-region reconstruction RMSE': ('critical', 'reconstruction_rmse'),
        'Macro material reconstruction RMSE': ('global', 'macro_material_rmse'),
        'Global abundance error (mean pixel RMSE)': ('global', 'abundance_mean_pixel_rmse'),
        'Boundary abundance error (mean pixel RMSE)': ('boundary_r1', 'abundance_mean_pixel_rmse'),
        'Critical-region abundance error (mean pixel RMSE)': ('critical', 'abundance_mean_pixel_rmse'),
        'Mean endmember SAD': ('global', 'mean_sad_deg'),
        'Worst endmember SAD': ('global', 'worst_sad_deg'),
        '95th-percentile reconstruction error': ('global', 'pixel_rmse_p95'),
    }
    for dataset in global_df['dataset'].unique():
        global_ds = global_df[global_df.dataset == dataset].set_index('method_key').reindex(METHODS)
        winner = global_ds.reconstruction_rmse.idxmin()
        for name, (region, metric) in targets.items():
            if region == 'global':
                values = global_ds[metric]
            else:
                part = region_df[(region_df.dataset == dataset) & (region_df.region == region)]
                values = part.set_index('method_key')[metric].reindex(METHODS)
            require(values.notna().all(), 'Empty primary comparison: {} {}'.format(dataset, name))
            rank = values.rank(method='average', ascending=True)
            best_value = float(values.min())
            best_methods = [LABELS[method] for method in METHODS
                            if np.isclose(values[method], best_value, rtol=1e-10, atol=1e-12)]
            for method in METHODS:
                ranks.append(dict(dataset=dataset, method_key=method, method=LABELS[method],
                                  metric=name, value=float(values[method]), rank=float(rank[method])))
            comparisons.append(dict(
                dataset=dataset, metric=name, global_winner=LABELS[winner],
                global_winner_rank=float(rank[winner]), global_winner_value=float(values[winner]),
                best_method=' / '.join(best_methods), best_value=best_value,
                global_winner_gap_percent=100 * (float(values[winner]) / best_value - 1) if best_value else np.nan,
                same_winner=bool(np.isclose(values[winner], best_value, rtol=1e-10, atol=1e-12)),
            ))
            if name != 'Global reconstruction RMSE':
                correlations.append(dict(dataset=dataset, comparison=name, n_methods=len(METHODS),
                    spearman_rho=float(spearmanr(global_ds.reconstruction_rmse, values)[0]),
                    interpretation='Descriptive within-dataset association; no significance claim'))
        for material in endmember_df[endmember_df.dataset == dataset].material.unique():
            part = endmember_df[(endmember_df.dataset == dataset) & (endmember_df.material == material)]
            part = part.set_index('method_key').reindex(METHODS)
            correlations.append(dict(dataset=dataset, comparison=material + ': SAD vs. material abundance RMSE',
                n_methods=len(METHODS), spearman_rho=float(spearmanr(part.sad_deg, part.material_abundance_rmse)[0]),
                interpretation='Descriptive within-material association; no significance claim'))
            for comparison, values in [
                ('global reconstruction RMSE vs. SAD', global_ds.reconstruction_rmse),
                ('material-region reconstruction RMSE vs. SAD', part.dominant_region_reconstruction_rmse),
            ]:
                correlations.append(dict(dataset=dataset, comparison=material + ': ' + comparison,
                    n_methods=len(METHODS), spearman_rho=float(spearmanr(values, part.sad_deg)[0]),
                    interpretation='Descriptive within-material association; no significance claim'))
    return pd.DataFrame(ranks), pd.DataFrame(correlations), pd.DataFrame(comparisons)


def configure_plots():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 10,
                         'axes.titlesize': 11, 'legend.fontsize': 8, 'savefig.dpi': 220})


def save_figure(fig, path, pdf=False):
    visible_text = '\n'.join(item.get_text() for item in fig.findobj(Text))
    require(not re.search(r'AE_PATCH|AE_Patch|\bVIT\b|\bVit\b', visible_text), 'Noncanonical method label')
    if path.name.startswith('spatial_errors_'):
        images = [image for axis in fig.axes for image in axis.get_images()]
        require(len(images) == 12, 'Expected two rows of six spatial error maps')
        for row in (images[:6], images[6:]):
            limits = row[0].get_clim()
            require(all(image.get_clim() == limits for image in row), 'Inconsistent color limits')
            for image in row:
                values = np.asarray(image.get_array())
                require(values.min() >= limits[0] and values.max() <= limits[1], 'Clipped spatial error map')
    fig.savefig(path.with_suffix('.png'), bbox_inches='tight')
    if pdf:
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_connections(global_df, region_df, endmember_df, rankings, cache, output):
    configure_plots()
    names = [item[0] for item in DATASETS.values()]
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for row, dataset in enumerate(names):
        global_ds = global_df[global_df.dataset == dataset].set_index('method_key')
        boundary = region_df[(region_df.dataset == dataset) & (region_df.region == 'boundary_r1')].set_index('method_key')
        for col, (values, label) in enumerate([
            (boundary.reconstruction_rmse, 'Boundary reconstruction RMSE'),
            (boundary.abundance_mean_pixel_rmse, 'Boundary abundance error'),
            (global_ds.mean_sad_deg, 'Mean endmember SAD (degrees)'),
        ]):
            ax = axes[row, col]
            for method in METHODS:
                x, y = global_ds.loc[method, 'reconstruction_rmse'], values[method]
                ax.scatter(x, y, color=COLORS[method], edgecolor='white', s=70, zorder=3, label=LABELS[method])
            rho = spearmanr(global_ds.reconstruction_rmse.reindex(METHODS), values.reindex(METHODS))[0]
            ax.set_title('{} | Spearman rho = {:.2f}'.format(dataset, rho))
            ax.set_xlabel('Global reconstruction RMSE')
            ax.set_ylabel(label)
            ax.margins(0.25)
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.035), ncol=6)
    save_figure(fig, output / 'global_local_endmember', pdf=True)

    selected = [
        'Global reconstruction RMSE', 'Boundary reconstruction RMSE',
        'Mixed-pixel reconstruction RMSE', 'Macro material reconstruction RMSE',
        'Global abundance error (mean pixel RMSE)',
        'Boundary abundance error (mean pixel RMSE)', 'Mean endmember SAD', 'Worst endmember SAD',
    ]
    labels = ['Global reconstruction', 'Boundary reconstruction', 'Mixed-pixel reconstruction',
              'Macro material reconstruction', 'Global abundance error', 'Boundary abundance error',
              'Mean endmember SAD', 'Worst endmember SAD']
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), constrained_layout=True)
    for ax, dataset in zip(axes, names):
        pivot = rankings[rankings.dataset == dataset].pivot(index='metric', columns='method_key', values='rank')
        values = pivot.reindex(index=selected, columns=METHODS).values
        im = ax.imshow(values, cmap='Blues_r', vmin=1, vmax=6, aspect='auto')
        ax.set_xticks(range(6)); ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=45, ha='right')
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels if ax is axes[0] else [''] * len(labels))
        ax.set_title(dataset)
        for row in range(len(labels)):
            for col in range(6):
                ax.text(col, row, '{:g}'.format(values[row, col]), ha='center', va='center',
                        color='white' if values[row, col] < 3.5 else '#0f172a')
    fig.colorbar(im, ax=axes, shrink=0.8, label='Rank (1 = lowest error)', ticks=range(1, 7))
    save_figure(fig, output / 'preservation_rankings', pdf=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    for ax, dataset in zip(axes, names):
        global_ds = global_df[global_df.dataset == dataset].set_index('method_key').reindex(METHODS)
        rows = [global_ds.reconstruction_rmse.rank().values]
        labels = ['Global reconstruction']
        for metric, prefix in [('dominant_region_reconstruction_rmse', 'Local RMSE: '), ('sad_deg', 'SAD: ')]:
            for material in cache[dataset]['materials']:
                values = endmember_df[(endmember_df.dataset == dataset) & (endmember_df.material == material)]
                rows.append(values.set_index('method_key').reindex(METHODS)[metric].rank().values)
                labels.append(prefix + material)
        values = np.asarray(rows)
        im = ax.imshow(values, cmap='Blues_r', vmin=1, vmax=6, aspect='auto')
        ax.set_xticks(range(6)); ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=45, ha='right')
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_title(dataset)
        for row in range(len(labels)):
            for col in range(6):
                ax.text(col, row, '{:g}'.format(values[row, col]), ha='center', va='center',
                        color='white' if values[row, col] < 3.5 else '#0f172a')
    fig.colorbar(im, ax=axes, shrink=0.8, label='Rank (1 = lowest error)', ticks=range(1, 7))
    save_figure(fig, output / 'material_preservation_rankings', pdf=True)

    spectral_max = max(float(item['pixel_rmse'].max()) for ds in cache.values() for item in ds['methods'].values())
    abundance_max = max(float(item['pixel_armse'].max()) for ds in cache.values() for item in ds['methods'].values())
    for dataset, data in cache.items():
        masks, materials = data['masks'], data['materials']
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
        im = axes[0].imshow(data['dominant'], cmap=ListedColormap([MATERIAL_COLORS[m] for m in materials]),
                            vmin=-0.5, vmax=len(materials)-0.5)
        bar = fig.colorbar(im, ax=axes[0], ticks=range(len(materials)), shrink=0.8)
        bar.ax.set_yticklabels(materials)
        axes[0].set_title('Dominant reference material')
        axes[1].imshow(masks['boundary_r1'], cmap='Greys', vmin=0, vmax=1)
        axes[1].set_title('Boundary band (one-pixel halo)')
        im = axes[2].imshow(data['max_abundance'], cmap='viridis', vmin=0, vmax=1)
        fig.colorbar(im, ax=axes[2], shrink=0.8, label='Maximum reference abundance')
        axes[2].set_title('Reference mixture structure')
        axes[3].imshow(masks['critical'], cmap='Greys', vmin=0, vmax=1)
        axes[3].set_title('Critical mask: boundary or mixed')
        for ax in axes:
            ax.axis('off')
        save_figure(fig, output / ('regions_' + slug(dataset)))

        fig, axes = plt.subplots(2, 6, figsize=(20, 7), constrained_layout=True)
        for col, method in enumerate(METHODS):
            item = data['methods'][method]
            im0 = axes[0, col].imshow(item['pixel_rmse'], cmap='turbo', vmin=0, vmax=spectral_max)
            im1 = axes[1, col].imshow(item['pixel_armse'], cmap='magma', vmin=0, vmax=abundance_max)
            axes[0, col].set_title('{}\nGlobal RMSE: {:.4f}'.format(LABELS[method], item['global_rmse']))
            axes[1, col].set_title('Mean abundance error: {:.4f}'.format(item['mean_armse']))
            axes[0, col].axis('off'); axes[1, col].axis('off')
        fig.colorbar(im0, ax=axes[0, :], shrink=0.8, label='Pixelwise reconstruction RMSE')
        fig.colorbar(im1, ax=axes[1, :], shrink=0.8, label='Pixelwise abundance RMSE')
        save_figure(fig, output / ('spatial_errors_' + slug(dataset)))

        fig, axes = plt.subplots(2, len(materials), figsize=(4 * len(materials), 6.5), constrained_layout=True)
        for index, material in enumerate(materials):
            reference = data['reference'][index]
            for row in range(2):
                ref = reference if row == 0 else reference / max(float(reference.max()), 1e-12)
                axes[row, index].plot(ref, color='black', linestyle='--', linewidth=2, label='Ground truth')
                for method in METHODS:
                    spectrum = data['methods'][method]['matched_endmembers'][index]
                    values = spectrum if row == 0 else spectrum / max(float(spectrum.max()), 1e-12)
                    axes[row, index].plot(values, color=COLORS[method], linewidth=1.1, label=LABELS[method])
                axes[row, index].grid(alpha=0.2)
                axes[row, index].set_xlabel('Spectral band index')
                if index == 0:
                    axes[row, index].set_ylabel('Spectral value' if row == 0 else 'Peak-normalized spectral value')
            axes[0, index].set_title(material)
        axes[0, 0].legend(fontsize=7)
        save_figure(fig, output / ('endmember_signatures_' + slug(dataset)))
    return {'reconstruction_color_limits': [0, spectral_max], 'abundance_color_limits': [0, abundance_max],
            'rank_color_limits': [1, 6], 'shared_across_methods_and_datasets': True}


def write_reports(output, global_df, region_df, endmember_df, correlations, comparisons, sensitivity, reference_consistency):
    lines = [
        '# ¿La mejor reconstrucción global preserva también regiones críticas y endmembers?',
        '', 'Análisis descriptivo de seis sistemas en tres datasets, a partir de los resultados guardados.',
        'Todos los criterios siguientes son errores: valores menores indican mejor resultado.', '',
        '## Resultado por dataset', '',
    ]
    english = ['# Draft discussion: connecting global, local, and endmember preservation', '',
        'We re-evaluated the saved outputs of all six systems using identical, method-independent '
        'reference masks. Boundary bands were obtained from four-neighbor changes in the dominant '
        'reference material followed by a one-pixel dilation. Mixed pixels had maximum reference '
        'abundance below 0.8. Endmembers were matched one-to-one by minimum total spectral angle, '
        'and the same permutation was used for abundance evaluation.', '']
    wanted = ['Global reconstruction RMSE', 'Boundary reconstruction RMSE',
              'Mixed-pixel reconstruction RMSE', 'Boundary abundance error (mean pixel RMSE)',
              'Mean endmember SAD', 'Worst endmember SAD']
    for dataset in global_df.dataset.unique():
        subset = comparisons[(comparisons.dataset == dataset) & comparisons.metric.isin(wanted)]
        global_best = subset[subset.metric == 'Global reconstruction RMSE'].iloc[0]
        lines += ['### ' + dataset, '',
                  'Mejor reconstrucción global: **{}**, RMSE = **{:.6f}**.'.format(global_best.best_method, global_best.best_value), '',
                  markdown_table(subset, ['metric', 'best_method', 'best_value', 'global_winner_rank', 'global_winner_gap_percent']), '',
                  '`global_winner_rank` es la posición del ganador global en cada criterio; '
                  '`global_winner_gap_percent` es su exceso de error respecto al mejor método de ese criterio.', '']
        boundary = subset[subset.metric == 'Boundary reconstruction RMSE'].iloc[0]
        abundance = subset[subset.metric == 'Boundary abundance error (mean pixel RMSE)'].iloc[0]
        sad = subset[subset.metric == 'Mean endmember SAD'].iloc[0]
        english += [
            'In {}, {} achieved the lowest global reconstruction RMSE ({:.4f}). '
            'The lowest boundary reconstruction error was obtained by {} ({:.4f}), '
            'whereas boundary abundance error was lowest for {} ({:.4f}). '
            'The best mean endmember SAD was obtained by {} ({:.2f} degrees); '
            'the global reconstruction winner ranked {:g} on this endmember criterion.'.format(
                dataset, global_best.best_method, global_best.best_value, boundary.best_method,
                boundary.best_value, abundance.best_method, abundance.best_value,
                sad.best_method, sad.best_value, sad.global_winner_rank), '']
    counts = comparisons[comparisons.metric.isin(wanted[1:])].groupby('metric').same_winner.agg(['sum', 'count']).reset_index()
    lines += ['## Coincidencia de ganadores', '', markdown_table(counts, ['metric', 'sum', 'count']), '',
              '`sum` cuenta los datasets en que el ganador global también minimiza el criterio local o espectral.', '',
              '## Asociación entre rankings (seis métodos por dataset)', '',
              'Spearman positivo indica que menor error global tiende a acompañar menor error en el otro criterio. '
              'La coincidencia exacta del ganador y la asociación del ranking son preguntas distintas.', '',
              markdown_table(correlations[~correlations.comparison.str.contains(':')],
                             ['dataset', 'comparison', 'spearman_rho']), '',
              '## Materiales: conectar cada firma con su región y su abundancia', '',
              'SAD corresponde a una firma de endmember de toda la escena. El error de abundancia del material '
              'se mide en todos los píxeles; el RMSE local usa la región donde ese material domina según el GT.', '']
    material_rows, regional_rows = [], []
    material_wins = dict(reconstruction=0, signature=0, abundance=0)
    for (dataset, material), group in endmember_df.groupby(['dataset', 'material'], sort=False):
        best_global = global_df[global_df.dataset == dataset].sort_values('reconstruction_rmse').iloc[0].method_key
        global_row = group[group.method_key == best_global].iloc[0]
        best_sad = group.loc[group.sad_deg.idxmin()]
        best_ab = group.loc[group.material_abundance_rmse.idxmin()]
        best_local = group.loc[group.dominant_region_reconstruction_rmse.idxmin()]
        for key, metric in [('reconstruction', 'dominant_region_reconstruction_rmse'),
                            ('signature', 'sad_deg'), ('abundance', 'material_abundance_rmse')]:
            material_wins[key] += int(np.isclose(global_row[metric], group[metric].min(), rtol=1e-10, atol=1e-12))
        regional_rows.append(dict(dataset=dataset, material=material, pixels=int(global_row.dominant_pixels),
            global_winner=LABELS[best_global], global_winner_local_RMSE=global_row.dominant_region_reconstruction_rmse,
            best_local_method=best_labels(group, 'dominant_region_reconstruction_rmse'),
            best_local_RMSE=best_local.dominant_region_reconstruction_rmse,
            global_winner_gap_percent=100*(global_row.dominant_region_reconstruction_rmse/best_local.dominant_region_reconstruction_rmse-1)))
        material_rows.append(dict(dataset=dataset, material=material, global_winner=LABELS[best_global],
            global_winner_SAD=global_row.sad_deg, best_SAD_method=best_labels(group, 'sad_deg'),
            best_SAD=best_sad.sad_deg, global_winner_abundance_RMSE=global_row.material_abundance_rmse,
            best_abundance_method=best_labels(group, 'material_abundance_rmse'), best_abundance_RMSE=best_ab.material_abundance_rmse))
    material_summary = pd.DataFrame(material_rows)
    regional_summary = pd.DataFrame(regional_rows)
    material_summary.to_csv(output / 'material_connections.csv', index=False)
    regional_summary.to_csv(output / 'material_region_comparisons.csv', index=False)
    coverage_rows = []
    for dataset in global_df.dataset.unique():
        method = global_df[global_df.dataset == dataset].sort_values('reconstruction_rmse').iloc[0].method
        for name in ('boundary_r1', 'critical'):
            selected = region_df[(region_df.dataset == dataset) & (region_df.method == method) & (region_df.region == name)].iloc[0]
            coverage_rows.append(dict(dataset=dataset, method=method, region=name,
                scene_percent=100*selected.fraction, squared_error_percent=100*selected.reconstruction_sse_share,
                error_enrichment=selected.reconstruction_error_enrichment))
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output / 'winner_error_localization.csv', index=False)
    lines += [markdown_table(material_summary, list(material_summary.columns)), '',
              '### Excepciones en regiones dominadas por un material', '',
              'Un buen resultado en la máscara conjunta de bordes o mezcla no implica ganar en cada material. '
              'Los siguientes casos tienen menor RMSE local con otro método que con el ganador global:', '',
              markdown_table(regional_summary[regional_summary.global_winner_gap_percent > 1e-8],
                  list(regional_summary.columns)), '',
              '## Regiones y sensibilidad', '',
              'Las máscaras se fijaron con referencia, sin seleccionar zonas a partir del error de un método. '
              'Se evalúan también materiales minoritarios sin exigir un tamaño mínimo de ROI.', '',
              markdown_table(sensitivity, ['dataset', 'region', 'pixels', 'best_reconstruction_method',
                                           'best_abundance_method', 'global_winner_reconstruction_rank',
                                           'global_winner_abundance_rank']), '',
              '### Cobertura de las máscaras y contribución al error del ganador global', '',
              markdown_table(coverage, list(coverage.columns)), '',
              'Las máscaras agregadas pueden abarcar una fracción amplia de la escena; su concordancia '
              'con el ranking global no sustituye el análisis por material. Un `error_enrichment` mayor '
              'que uno indica mayor error cuadrático medio dentro de la máscara que en el conjunto de la '
              'escena. Una gran contribución total al error no implica enriquecimiento si la máscara '
              'también ocupa una gran superficie. No se debe afirmar que los errores siempre se '
              'concentran en bordes: esa afirmación depende del método y de la escena.', '',
              '## Alcance y precauciones de interpretación', '',
              '- Una reconstrucción espectral pequeña puede coexistir con firmas o abundancias incorrectas; '
              'las métricas se comparan directamente y no se asume equivalencia entre ellas.',
              '- Las correlaciones describen seis sistemas por escena. No hay repeticiones independientes '
              'ni pruebas de significación; los píxeles no se tratan como réplicas independientes.',
              '- Las máscaras se pueden solapar. Sus porcentajes de error cuadrático no deben sumarse. '
              'Solo `critical` y `noncritical` constituyen una partición disjunta.',
              '- El decil de señal baja es un proxy de brillo, no una anotación de sombra. '
              'Los bordes de etiquetas dominantes no abarcan todas las transiciones continuas.',
              '- Estas asociaciones no identifican atención, contexto espacial ni autodestilación como causas. '
              'Harían falta ablaciones controladas para esa atribución.',
              '- El GT de abundancias y firmas se usa como referencia del benchmark; '
              'no se demuestra una verdad física independiente mediante estas comparaciones.',
              '- Los resultados son comparaciones de sistemas completos dentro de cada dataset. '
              'Las imágenes se normalizan por escena; no se interpretan RMSE entre escenas como unidades físicas comunes.', '',
              '## Auditoría de métricas y archivos', '',
              'La columna histórica `rRMSE` del proyecto coincide con el RMSE convencional del cubo normalizado. '
              'Se exportan `reconstruction_rmse` y `relative_rmse` (cociente de normas) por separado. '
              'En cada dataset su denominador es constante entre métodos, por lo que sus rankings coinciden.', '',
              'La columna histórica `aRMSE` es la **media de los RMSE por píxel**, '
              'no la raíz del error cuadrático medio agrupado. Ambos se exportan y no se intercambian.', '',
              '### Consistencia de las referencias con el cubo observado', '',
              'Se calculó también la reconstrucción usando directamente las abundancias y firmas GT, '
              'sin ajuste ni recalibración. Esta es una comprobación del modelo de referencia bajo '
              'las mismas escalas que carga el proyecto, no un método competidor:', '',
              markdown_table(reference_consistency, ['dataset', 'reference_mixture_rmse',
                  'reference_mixture_relative_rmse', 'reference_mixture_sam_deg']), '',
              '**El producto de firmas y abundancias GT no reproduce exactamente el cubo observado.** '
              'En particular, el residual de Samson es grande frente a los residuos de los sistemas '
              'evaluados. Este diagnóstico no identifica su causa; no se han cambiado las escalas ni '
              'se ha supuesto una calibración radiométrica común entre firmas y cubo. Por tanto, los '
              'desacuerdos entre RMSE de reconstrucción, SAD y abundancias se interpretan como '
              'desacuerdos entre los objetivos de este benchmark, sin atribuirlos exclusivamente '
              'a pérdida de información del modelo. SAD compara forma y no valida por sí solo amplitud.', '',
              'Consultar `metric_audit.csv`, `region_definitions.csv`, `rank_correlations.csv`, '
              '`endmember_metrics.csv` y `provenance.json` para reproducir cada afirmación.', '']
    regional_total = len(regional_summary)
    boundary_wins = int(comparisons[comparisons.metric == 'Boundary reconstruction RMSE'].same_winner.sum())
    mixed_wins = int(comparisons[comparisons.metric == 'Mixed-pixel reconstruction RMSE'].same_winner.sum())
    mean_sad_wins = int(comparisons[comparisons.metric == 'Mean endmember SAD'].same_winner.sum())
    number_datasets = int(global_df.dataset.nunique())
    jasper_strict = sensitivity[(sensitivity.dataset == 'Jasper Ridge') & (sensitivity.region == 'mixed_t070')].iloc[0]
    jasper_road = regional_summary[(regional_summary.dataset == 'Jasper Ridge') & (regional_summary.material == 'Road')].iloc[0]
    urban_roof = regional_summary[(regional_summary.dataset == 'Urban') & (regional_summary.material == 'Roof')].iloc[0]
    samson_sad = comparisons[(comparisons.dataset == 'Samson') & (comparisons.metric == 'Mean endmember SAD')].iloc[0]
    summary = [
        '## Respuesta a la pregunta', '',
        '**Hay concordancia para regiones agregadas, pero no una equivalencia general entre '
        'reconstrucción global, conservación local de cada material y endmembers.**', '',
        '- El ganador global también gana en reconstrucción de bordes en {}/{} escenas y de píxeles '
        'mezclados en {}/{} con las máscaras principales.'.format(boundary_wins, number_datasets, mixed_wins, number_datasets),
        '- Al separar regiones por material dominante, ese ganador solo minimiza el RMSE local '
        'en {}/{} combinaciones material-escena; las otras {} tienen otro ganador.'.format(
            material_wins['reconstruction'], regional_total, regional_total-material_wins['reconstruction']),
        '- El menor RMSE global coincide con el menor mSAD en {}/{} escenas y con el menor SAD '
        'de cada material en {}/{} combinaciones.'.format(mean_sad_wins, number_datasets,
            material_wins['signature'], regional_total),
        '- El ganador global minimiza el RMSE de abundancia de cada material en {}/{} combinaciones.'.format(
            material_wins['abundance'], regional_total),
        '- En Urban, aunque DINO gana ambos criterios agregados, los rankings de reconstrucción '
        'global y abundancia en bordes presentan rho = {:.4f}; coincidir en el primer puesto '
        'no implica concordancia del orden completo.'.format(float(correlations[
            (correlations.dataset == 'Urban') &
            (correlations.comparison == 'Boundary abundance error (mean pixel RMSE)')].iloc[0].spearman_rho)),
        '- La sensibilidad también importa: revisar `sensitivity.csv`; en Jasper Ridge, con '
        'mezcla más estricta (máxima abundancia < 0.7), gana {} y el ganador global queda '
        'en posición {:g}.'.format(jasper_strict.best_reconstruction_method, jasper_strict.global_winner_reconstruction_rank),
        '', 'Los conteos por material son descriptivos y no representan muestras independientes.', '',
    ]
    lines[4:4] = [''] + summary
    english += [
        'The global reconstruction winner also minimized boundary and mixed-pixel reconstruction '
        'error in {}/{} and {}/{} scenes, respectively, under the primary masks. '
        'However, it minimized reconstruction error in only {}/{} dominant-material regions, '
        'individual endmember SAD in {}/{} material-scene combinations, and individual '
        'material abundance RMSE in {}/{} combinations. These are descriptive counts, '
        'not independent statistical replicates.'.format(boundary_wins, number_datasets, mixed_wins,
            number_datasets, material_wins['reconstruction'], regional_total, material_wins['signature'],
            regional_total, material_wins['abundance'], regional_total), '',
        'For example, {} had the lowest global reconstruction error in Jasper Ridge, while {} '
        'had the lowest reconstruction error within road-dominant pixels ({:.4f} versus {:.4f}). '
        'In Urban, the global winner was {}, while {} had the lowest reconstruction error '
        'in roof-dominant pixels ({:.4f} versus {:.4f}). In Samson, {} led global reconstruction, '
        'while {} had the lowest mean endmember SAD. Thus, scene-level criteria can mask '
        'material-specific differences. Under the stricter mixed-pixel sensitivity mask '
        'in Jasper Ridge, the best local reconstruction was obtained by {}.'.format(
            jasper_road.global_winner, jasper_road.best_local_method, jasper_road.best_local_RMSE,
            jasper_road.global_winner_local_RMSE, urban_roof.global_winner, urban_roof.best_local_method,
            urban_roof.best_local_RMSE, urban_roof.global_winner_local_RMSE,
            samson_sad.global_winner, samson_sad.best_method, jasper_strict.best_reconstruction_method), '',
        'These comparisons distinguish agreement in reconstruction rankings from preservation of '
        'material signatures and abundances. A low global residual is not by itself a sufficient '
        'validation of endmember or abundance accuracy. Region-specific and material-specific '
        'metrics should therefore accompany global reconstruction measures. All reconstruction '
        'maps share fixed color limits across systems, and abundance maps use a separate shared scale.', '',
        'The analysis is descriptive: each scene contains six evaluated systems, and the spatial '
        'pixels are not independent experimental replicates. Boundary widths and mixture thresholds '
        'were varied as sensitivity checks. These observations do not isolate architectural or '
        'training mechanisms. Low-signal masks are brightness proxies and cannot establish shadow.', '',
        'Metric clarification: the existing CSV field rRMSE contains conventional reconstruction '
        'RMSE, whereas relative RMSE is a separate norm ratio. The existing abundance aRMSE '
        'aggregates by averaging pixelwise RMSE rather than taking a pooled square root. '
        'The manuscript terminology and equations should be reconciled with these definitions.', '',
        'Reference consistency is an additional limitation. Under the scales loaded by the repository, '
        'the product of reference abundances and reference signatures does not exactly reproduce the '
        'observed cube. Its reconstruction RMSE is {:.4f} in Samson, {:.4f} in Jasper Ridge, and {:.4f} '
        'in Urban. No scale correction was fitted or applied. Consequently, disagreements among '
        'reconstruction, spectral-angle, and abundance criteria cannot be attributed solely to '
        'representation learning; they are conditional on the benchmark reference conventions. '
        'Spectral-angle fidelity does not by itself establish radiometric amplitude fidelity.'.format(
            *[float(reference_consistency[reference_consistency.dataset == name].iloc[0].reference_mixture_rmse)
              for name in ('Samson', 'Jasper Ridge', 'Urban')]), '']
    lines += ['Los empates en los rankings reciben el rango promedio; los ganadores empatados '
              'se indican juntos en `winner_agreement.csv`.', '']
    (output / 'analysis_report.md').write_text('\n'.join(lines), encoding='utf-8')
    (output / 'discussion_draft.md').write_text('\n'.join(english), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, output = args.project_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = root / 'config.yaml'
    config = yaml.safe_load(config_path.read_text())
    master_path = root / 'results_cl' / 'Global_Summary' / 'master_metrics.csv'
    master = pd.read_csv(master_path).drop_duplicates(['Dataset', 'Method_DimRed'], keep='last')
    inputs = {config_path, master_path}
    globals_out, regions_out, endmembers_out, audits, definitions, sensitivities = [], [], [], [], [], []
    reference_diagnostics = []
    cache = {}
    for key, (dataset, materials) in DATASETS.items():
        cfg = config['datasets'][key]
        inputs.update([root / cfg['image_path'], root / cfg['gt_path']])
        original, reference, gt_ab, shape, divisor = load_scene(root, cfg)
        masks, dominant, max_abundance = make_masks(gt_ab, original, shape, materials)
        np.savez_compressed(output / ('reference_masks_' + slug(dataset) + '.npz'), **masks)
        signal_rms = float(np.sqrt(np.mean(original ** 2)))
        reference_mse, reference_sam = reconstruct_errors(original, gt_ab, reference)
        reference_rmse = float(np.sqrt(reference_mse.mean()))
        reference_diagnostics.append(dict(dataset=dataset, reference_mixture_rmse=reference_rmse,
            reference_mixture_relative_rmse=reference_rmse/signal_rms,
            reference_mixture_sam_deg=float(reference_sam.mean()),
            image_normalization_divisor=divisor, reference_signatures_rescaled=False))
        cache[dataset] = dict(materials=materials, masks=masks, dominant=dominant,
                              max_abundance=max_abundance, reference=reference, methods={})
        for name, mask in masks.items():
            definitions.append(dict(dataset=dataset, region=name, pixels=int(mask.sum()),
                fraction=float(mask.mean()), depends_on_method=False,
                is_primary=name in ('all', 'boundary_r1', 'mixed_t080', 'critical', 'pure_interior')))
        for method in METHODS:
            abundance_path, endmember_path = find_arrays(root, dataset, method)
            inputs.update([abundance_path, endmember_path])
            raw_abundance = np.asarray(np.load(abundance_path), dtype=float)
            require(raw_abundance.shape in (shape + (len(reference),), (len(original), len(reference))),
                    'Estimated abundance shape mismatch')
            abundance = raw_abundance.reshape(len(original), len(reference))
            estimated = np.asarray(np.load(endmember_path), dtype=float)
            require(estimated.shape == reference.shape, 'Estimated endmember shape mismatch')
            require(np.isfinite(abundance).all() and np.isfinite(estimated).all(), 'Invalid saved outputs')
            require(abundance.min() >= -1e-8 and np.allclose(abundance.sum(1), 1, atol=1e-6), 'Abundance constraints failed')
            permutation, sad = match_signatures(estimated, reference)
            matched = abundance[:, permutation]
            abundance_mse = np.mean((matched - gt_ab) ** 2, axis=1)
            pixel_armse = np.sqrt(abundance_mse)
            spectral_mse, angular = reconstruct_errors(original, abundance, estimated)
            pixel_rmse = np.sqrt(spectral_mse)
            rmse = float(np.sqrt(spectral_mse.mean()))
            macro = np.mean([np.sqrt(spectral_mse[dominant.ravel() == index].mean()) for index in range(len(materials))])
            result = dict(dataset=dataset, method_key=method, method=LABELS[method],
                reconstruction_rmse=rmse, relative_rmse=rmse / signal_rms,
                sam_deg=float(angular.mean()), mean_sad_deg=float(sad.mean()), worst_sad_deg=float(sad.max()),
                abundance_mean_pixel_rmse=float(pixel_armse.mean()), abundance_pooled_rmse=float(np.sqrt(abundance_mse.mean())),
                macro_material_rmse=float(macro), pixel_rmse_p95=float(np.quantile(pixel_rmse, .95)),
                signal_rms=signal_rms, image_normalization_divisor=divisor)
            globals_out.append(result)
            source = master[(master.Dataset == dataset) & (master.Method_DimRed == method)]
            require(len(source) == 1, 'Ambiguous summary row')
            for field, computed in [('rRMSE', rmse), ('SAM', result['sam_deg']),
                                    ('mSAD', result['mean_sad_deg']), ('aRMSE', result['abundance_mean_pixel_rmse'])]:
                expected = float(source.iloc[0][field])
                passed = bool(np.isclose(computed, expected, rtol=1e-6, atol=1e-7))
                audits.append(dict(dataset=dataset, method=LABELS[method], field=field, saved=expected,
                                   recomputed=computed, absolute_difference=abs(computed-expected), passed=passed))
                require(passed, 'Source metric mismatch: {} {} {}'.format(dataset, method, field))
            for name, mask in masks.items():
                flat = mask.ravel()
                count = int(flat.sum())
                regions_out.append(dict(dataset=dataset, method_key=method, method=LABELS[method], region=name,
                    pixels=count, fraction=count / len(original),
                    reconstruction_rmse=float(np.sqrt(spectral_mse[flat].mean())) if count else np.nan,
                    pixel_rmse_mean=float(pixel_rmse[flat].mean()) if count else np.nan,
                    pixel_rmse_p95=float(np.quantile(pixel_rmse[flat], .95)) if count else np.nan,
                    sam_deg=float(angular[flat].mean()) if count else np.nan,
                    abundance_mean_pixel_rmse=float(pixel_armse[flat].mean()) if count else np.nan,
                    abundance_pooled_rmse=float(np.sqrt(abundance_mse[flat].mean())) if count else np.nan,
                    reconstruction_sse_share=float(spectral_mse[flat].sum()/spectral_mse.sum()),
                    reconstruction_error_enrichment=float(spectral_mse[flat].mean()/spectral_mse.mean()) if count else np.nan))
            partition_sse = sum(spectral_mse[masks[name].ravel()].sum() for name in ('critical', 'noncritical'))
            require(np.isclose(partition_sse, spectral_mse.sum(), rtol=1e-12), 'SSE partition failed')
            require(np.isclose(rmse ** 2, np.mean(pixel_rmse ** 2), rtol=1e-12), 'RMSE decomposition failed')
            for index, material in enumerate(materials):
                dominant_mask = dominant.ravel() == index
                weights = gt_ab[:, index]
                shape_ref = reference[index] / max(reference[index].max(), 1e-12)
                shape_est = estimated[permutation[index]] / max(estimated[permutation[index]].max(), 1e-12)
                endmembers_out.append(dict(dataset=dataset, method_key=method, method=LABELS[method],
                    material=material, reference_index=index, estimated_index=int(permutation[index]),
                    sad_deg=float(sad[index]), peak_normalized_signature_rmse=float(np.sqrt(np.mean((shape_ref-shape_est)**2))),
                    dominant_pixels=int(dominant_mask.sum()), dominant_fraction=float(dominant_mask.mean()),
                    mean_reference_abundance=float(weights.mean()),
                    dominant_region_reconstruction_rmse=float(np.sqrt(spectral_mse[dominant_mask].mean())),
                    abundance_weighted_reconstruction_rmse=float(np.sqrt(np.sum(weights*spectral_mse)/weights.sum())),
                    material_abundance_rmse=float(np.sqrt(np.mean((matched[:, index]-weights)**2))),
                    dominant_region_material_abundance_rmse=float(np.sqrt(np.mean((matched[dominant_mask, index]-weights[dominant_mask])**2)))))
            cache[dataset]['methods'][method] = dict(pixel_rmse=pixel_rmse.reshape(shape),
                pixel_armse=pixel_armse.reshape(shape), matched_endmembers=estimated[permutation],
                global_rmse=rmse, mean_armse=result['abundance_mean_pixel_rmse'])
            print('{} / {}: RMSE={:.6f}, mSAD={:.3f}, mean-pixel aRMSE={:.6f}'.format(
                dataset, LABELS[method], rmse, sad.mean(), pixel_armse.mean()), flush=True)
    global_df, region_df, endmember_df = map(pd.DataFrame, (globals_out, regions_out, endmembers_out))
    rankings, correlations, comparisons = summarize_rankings(global_df, region_df, endmember_df)
    for dataset in global_df.dataset.unique():
        global_winner = global_df[global_df.dataset == dataset].sort_values('reconstruction_rmse').iloc[0].method_key
        for name in ('boundary_r0', 'boundary_r1', 'boundary_r2', 'mixed_t070', 'mixed_t080', 'mixed_t090'):
            rows = region_df[(region_df.dataset == dataset) & (region_df.region == name)].set_index('method_key')
            require((rows.pixels > 0).all(), 'Empty sensitivity region')
            sensitivities.append(dict(dataset=dataset, region=name, pixels=int(rows.iloc[0].pixels),
                best_reconstruction_method=rows.loc[rows.reconstruction_rmse.idxmin(), 'method'],
                best_abundance_method=rows.loc[rows.abundance_mean_pixel_rmse.idxmin(), 'method'],
                global_winner_reconstruction_rank=float(rows.reconstruction_rmse.rank()[global_winner]),
                global_winner_abundance_rank=float(rows.abundance_mean_pixel_rmse.rank()[global_winner])))
    sensitivity_df = pd.DataFrame(sensitivities)
    reference_df = pd.DataFrame(reference_diagnostics)
    for filename, frame in [('global_metrics', global_df), ('region_metrics', region_df),
        ('endmember_metrics', endmember_df), ('rankings', rankings), ('rank_correlations', correlations),
        ('winner_agreement', comparisons), ('metric_audit', pd.DataFrame(audits)),
        ('region_definitions', pd.DataFrame(definitions)), ('sensitivity', sensitivity_df),
        ('reference_consistency', reference_df)]:
        frame.to_csv(output / (filename + '.csv'), index=False)
    scales = plot_connections(global_df, region_df, endmember_df, rankings, cache, output)
    write_reports(output, global_df, region_df, endmember_df, correlations, comparisons, sensitivity_df, reference_df)
    provenance = dict(
        analysis_script_sha256=sha256(Path(__file__)),
        input_files={str(path.relative_to(root)): sha256(path) for path in sorted(inputs)},
        source_metric_checks=len(audits), source_metric_checks_passed=all(item['passed'] for item in audits),
        figure_scales=scales,
        versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__),
        protocol='spec.md', no_training_or_parameter_selection=True,
    )
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Finished: {} source metric checks passed. Outputs: {}'.format(len(audits), output), flush=True)


if __name__ == '__main__':
    main()
