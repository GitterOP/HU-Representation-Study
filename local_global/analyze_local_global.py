#!/usr/bin/env python3
"""Summarize global reconstruction, material-region RMSE, and endmember SAD."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


def write_note(results, output):
    rows = []
    for dataset, part in results.groupby('dataset', sort=False):
        winner = part.groupby('method').global_rmse.first().idxmin()
        chosen = part[part.method == winner]
        best_local = part.groupby('material').local_rmse.min()
        count = sum(np.isclose(row.local_rmse, best_local[row.material], rtol=1e-10, atol=1e-12)
                    for row in chosen.itertuples())
        best_signatures = part.groupby('method').sad_deg.mean().idxmin()
        rows.append('| {} | {} | {} / {} | {} |'.format(dataset, winner, count, len(chosen), best_signatures))
    urban = results[results.dataset == 'Urban']
    global_winner = urban.groupby('method').global_rmse.first().idxmin()
    roof = urban[urban.material == 'Roof'].set_index('method')
    local_winner = roof.local_rmse.idxmin()
    if local_winner != global_winner:
        example = ('In Urban, {} achieves the lowest global reconstruction RMSE. However, {} '
                   'has lower RMSE in roof-dominant pixels ({:.4f} versus {:.4f}). The corresponding '
                   'Roof endmember SAD values are {:.2f} and {:.2f} degrees, respectively. '
                   'Thus, scene-level reconstruction quality does not guarantee the best '
                   'preservation of every material region or endmember.').format(
                       global_winner, local_winner, roof.loc[local_winner, 'local_rmse'],
                       roof.loc[global_winner, 'local_rmse'], roof.loc[local_winner, 'sad_deg'],
                       roof.loc[global_winner, 'sad_deg'])
    else:
        example = ('In Urban, {} has the lowest reconstruction RMSE both globally and in '
                   'roof-dominant pixels. The table distinguishes this local agreement from '
                   'mean endmember fidelity.').format(global_winner)
    note = '''# Relación entre reconstrucción global, regiones y endmembers

Este análisis responde una pregunta: ¿el ganador global también gana en cada
material y en la fidelidad media de las firmas? Se reutilizan los resultados
existentes, sin entrenar modelos ni generar figuras adicionales.

## Tabla propuesta para el paper

| Dataset | Lowest global RMSE | Material regions also won | Lowest mean SAD |
| --- | --- | --- | --- |
'''+ '\n'.join(rows) + '''

Cada región reúne todos los píxeles donde un material tiene la mayor abundancia
GT. La tercera columna cuenta materiales, no porcentaje de píxeles; los empates
locales también cuentan como coincidencias. La última columna compara el SAD
promedio de los endmembers, no exige ganar en cada firma individual.

## Párrafo ilustrativo para el paper

'''+ example + '''

## Reproducción y datos

Desde la raíz del repositorio: `python local_global/analyze_local_global.py`.
También funciona desde otra carpeta usando la ruta completa del script.
Dependencias: NumPy, pandas, SciPy y PyYAML del entorno del proyecto.

`material_results.csv` contiene los seis métodos para los 11 materiales de las
tres escenas (66 filas): dataset, método, material, número de píxeles, RMSE
global, RMSE local y SAD en grados. El RMSE global se repite en las filas de los
materiales del mismo método; no son observaciones globales independientes.

El script verifica RMSE global y mSAD frente al resumen original. Los mapas y
curvas de `paper_figures` aportan la evidencia visual; esta carpeta no los duplica.

## Notas de interpretación

- RMSE es el error convencional del cubo normalizado; el CSV original lo llama
  `rRMSE`, aunque no calcula el cociente de normas del RMSE relativo.
- SAD compara forma espectral. Una buena media no garantiza la mejor firma de
  cada material ni la preservación de su amplitud.
- Son comparaciones descriptivas de los sistemas guardados. La revisión previa
  de coherencia de escalas del GT sigue pendiente; simplificar no la corrige.
'''
    (output / 'README.md').write_text(note, encoding='utf-8')


if __name__ == '__main__':
    output = Path(__file__).resolve().parent
    results = analyze()
    results.to_csv(output / 'material_results.csv', index=False)
    write_note(results, output)
    print('Updated material_results.csv ({} rows) and README.md.'.format(len(results)))
