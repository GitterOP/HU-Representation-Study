import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------------------------------------------------------
# Configurar paths para ejecución flexible (desde raíz o dentro de graphs_paper)
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset_loaders_gt import GroundTruthDataset

# Paleta de colores oficial curada por modelo (idéntica al resto del paper)
COLORS = {
    'PCA': '#475569',        # Gris Pizarra / Slate
    'AE': '#3b82f6',         # Azul Eléctrico
    'AE_PATCH': '#06b6d4',   # Turquesa / Cian
    'VIT': '#8b5cf6',        # Violeta / Morado
    'MAE': '#ec4899',        # Rosa Fucsia
    'DINO': '#10b981'        # Verde Esmeralda
}

DATASETS = [
    {
        'csv_name': 'Jasper Ridge',
        'cfg_key': 'jasper',
        'out_filename': 'reconstruccion_jasper_ridge.png'
    },
    {
        'csv_name': 'Samson',
        'cfg_key': 'samson',
        'out_filename': 'reconstruccion_samson.png'
    },
    {
        'csv_name': 'Urban',
        'cfg_key': 'urban',
        'out_filename': 'reconstruccion_urban.png'
    }
]

METHODS = ['PCA', 'AE', 'AE_PATCH', 'VIT', 'MAE', 'DINO']
LETTERS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']


def load_metrics():
    """Carga métricas maestras si existen para asegurar consistencia exacta."""
    csv_path = os.path.join(PROJECT_ROOT, 'results_cl', 'Global_Summary', 'master_metrics.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include=['object', 'string']).columns:
            df[col] = df[col].astype(str).str.strip()
        df = df.drop_duplicates(subset=['Dataset', 'Method_DimRed'], keep='last')
        return df
    return None


def generate_reconstruction_maps():
    print("=" * 70)
    print("  GENERACIÓN DE MAPAS DE ERROR RMSE (ESCALA HOMOGÉNEA GLOBAL)")
    print("=" * 70)

    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    metrics_df = load_metrics()

    # -------------------------------------------------------------------------
    # FASE 1: Pre-calcular mapas de error y encontrar el máximo global absoluto
    # -------------------------------------------------------------------------
    print("\n[*] FASE 1: Calculando mapas de error para determinar el máximo global...")
    cached_data = {}
    global_max_error = 0.0

    for ds_info in DATASETS:
        csv_name = ds_info['csv_name']
        cfg_key = ds_info['cfg_key']
        cached_data[csv_name] = {}

        ds_cfg = config['datasets'][cfg_key]
        dataset_loader = GroundTruthDataset(ds_cfg)
        Y, _, _, meta = dataset_loader.load()
        rows, cols, bands = meta['shape']['rows'], meta['shape']['cols'], meta['shape']['bands']
        original = Y.T.reshape(rows, cols, bands)

        ds_max = 0.0
        for method in METHODS:
            k_options = ["", "k_1.0", "k_1", "k_1.00"]
            found_ab = None
            found_em = None

            for opt in k_options:
                sub_path = opt if opt else ""
                p_ab = os.path.join(PROJECT_ROOT, 'results_cl', 'Unmixing', method, csv_name, sub_path, f"{csv_name}_FCLS_abundances.npy")
                p_em = os.path.join(PROJECT_ROOT, 'results_cl', 'Unmixing', method, csv_name, sub_path, f"{csv_name}_FCLS_endmembers.npy")
                if os.path.exists(p_ab) and os.path.exists(p_em):
                    found_ab = p_ab
                    found_em = p_em
                    break

            if found_ab and found_em:
                abundances = np.load(found_ab)
                endmembers = np.load(found_em)
                n_end = endmembers.shape[0]

                reconstructed = np.dot(abundances.reshape(-1, n_end), endmembers).reshape(rows, cols, bands)
                error_map = np.sqrt(np.mean((original - reconstructed) ** 2, axis=2))

                rmse_val = None
                if metrics_df is not None:
                    row_match = metrics_df[(metrics_df['Dataset'] == csv_name) & (metrics_df['Method_DimRed'] == method)]
                    if not row_match.empty and 'rRMSE' in row_match.columns:
                        rmse_val = row_match['rRMSE'].values[0]

                if rmse_val is None:
                    rmse_val = np.sqrt(np.mean((original - reconstructed) ** 2))

                curr_max = float(np.max(error_map))
                ds_max = max(ds_max, curr_max)
                global_max_error = max(global_max_error, curr_max)

                cached_data[csv_name][method] = {
                    'error_map': error_map,
                    'rmse_val': rmse_val,
                    'max_err': curr_max
                }
            else:
                cached_data[csv_name][method] = None

        print(f"    - {csv_name}: Error máximo local = {ds_max:.4f}")

    print(f"\n[+] ESCALA HOMOGENIZADA: vmin = 0.0000 | vmax = {global_max_error:.4f} (en los 3 datasets)")

    # -------------------------------------------------------------------------
    # FASE 2: Generar figuras con escala unificada compartida
    # -------------------------------------------------------------------------
    print("\n[*] FASE 2: Renderizando figuras con escala unificada...")

    for ds_info in DATASETS:
        csv_name = ds_info['csv_name']
        out_name = ds_info['out_filename']
        out_path = os.path.join(SCRIPT_DIR, out_name)

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9.5), facecolor='white')
        axes = axes.flatten()

        for idx, method in enumerate(METHODS):
            ax = axes[idx]
            letter = LETTERS[idx]
            m_data = cached_data[csv_name].get(method)

            if m_data is not None:
                error_map = m_data['error_map']
                rmse_val = m_data['rmse_val']

                # Escala estricta compartida de 0 a global_max_error
                im = ax.imshow(error_map, cmap='turbo', vmin=0.0, vmax=global_max_error)
                ax.axis('off')
                ax.set_title(
                    f"{letter} {method} (RMSE: {rmse_val:.4f})",
                    fontsize=12,
                    fontweight='bold',
                    color=COLORS[method],
                    pad=8
                )

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="4.5%", pad=0.08)
                cbar = fig.colorbar(im, cax=cax)
                cbar.ax.tick_params(labelsize=8)
                cbar.set_label('RMSE', fontsize=9, labelpad=4)
            else:
                ax.axis('off')
                ax.text(
                    0.5, 0.5,
                    f"{letter} {method}\n[Datos no disponibles]",
                    ha='center', va='center',
                    fontsize=11, color='#888888', fontweight='semibold'
                )
                ax.set_title(
                    f"{letter} {method}",
                    fontsize=12,
                    fontweight='bold',
                    color='#888888',
                    pad=8
                )

        plt.tight_layout(h_pad=2.2, w_pad=1.5)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[✓] Guardado con escala homogénea: {out_path}")

    print("\n" + "=" * 70)
    print("  ¡TODAS LAS FIGURAS 2x3 DE RECONSTRUCCIÓN GENERADAS EXITOSAMENTE!")
    print("=" * 70)


if __name__ == '__main__':
    generate_reconstruction_maps()
