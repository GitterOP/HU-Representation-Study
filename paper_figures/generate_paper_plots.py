import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment

# -----------------------------------------------------------------------------
# Paths y configuración de entorno
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset_loaders_gt import GroundTruthDataset

# Estilo de visualización limpio para publicación IEEE
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8
})

COLORS = {
    'PCA': '#475569',        # Gris Pizarra / Slate
    'AE': '#3b82f6',         # Azul Eléctrico
    'AE_PATCH': '#06b6d4',   # Turquesa / Cian
    'VIT': '#8b5cf6',        # Violeta / Morado
    'MAE': '#ec4899',        # Rosa Fucsia
    'DINO': '#10b981'        # Verde Esmeralda
}

LINE_STYLES = {
    'PCA': '-',
    'AE': '-',
    'AE_PATCH': '--',
    'VIT': '-.',
    'MAE': ':',
    'DINO': '-'
}

METHOD_LABELS = {
    'PCA': 'PCA',
    'AE': 'AE',
    'AE_PATCH': 'AE_PATCH',
    'VIT': 'ViT',
    'MAE': 'MAE',
    'DINO': 'DINO'
}

MATERIAL_NAMES = {
    'Samson': ['Soil', 'Tree', 'Water'],
    'Urban': ['Asphalt', 'Grass', 'Tree', 'Roof'],
    'Jasper Ridge': ['Tree', 'Water', 'Soil', 'Road']
}

CSV_TO_CONFIG_KEY = {
    'Samson': 'samson',
    'Jasper Ridge': 'jasper',
    'Urban': 'urban'
}

DATASET_ORDER = ['Samson', 'Jasper Ridge', 'Urban']
METHODS_ORDER = ['PCA', 'AE', 'AE_PATCH', 'VIT', 'MAE', 'DINO']


def load_data():
    csv_path = os.path.join(PROJECT_ROOT, 'results_cl', 'Global_Summary', 'master_metrics.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] No se encontró el archivo de métricas globales en '{csv_path}'.")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip()
    df = df.drop_duplicates(subset=['Dataset', 'Method_DimRed'], keep='last')
    return df


def generate_heatmap(df):
    """Genera heatmap_rendimiento.png sin título y con etiquetas formales en inglés."""
    print("[*] Generando heatmap_rendimiento.png...")
    pivot_df = df.pivot_table(index='Method_DimRed', columns='Dataset', values='rRMSE', aggfunc='mean')
    pivot_df = pivot_df.reindex([m for m in METHODS_ORDER if m in pivot_df.index])
    pivot_df = pivot_df[[d for d in DATASET_ORDER if d in pivot_df.columns]]

    fig, ax = plt.subplots(figsize=(6.2, 5.0), facecolor='white')
    im = ax.imshow(pivot_df.values, cmap='Blues_r', aspect='auto')

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns, fontweight='semibold')
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in pivot_df.index], fontweight='semibold')

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('rRMSE', rotation=-90, va="bottom", fontweight='semibold')
    cbar.ax.tick_params(labelsize=8.5)

    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.values[i, j]
            text_color = 'white' if val < pivot_df.values.mean() else '#0f172a'
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=text_color, fontweight='bold', fontsize=10.5)

    # Sin título (para el paper IEEE)
    ax.grid(False)
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, 'heatmap_rendimiento.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Guardado: {out_path}")


def generate_efficiency_frontier(df):
    """Genera frontera_eficiencia.png sin título y con etiquetas formales en inglés."""
    print("[*] Generando frontera_eficiencia.png...")
    summary = df.groupby(['Dataset', 'Method_DimRed']).agg({'rRMSE': 'mean', 'Time_Total': 'mean'}).reset_index()

    fig, ax = plt.subplots(figsize=(8.0, 5.5), facecolor='white')
    markers = {'Samson': 'o', 'Jasper Ridge': '^', 'Urban': 's'}

    for dataset in DATASET_ORDER:
        df_ds = summary[summary['Dataset'] == dataset]
        marker = markers.get(dataset, 'o')

        for _, row in df_ds.iterrows():
            method = row['Method_DimRed']
            color = COLORS.get(method, '#333333')
            disp_label = METHOD_LABELS.get(method, method)

            ax.scatter(
                row['Time_Total'], row['rRMSE'],
                s=120,
                color=color,
                marker=marker,
                edgecolors='black',
                linewidths=0.7,
                alpha=0.9
            )

            ax.annotate(
                disp_label,
                (row['Time_Total'], row['rRMSE']),
                textcoords="offset points",
                xytext=(0, 6),
                ha='center',
                fontsize=8,
                alpha=0.85
            )

    method_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[m], markersize=8.5, label=METHOD_LABELS.get(m, m))
        for m in METHODS_ORDER if m in summary['Method_DimRed'].unique()
    ]

    dataset_legend = [
        Line2D([0], [0], marker=markers[ds], color='w', markerfacecolor='#94a3b8', markersize=8.5, label=ds, markeredgecolor='black')
        for ds in DATASET_ORDER if ds in summary['Dataset'].unique()
    ]

    leg1 = ax.legend(
        handles=method_legend,
        title="Method",
        loc="upper right",
        bbox_to_anchor=(1, 1),
        frameon=True,
        facecolor='white',
        edgecolor='#e2e8f0',
        title_fontsize='9.5'
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=dataset_legend,
        title="Dataset",
        loc="upper right",
        bbox_to_anchor=(1, 0.62),
        frameon=True,
        facecolor='white',
        edgecolor='#e2e8f0',
        title_fontsize='9.5'
    )

    ax.set_xscale('log')
    ax.set_xlabel('Total Execution Time (s)', fontweight='semibold')
    ax.set_ylabel('rRMSE', fontweight='semibold')

    # Sin título (para el paper IEEE)
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, 'frontera_eficiencia.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Guardado: {out_path}")


def generate_endmember_plots(config):
    """Genera las 3 figuras de firmas espectrales vs Ground Truth sin suptitle y con ejes en inglés."""
    print("[*] Generando figuras de Endmembers vs Ground Truth...")

    for csv_ds_name, cfg_key in CSV_TO_CONFIG_KEY.items():
        try:
            ds_cfg = config['datasets'][cfg_key]
            dataset_loader = GroundTruthDataset(ds_cfg)
            _, M_true, _, meta = dataset_loader.load()

            references = M_true.T
            n_ref = references.shape[0]
            mat_names = MATERIAL_NAMES.get(csv_ds_name, [f"Material {idx+1}" for idx in range(n_ref)])

            fig, axes = plt.subplots(1, n_ref, figsize=(4.2 * n_ref, 3.8), facecolor='white')
            if n_ref == 1:
                axes = [axes]

            # Graficar Ground Truth de referencia
            for j in range(n_ref):
                ref_sig = references[j]
                ref_max = np.max(ref_sig) if np.max(ref_sig) > 0 else 1.0
                axes[j].plot(ref_sig / ref_max, label='Ground Truth', color='#0f172a', linestyle='--', linewidth=2.0, zorder=5)

            for method in METHODS_ORDER:
                color = COLORS[method]
                disp_label = METHOD_LABELS.get(method, method)

                k_str_options = ["", "k_1.0", "k_1", "k_1.00"]
                found_path = None
                for opt in k_str_options:
                    sub_p = opt if opt else ""
                    p = os.path.join(PROJECT_ROOT, 'results_cl', 'Unmixing', method, csv_ds_name, sub_p, f"{csv_ds_name}_FCLS_endmembers.npy")
                    if os.path.exists(p):
                        found_path = p
                        break

                if not found_path:
                    continue

                extracted_endmembers = np.load(found_path)
                n_extracted = extracted_endmembers.shape[0]

                cost_matrix = np.zeros((n_extracted, n_ref))
                for i in range(n_extracted):
                    for j in range(n_ref):
                        dot = np.dot(extracted_endmembers[i], references[j])
                        norm = (np.linalg.norm(extracted_endmembers[i]) * np.linalg.norm(references[j])) + 1e-10
                        cost_matrix[i, j] = np.arccos(np.clip(dot / norm, -1.0, 1.0))

                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for i, j in zip(row_ind, col_ind):
                    ext_sig = extracted_endmembers[i]
                    ext_max = np.max(ext_sig) if np.max(ext_sig) > 0 else 1.0

                    dot = np.dot(ext_sig, references[j])
                    norm = (np.linalg.norm(ext_sig) * np.linalg.norm(references[j])) + 1e-10
                    sad_deg = np.arccos(np.clip(dot / norm, -1.0, 1.0)) * (180 / np.pi)

                    linestyle = LINE_STYLES.get(method, '-')
                    axes[j].plot(
                        ext_sig / ext_max,
                        label=f"{disp_label} ({sad_deg:.1f}°)",
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.3,
                        alpha=0.85
                    )

            for j in range(n_ref):
                axes[j].set_title(mat_names[j], fontsize=11, fontweight='bold', pad=8)
                axes[j].set_xlabel('Spectral Band', fontsize=9.5)
                if j == 0:
                    axes[j].set_ylabel('Normalized Reflectance', fontsize=9.5)
                axes[j].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92, edgecolor='#e2e8f0', fontsize=8.0)
                axes[j].tick_params(axis='both', which='major', labelsize=8.5)
                axes[j].grid(True, alpha=0.25)

            # Sin suptitle (para el paper IEEE)
            plt.tight_layout()
            out_filename = f"08_endmembers_vs_gt_{cfg_key}.png"
            out_path = os.path.join(SCRIPT_DIR, out_filename)
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[✓] Guardado: {out_path}")

        except Exception as ex:
            print(f"    [!] Error en {csv_ds_name}: {ex}")
            import traceback
            traceback.print_exc()


def main():
    print("=" * 70)
    print("  ACTUALIZACIÓN DE GRÁFICOS PARA EL PAPER IEEE (SIN TÍTULOS, EJES LIMPIOS)")
    print("=" * 70)

    df = load_data()

    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Heatmap de rRMSE (sin título, cbar 'rRMSE', ejes limpios)
    generate_heatmap(df)

    # 2. Frontera de Eficiencia (sin título, 'Total Execution Time (s)', ejes en inglés)
    generate_efficiency_frontier(df)

    # 3. Endmembers vs Ground Truth por dataset (sin suptitle, ejes en inglés)
    generate_endmember_plots(config)

    print("\n" + "=" * 70)
    print("  ¡TODOS LOS GRÁFICOS DEL PAPER FUERON ACTUALIZADOS CON ÉXITO!")
    print("=" * 70)


if __name__ == '__main__':
    main()
