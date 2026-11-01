import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .loaders import DATASET_CONFIG, DATASET_REGISTRY

from .plot_simplex_3d import plot_latent_simplex


def generate_spectral_comparison(extracted, matched_results, ds_name, method, out_dir='Results', M_true=None):
    """
    Genera un grafico comparativo de las firmas espectrales (Extraidas vs Referencia) y lo guarda en PNG/.
    """
    if M_true is not None:
        references = M_true.T
    else:
        ref_path = os.path.join('Spectral_signatures', ds_name, f'signatures_{ds_name.lower()}.npy')
        if not os.path.exists(ref_path):
            return
        references = np.load(ref_path).T
        
    n_pairs = len(matched_results)
    
    # Obtener nombres de clases y estadísticas
    ds_registry = DATASET_REGISTRY.get(ds_name, {})
    class_stats = ds_registry.get('class_stats', {})
    class_names = list(class_stats.keys())
    class_pixels = list(class_stats.values())
    total_pixels = sum(class_pixels)
    if total_pixels == 0:
        total_pixels = 1
    
    # Determinar layout del grid
    cols = 3
    rows = (n_pairs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    axes_flat = axes.flatten() if n_pairs > 1 else [axes]

    for i, res in enumerate(matched_results):
        ext_idx = res['Extracted_ID'] - 1
        ref_idx = res['Matched_Class_ID'] - 1
        
        # Obtener metadatos de la clase si existen
        c_name = class_names[ref_idx] if ref_idx < len(class_names) else f"Unknown"
        c_pixels = class_pixels[ref_idx] if ref_idx < len(class_pixels) else 0
        c_perc = (c_pixels / total_pixels) * 100
        
        ax = axes_flat[i]
        # Normalizar individualmente por su valor maximo para comparar formas en la misma escala
        ref_sig = references[ref_idx]
        ext_sig = extracted[ext_idx]
        ref_max = np.max(ref_sig) if np.max(ref_sig) > 0 else 1.0
        ext_max = np.max(ext_sig) if np.max(ext_sig) > 0 else 1.0
        
        ax.plot(ref_sig / ref_max, color='black', linestyle='--', linewidth=1.5, label='Referencia (GT)')
        ax.plot(ext_sig / ext_max, color='red', linewidth=1.5, alpha=0.8, label='Extraído (N-FINDR)')
        
        title = f"ID:{ref_idx + 1} - {c_name}\n({c_pixels} px, {c_perc:.1f}%) | SAM: {res['SAM_deg']:.2f}°"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Banda")
        ax.set_ylabel("Reflectancia Norm.")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()

    # Ocultar subplots vacíos
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    
    png_dir = os.path.join(out_dir, 'PNG')
    os.makedirs(png_dir, exist_ok=True)
    out_path = os.path.join(png_dir, f"{ds_name}_N-FINDR_signatures_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"       [OK] Gráfico de firmas guardado en: {out_path}")


def generate_visual_comparison(original, reconstructed, meta, method, out_dir='Results'):
    """
    Genera un dashboard FCC, reconstrucción y mapa de error y lo guarda en PNG/.
    """
    ds_name = meta.get('dataset', meta.get('name', 'Unknown'))
    
    # Buscar configuración coincidente de forma insensible a mayúsculas y espacios
    config_key = None
    for k in DATASET_CONFIG:
        if k.lower() == ds_name.lower() or k.lower().replace(" ", "") == ds_name.lower().replace(" ", ""):
            config_key = k
            break

    if config_key is None or 'rgb_bands' not in DATASET_CONFIG[config_key]:
        print(f"       [!] No hay configuración RGB para {ds_name}. Saltando visualización.")
        return

    rgb_indices = [b - 1 for b in DATASET_CONFIG[config_key]['rgb_bands']]

    # Calcular límites de contraste basados en la imagen original para consistencia
    rgb_orig_raw = original[:, :, rgb_indices]
    lows  = [np.percentile(rgb_orig_raw[:,:,i], 2) for i in range(3)]
    highs = [np.percentile(rgb_orig_raw[:,:,i], 98) for i in range(3)]

    def apply_contrast(cube, l_list, h_list):
        rgb = cube[:, :, rgb_indices].copy()
        for i in range(3):
            l, h = l_list[i], h_list[i]
            if h > l:
                rgb[:, :, i] = np.clip((rgb[:, :, i] - l) / (h - l), 0, 1)
            else:
                rgb[:, :, i] = np.clip(rgb[:, :, i], 0, 1)
        return rgb

    img_orig_rgb = apply_contrast(original, lows, highs)
    img_rec_rgb = apply_contrast(reconstructed, lows, highs)
    
    # Calcular mapa de error (RMSE por píxel colapsando bandas)
    error_map = np.sqrt(np.mean((original - reconstructed)**2, axis=2))

    # Ajustamos el figsize y wspace para que las imágenes estén más juntas
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), gridspec_kw={'wspace': 0.02})
    
    # 1. Original
    axes[0].imshow(img_orig_rgb)
    axes[0].set_title(f"Original FCC ({ds_name})\nBands {DATASET_CONFIG[config_key]['rgb_bands']}", fontsize=10)
    axes[0].axis('off')
    
    # 2. Reconstruida
    axes[1].imshow(img_rec_rgb)
    axes[1].set_title(f"Reconstruida FCC\nMétodo: {method}", fontsize=10)
    axes[1].axis('off')
    
    # 3. Mapa de Calor de Error
    im = axes[2].imshow(error_map, cmap='jet')
    axes[2].set_title("Mapa de Error (RMSE por píxel)\nAzul=Bajo, Rojo=Alto", fontsize=10)
    axes[2].axis('off')
    
    # Colorbar ajustada para no separar tanto el plot
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # Usamos subplots_adjust en lugar de tight_layout para mayor control
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    
    # Guardar
    png_dir = os.path.join(out_dir, 'PNG')
    os.makedirs(png_dir, exist_ok=True)
    out_path = os.path.join(png_dir, f"{ds_name}_{method}_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"       [OK] Dashboard visual guardado en: {out_path}")


def generate_simplex_plot(config, image, endmembers, reducer, M_true, method, dr_method, out_dir, A_true=None, abundances=None):
    """
    Genera un gráfico de dispersión en 2D de los píxeles proyectados, y
    las visualizaciones interactivas de Plotly en el espacio latente real (Z)
    y en el simplex de abundancias estándar (A).
    Guarda HTMLs en HTML/ y PNGs en PNG/.
    """
    print("[4.7/6] Generando visualización del simplex...")
    
    rows, cols, bands = image.shape
    data_2d = image.reshape(-1, bands)
    n_endmembers = endmembers.shape[0]
    n_comp = n_endmembers - 1  # Dimensión del espacio latente (2 para M=3, 3 para M=4)
    
    # Alinear la escala de amplitud de las firmas GT con las extraídas
    references_scaled = None
    if M_true is not None:
        references_scaled = M_true.T.copy()
        for j in range(references_scaled.shape[0]):
            ref_sig = references_scaled[j]
            best_sam = float('inf')
            best_ext_max = 1.0
            for i in range(endmembers.shape[0]):
                ext_sig = endmembers[i]
                dot = np.dot(ext_sig, ref_sig)
                norm = (np.linalg.norm(ext_sig) * np.linalg.norm(ref_sig)) + 1e-10
                sam = np.arccos(np.clip(dot / norm, -1.0, 1.0))
                if sam < best_sam:
                    best_sam = sam
                    best_ext_max = np.max(ext_sig)
            
            ref_max = np.max(ref_sig) if np.max(ref_sig) > 0 else 1.0
            references_scaled[j] = ref_sig * (best_ext_max / ref_max)

    z_pixels_full = None
    z_est_full = None
    z_true_vertices_full = None
    
    # Intentar proyectar con el reductor activo usando la dimensión completa
    if reducer is not None and hasattr(reducer, 'transform'):
        try:
            # Proyectar píxeles
            z_all = reducer.transform(data_2d)
            z_pixels_full = z_all[:, :n_comp]
            
            # Proyectar endmembers estimados
            z_est_all = reducer.transform(endmembers)
            z_est_full = z_est_all[:, :n_comp]
            
            # Proyectar endmembers de Ground Truth (escalados)
            if references_scaled is not None:
                z_true_all = reducer.transform(references_scaled)
                z_true_vertices_full = z_true_all[:, :n_comp]
        except Exception as e:
            print(f"       [!] Error al proyectar usando el reductor original ({e}). Usando PCA para visualización.")
            z_pixels_full = None
            
    # Si no hay reductor o falló la proyección, usar un PCA temporal de n_comp componentes
    if z_pixels_full is None:
        try:
            from sklearn.decomposition import PCA
            temp_pca = PCA(n_components=n_comp, whiten=True, random_state=42)
            z_pixels_full = temp_pca.fit_transform(data_2d)
            z_est_full = temp_pca.transform(endmembers)
            if references_scaled is not None:
                z_true_vertices_full = temp_pca.transform(references_scaled)
        except Exception as e:
            print(f"       [!] Error fatal en la proyección del simplex: {e}")
            return

    # Guardar espacio latente completo para auditoría
    latent_dir = out_dir.replace('Reports', 'Latent')
    os.makedirs(latent_dir, exist_ok=True)
    np.save(os.path.join(latent_dir, 'latent_Z.npy'), z_pixels_full)
    np.save(os.path.join(latent_dir, 'latent_Z_est.npy'), z_est_full)
    if z_true_vertices_full is not None:
        np.save(os.path.join(latent_dir, 'latent_Z_true.npy'), z_true_vertices_full)
        
    # --- Generar los gráficos interactivos HTML de Plotly en el espacio latente y de abundancias ---
    try:
        ds_name = config['dataset']['name'].lower()
        if 'samson' in ds_name:
            ds_name = 'samson'
        elif 'jasper' in ds_name:
            ds_name = 'jasper'
        elif 'urban' in ds_name:
            ds_name = 'urban'
        elif 'apex' in ds_name:
            ds_name = 'apex'
            
        DATASET_CLASSES = {
            "samson": ["Soil", "Tree", "Water"],
            "jasper": ["Tree", "Water", "Soil", "Road"],
            "urban": ["Asphalt", "Grass", "Tree", "Roof"],
            "apex": ["Material 1", "Material 2", "Material 3", "Material 4"]
        }
        if ds_name in DATASET_CLASSES:
            class_names = DATASET_CLASSES[ds_name]
            if len(class_names) < n_endmembers:
                class_names = class_names + [f"Material {i+1}" for i in range(len(class_names), n_endmembers)]
            else:
                class_names = class_names[:n_endmembers]
        else:
            class_names = [f"Material {i+1}" for i in range(n_endmembers)]
            
        if n_endmembers == 3:
            edges = [(0, 1), (1, 2), (2, 0)]
        elif n_endmembers == 4:
            edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        else:
            edges = []
            for i in range(n_endmembers):
                for j in range(i+1, n_endmembers):
                    edges.append((i, j))
                    
        k_neighbors = config.get('unmixing', {}).get('knn_k', 5)
        html_dir = os.path.join(out_dir, 'HTML')
        os.makedirs(html_dir, exist_ok=True)

        # 1. Gráficos en el Espacio Latente (Z): Se guardan usando dr_method
        z_true_pixels = None
        if A_true is not None and z_true_vertices_full is not None:
            if A_true.shape[0] == z_true_vertices_full.shape[0]:
                z_true_pixels = np.dot(z_true_vertices_full.T, A_true)
            else:
                print(f"       [WARNING] Dimensión de clases A_true {A_true.shape} no coincide con vertices GT {z_true_vertices_full.shape}. No se graficará GT en espacio latente.")
                
        z_pixels_plot = z_pixels_full.T  # (dim, N)

        latent_modes = {
            "basic": f"{config['dataset']['name']}_{dr_method}_simplex_3d.html"
        }

        for mode_name, filename in latent_modes.items():
            out_html_path = os.path.join(html_dir, filename)
            try:
                plot_latent_simplex(
                    z_pixels=z_pixels_plot,
                    vertices=z_est_full,
                    edges=edges,
                    class_names=class_names,
                    method_name=dr_method,
                    ds_name=config['dataset']['name'],
                    out_html=out_html_path,
                    mode=mode_name,
                    z_true=z_true_pixels,
                    k_neighbors=k_neighbors,
                    vertices_true=z_true_vertices_full
                )
            except Exception as ex:
                print(f"       [!] Error al generar gráfico HTML latente '{mode_name}': {ex}")

        # 2. Gráficos de Abundancias (A): Se proyectan en el simplex estándar y se guardan usando method (ej. FCLS)
        if abundances is not None:
            # Definir coordenadas geométricas del simplex estándar
            if n_endmembers == 3:
                vertices_std = np.array([
                    [0.0, 1.0],
                    [-np.sqrt(3)/2, -0.5],
                    [np.sqrt(3)/2, -0.5]
                ])
            elif n_endmembers == 4:
                vertices_std = np.array([
                    [1.0,  1.0,  1.0],
                    [1.0, -1.0, -1.0],
                    [-1.0,  1.0, -1.0],
                    [-1.0, -1.0,  1.0]
                ])
            else:
                np.random.seed(42)
                vertices_std = np.random.randn(n_endmembers, 3)
                norms = np.linalg.norm(vertices_std, axis=1, keepdims=True)
                vertices_std = vertices_std / (norms + 1e-10)

            # abundances has shape (rows, cols, n_endmembers) -> reshape a (N, n_endmembers) y transpose a (n_endmembers, N)
            A_flat = abundances.reshape(-1, n_endmembers).T
            z_pixels_abund = np.dot(vertices_std.T, A_flat)
            
            z_true_abund = None
            if A_true is not None:
                # A_true has shape (n_endmembers, N)
                if A_true.shape[0] == n_endmembers:
                    z_true_abund = np.dot(vertices_std.T, A_true)

            abund_modes = {
                "compare": f"{config['dataset']['name']}_{method}_compare.html",
                "error_heatmap": f"{config['dataset']['name']}_{method}_error_heatmap.html"
            }

            for mode_name, filename in abund_modes.items():
                out_html_path = os.path.join(html_dir, filename)
                try:
                    plot_latent_simplex(
                        z_pixels=z_pixels_abund,
                        vertices=vertices_std,
                        edges=edges,
                        class_names=class_names,
                        method_name=method,
                        ds_name=config['dataset']['name'],
                        out_html=out_html_path,
                        mode=mode_name,
                        z_true=z_true_abund,
                        k_neighbors=k_neighbors,
                        vertices_true=None
                    )
                except Exception as ex:
                    print(f"       [!] Error al generar gráfico HTML de abundancias '{mode_name}': {ex}")

    except Exception as e:
        print(f"       [!] Error al importar o llamar a plot_latent_simplex: {e}")

    # Squeeze/slice a 2D para el gráfico estático de Matplotlib
    z_pixels = z_pixels_full[:, :2]
    z_est = z_est_full[:, :2]
    z_true = z_true_vertices_full[:, :2] if z_true_vertices_full is not None else None
    
    # Graficar
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Submuestrear los píxeles para el scatter plot si son demasiados
    n_pixels = z_pixels.shape[0]
    if n_pixels > 5000:
        np.random.seed(42)
        sub_idx = np.random.choice(n_pixels, 5000, replace=False)
        ax.scatter(z_pixels[sub_idx, 0], z_pixels[sub_idx, 1], c='lightgray', s=3, alpha=0.5, label='Píxeles (submuestreo)')
    else:
        ax.scatter(z_pixels[:, 0], z_pixels[:, 1], c='lightgray', s=3, alpha=0.5, label='Píxeles')
        
    # Graficar simplex estimado
    ax.scatter(z_est[:, 0], z_est[:, 1], c='red', s=120, marker='^', edgecolor='black', zorder=5, label='Endmembers Est. (N-FINDR)')
    if len(z_est) >= 3:
        centroid = np.mean(z_est, axis=0)
        angles = np.arctan2(z_est[:, 1] - centroid[1], z_est[:, 0] - centroid[0])
        sort_idx = np.argsort(angles)
        z_est_sorted = z_est[sort_idx]
        polygon = plt.Polygon(z_est_sorted, fill=False, edgecolor='red', linewidth=2.5, linestyle='-', label='Simplex Est.')
        ax.add_patch(polygon)
    elif len(z_est) == 2:
        ax.plot(z_est[:, 0], z_est[:, 1], color='red', linewidth=2.5, linestyle='-')
        
    # Graficar simplex real (Ground Truth)
    if z_true is not None:
        ax.scatter(z_true[:, 0], z_true[:, 1], c='blue', s=120, marker='o', edgecolor='black', zorder=4, label='Endmembers GT (Real)')
        if len(z_true) >= 3:
            centroid_true = np.mean(z_true, axis=0)
            angles_true = np.arctan2(z_true[:, 1] - centroid_true[1], z_true[:, 0] - centroid_true[0])
            sort_idx_true = np.argsort(angles_true)
            z_true_sorted = z_true[sort_idx_true]
            polygon_true = plt.Polygon(z_true_sorted, fill=False, edgecolor='blue', linewidth=2.0, linestyle='--', label='Simplex GT')
            ax.add_patch(polygon_true)
        elif len(z_true) == 2:
            ax.plot(z_true[:, 0], z_true[:, 1], color='blue', linewidth=2.0, linestyle='--')
            
    # Etiquetas para los endmembers
    for idx, pt in enumerate(z_est):
        ax.annotate(f"E{idx+1}", (pt[0], pt[1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold', color='darkred')
    if z_true is not None:
        for idx, pt in enumerate(z_true):
            ax.annotate(f"GT{idx+1}", (pt[0], pt[1]), textcoords="offset points", xytext=(0,-18), ha='center', fontsize=9, fontweight='bold', color='darkblue')
            
    ax.set_title(f"Visualización de Simplex (Proyección 2D)\nDataset: {config['dataset']['name']} | Reducción: {dr_method} | Extractor: N-FINDR", fontsize=11)
    ax.set_xlabel("Componente Proyectada 1")
    ax.set_ylabel("Componente Proyectada 2")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Guardar en PNG/ subfolder
    png_dir = os.path.join(out_dir, 'PNG')
    os.makedirs(png_dir, exist_ok=True)
    out_path = os.path.join(png_dir, f"{config['dataset']['name']}_N-FINDR_simplex.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"       [OK] Gráfico del simplex guardado en: {out_path}")
