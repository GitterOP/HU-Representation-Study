#!/usr/bin/env python
# Restored from GitterOP/EMVisual; upstream blob 7c4f02b49eadaf286940e2af7c2f7a671cd604b8.
"""
plot_simplex_3d.py

Genera visualizaciones interactivas (2D si M=3, 3D si M=4) del espacio latente
de abundancias. Admite los siguientes modos de comparación frente al Ground Truth:
- basic: Solo muestra la nube de abundancias estimadas.
- compare: Muestra la nube de abundancias estimadas y la de Ground Truth superpuestas.
- error_heatmap: Colorea los píxeles según el error (RMSE) entre estimado y real.
- vectors: Dibuja vectores de desplazamiento (error) desde la posición real a la estimada.
"""

import os
import argparse
import yaml
import numpy as np
# Option: lazy import plotly when actual visualization is needed
# import plotly.graph_objects as go
from scipy.io import loadmat

# Registro estático de clases como fallback
DATASET_CLASSES = {
    "samson": ["Soil", "Tree", "Water"],
    "jasper": ["Tree", "Water", "Soil", "Road"],
    "urban": ["Asphalt", "Grass", "Tree", "Roof"],
    "apex": ["Material 1", "Material 2", "Material 3", "Material 4"]
}


def load_abundance_data(file_path):
    """
    Carga la matriz de abundancias desde un archivo .npy o .mat.
    Retorna: (A, M, N)
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.npy':
        data = np.load(file_path)
    elif ext == '.mat':
        mat_dict = loadmat(file_path)
        keys = [k for k in mat_dict.keys() if not k.startswith('__')]
        if len(keys) == 0:
            raise KeyError(f"No se encontraron variables en el archivo .mat: {file_path}")
        data = mat_dict[keys[0]]
    else:
        raise ValueError(f"Formato no soportado: {ext}")

    if data.ndim == 3:
        rows, cols, M = data.shape
        A = data.reshape(-1, M).T
    elif data.ndim == 2:
        d1, d2 = data.shape
        if d1 < d2:
            A = data.astype(np.float64)
        else:
            A = data.T.astype(np.float64)
    else:
        raise ValueError(f"Dimensiones de datos inesperadas: {data.ndim}D")

    return A, A.shape[0], A.shape[1]


def load_ground_truth(dataset_name):
    """
    Lee config.yaml para localizar el archivo Ground Truth y carga la matriz A_true.
    Retorna: A_true (M, N) o None
    """
    if not os.path.exists("config.yaml"):
        print("       [WARNING] No se encontró config.yaml en el directorio raíz.")
        return None

    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        dataset_config = None
        for key, val in config.get("datasets", {}).items():
            if val.get("name", "").lower() == dataset_name.lower() or key.lower() == dataset_name.lower():
                dataset_config = val
                break

        if dataset_config is None:
            print(f"       [WARNING] No se encontró la configuración del dataset '{dataset_name}' en config.yaml.")
            return None

        gt_path = dataset_config.get("gt_path")
        if not gt_path or not os.path.exists(gt_path):
            print(f"       [WARNING] Archivo de Ground Truth no especificado o no encontrado: {gt_path}")
            return None

        a_key = dataset_config.get("mat_keys", {}).get("A")
        if not a_key:
            print("       [WARNING] Clave de abundancias 'A' no especificada en mat_keys de config.yaml.")
            return None

        # Cargar el archivo .mat de Ground Truth
        mat_dict = loadmat(gt_path)
        if a_key not in mat_dict:
            print(f"       [WARNING] Clave '{a_key}' no encontrada en el archivo GT: {gt_path}")
            return None

        A_true_raw = mat_dict[a_key]

        # Ajustar dimensiones de A_true
        if A_true_raw.ndim == 3:
            rows, cols, M = A_true_raw.shape
            A_true = A_true_raw.reshape(-1, M).T
        elif A_true_raw.ndim == 2:
            d1, d2 = A_true_raw.shape
            if d1 < d2:
                A_true = A_true_raw.astype(np.float64)
            else:
                A_true = A_true_raw.T.astype(np.float64)
        else:
            return None

        print(f"       [LOAD] Ground Truth cargado con éxito desde: {gt_path}")
        return A_true
    except Exception as e:
        print(f"       [WARNING] Error al cargar Ground Truth: {e}")
        return None


def project_to_simplex(A, vertices):
    """
    Proyecta abundancias de forma vectorizada: P = V.T * A
    """
    return np.dot(vertices.T, A)


def plot_latent_simplex(z_pixels, vertices, edges, class_names, method_name, ds_name, out_html, mode="basic", z_true=None, k_neighbors=5, vertices_true=None, col_ind=None):
    """
    Genera el HTML usando Plotly según el modo y dimensionalidad.
    """
    import plotly.graph_objects as go
    M = vertices.shape[0]
    dim = vertices.shape[1]
    N = z_pixels.shape[1]

    # Submuestrear para agilizar Plotly (10k píxeles max)
    if N > 10000:
        np.random.seed(42)
        sub_idx = np.random.choice(N, 10000, replace=False)
        plot_est = z_pixels[:, sub_idx]
        plot_true = z_true[:, sub_idx] if z_true is not None else None
        hover_text = [f"Píxel #{idx}" for idx in sub_idx]
    else:
        plot_est = z_pixels
        plot_true = z_true if z_true is not None else None
        hover_text = [f"Píxel #{idx}" for idx in range(N)]
        sub_idx = np.arange(N)

    # 1. Preparar trazas de Píxeles
    pixel_traces = []

    # Si requerimos Ground Truth pero no está disponible, forzamos modo "basic"
    effective_mode = mode
    if z_true is None and mode in ["compare", "error_heatmap", "vectors"]:
        effective_mode = "basic"

    if effective_mode == "basic":
        # Modo Básico: Solo estimado
        if dim == 2:
            pixel_trace = go.Scatter(
                x=plot_est[0, :], y=plot_est[1, :], mode='markers',
                marker=dict(size=3.5, color='rgb(100, 110, 140)', opacity=0.45),
                name='Abundancias Estimadas', hoverinfo='text', hovertext=hover_text
            )
        else:
            pixel_trace = go.Scatter3d(
                x=plot_est[0, :], y=plot_est[1, :], z=plot_est[2, :], mode='markers',
                marker=dict(size=1.5, color='rgb(100, 110, 140)', opacity=0.35),
                name='Abundancias Estimadas', hoverinfo='text', hovertext=hover_text
            )
        pixel_traces.append(pixel_trace)

    elif effective_mode == "compare":
        # Modo Comparar: Estimado (rojo/naranja) vs Real (azul/cyan)
        if dim == 2:
            trace_est = go.Scatter(
                x=plot_est[0, :], y=plot_est[1, :], mode='markers',
                marker=dict(size=3.5, color='rgb(220, 80, 80)', opacity=0.4),
                name='Estimadas (Modelo)', hoverinfo='text', hovertext=hover_text
            )
            trace_true = go.Scatter(
                x=plot_true[0, :], y=plot_true[1, :], mode='markers',
                marker=dict(size=3.5, color='rgb(60, 140, 220)', opacity=0.4),
                name='Reales (Ground Truth)', hoverinfo='text', hovertext=hover_text
            )
        else:
            trace_est = go.Scatter3d(
                x=plot_est[0, :], y=plot_est[1, :], z=plot_est[2, :], mode='markers',
                marker=dict(size=1.5, color='rgb(220, 80, 80)', opacity=0.3),
                name='Estimadas (Modelo)', hoverinfo='text', hovertext=hover_text
            )
            trace_true = go.Scatter3d(
                x=plot_true[0, :], y=plot_true[1, :], z=plot_true[2, :], mode='markers',
                marker=dict(size=1.5, color='rgb(60, 140, 220)', opacity=0.3),
                name='Reales (Ground Truth)', hoverinfo='text', hovertext=hover_text
            )
        pixel_traces.extend([trace_true, trace_est])

    elif effective_mode == "error_heatmap":
        # Modo Mapa de Calor: Colorear por RMSE local
        # Calcular RMSE por píxel en el submuestreo
        rmse = np.sqrt(np.mean((plot_est - plot_true)**2, axis=0)) if z_true is not None else np.zeros(plot_est.shape[1])

        if dim == 2:
            pixel_trace = go.Scatter(
                x=plot_est[0, :], y=plot_est[1, :], mode='markers',
                marker=dict(
                    size=4.0, color=rmse, colorscale='Jet', opacity=0.6,
                    colorbar=dict(title="RMSE Abundancia", thickness=15),
                    showscale=True
                ),
                name='Píxeles (Color = Error)', hoverinfo='text',
                hovertext=[f"Píxel #{idx}<br>RMSE: {err:.4f}" for idx, err in zip(sub_idx, rmse)]
            )
        else:
            pixel_trace = go.Scatter3d(
                x=plot_est[0, :], y=plot_est[1, :], z=plot_est[2, :], mode='markers',
                marker=dict(
                    size=1.8, color=rmse, colorscale='Jet', opacity=0.5,
                    colorbar=dict(title="RMSE Abundancia", thickness=15),
                    showscale=True
                ),
                name='Píxeles (Color = Error)', hoverinfo='text',
                hovertext=[f"Píxel #{idx}<br>RMSE: {err:.4f}" for idx, err in zip(sub_idx, rmse)]
            )
        pixel_traces.append(pixel_trace)

    elif effective_mode == "vectors":
        # Modo Vectores: Dibujar líneas de desplazamiento
        # Para evitar lag en el navegador, submuestreamos a un máximo de 1000 vectores
        K = min(1000, plot_est.shape[1])
        step = max(1, plot_est.shape[1] // K)
        v_est = plot_est[:, ::step]
        v_true = plot_true[:, ::step]

        # Armar coordenadas de líneas separadas por None
        line_x, line_y, line_z = [], [], []
        for i in range(v_est.shape[1]):
            line_x.extend([v_true[0, i], v_est[0, i], None])
            line_y.extend([v_true[1, i], v_est[1, i], None])
            if dim == 3:
                line_z.extend([v_true[2, i], v_est[2, i], None])

        # Trazar puntos de destino (estimados)
        if dim == 2:
            pixel_trace = go.Scatter(
                x=v_est[0, :], y=v_est[1, :], mode='markers',
                marker=dict(size=3.0, color='rgb(220, 80, 80)', opacity=0.5),
                name='Estimadas (Modelo)'
            )
            # Trazar puntos de origen (reales)
            origin_trace = go.Scatter(
                x=v_true[0, :], y=v_true[1, :], mode='markers',
                marker=dict(size=3.0, color='rgb(60, 140, 220)', opacity=0.5),
                name='Reales (Ground Truth)'
            )
            # Trazar líneas
            vector_trace = go.Scatter(
                x=line_x, y=line_y, mode='lines',
                line=dict(color='rgba(150, 150, 150, 0.4)', width=1),
                name='Desviación (Vector de Error)'
            )
            pixel_traces.extend([origin_trace, pixel_trace, vector_trace])
        else:
            pixel_trace = go.Scatter3d(
                x=v_est[0, :], y=v_est[1, :], z=v_est[2, :], mode='markers',
                marker=dict(size=1.5, color='rgb(220, 80, 80)', opacity=0.4),
                name='Estimadas (Modelo)'
            )
            origin_trace = go.Scatter3d(
                x=v_true[0, :], y=v_true[1, :], z=v_true[2, :], mode='markers',
                marker=dict(size=1.5, color='rgb(60, 140, 220)', opacity=0.4),
                name='Reales (Ground Truth)'
            )
            vector_trace = go.Scatter3d(
                x=line_x, y=line_y, z=line_z, mode='lines',
                line=dict(color='rgba(150, 150, 150, 0.4)', width=1),
                name='Desviación (Vector de Error)'
            )
            pixel_traces.extend([origin_trace, pixel_trace, vector_trace])

    elif effective_mode == "knn":
        # Modo K-NN Graph: Dibuja las conexiones de los k vecinos más cercanos en el espacio reducido.
        # Para que sea interactivo en la web (HTML), precalculamos las aristas para
        # un conjunto de valores de k y añadimos un slider nativo de Plotly.

        K = min(1000, plot_est.shape[1])
        step = max(1, plot_est.shape[1] // K)
        v_est = plot_est[:, ::step]
        X_est = v_est.T  # (K, dim)

        v_true = plot_true[:, ::step] if plot_true is not None else None
        X_true = v_true.T if v_true is not None else None

        C = np.mean(vertices, axis=0)

        # Añadir las trazas estáticas de píxeles
        if dim == 2:
            pixel_traces.append(go.Scatter(
                x=X_est[:, 0], y=X_est[:, 1], mode='markers',
                marker=dict(size=4.0, color='rgb(220, 80, 80)', opacity=0.8),
                name='Píxeles Estimados (Modelo)'
            ))
            if X_true is not None:
                pixel_traces.append(go.Scatter(
                    x=X_true[:, 0], y=X_true[:, 1], mode='markers',
                    marker=dict(size=4.0, color='rgb(60, 140, 220)', opacity=0.8),
                    name='Píxeles Reales (GT)', visible='legendonly'
                ))
        else:
            pixel_traces.append(go.Scatter3d(
                x=X_est[:, 0], y=X_est[:, 1], z=X_est[:, 2], mode='markers',
                marker=dict(size=2.0, color='rgb(220, 80, 80)', opacity=0.7),
                name='Píxeles Estimados (Modelo)'
            ))
            if X_true is not None:
                pixel_traces.append(go.Scatter3d(
                    x=X_true[:, 0], y=X_true[:, 1], z=X_true[:, 2], mode='markers',
                    marker=dict(size=2.0, color='rgb(60, 140, 220)', opacity=0.7),
                    name='Píxeles Reales (GT)', visible='legendonly'
                ))

        # Valores de k a precalcular
        k_list = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30]
        # El valor de k inicial será el configurado por el usuario o 5
        k_default = k_neighbors if k_neighbors in k_list else 5

        from sklearn.neighbors import NearestNeighbors

        dynamic_traces = []
        knn_traces_info = [] # Lista de tuplas (k_val, start_idx, end_idx)

        for k_val in k_list:
            # --- 1. Calcular K-NN para el Estimado (Modelo) ---
            nbrs_est = NearestNeighbors(n_neighbors=k_val + 1, algorithm='ball_tree').fit(X_est)
            _, indices_est = nbrs_est.kneighbors(X_est)

            line_x_est, line_y_est, line_z_est = [], [], []
            short_x_est, short_y_est, short_z_est = [], [], []

            for i in range(X_est.shape[0]):
                for neighbor_idx in indices_est[i, 1:]:
                    A_pt = X_est[i]
                    B_pt = X_est[neighbor_idx]

                    v = B_pt - A_pt
                    w = C - A_pt
                    v_len_sq = np.dot(v, v)
                    is_short = False
                    if v_len_sq > 0:
                        t = np.dot(w, v) / v_len_sq
                        t = np.clip(t, 0.0, 1.0)
                        closest = A_pt + t * v
                        dist_to_center = np.linalg.norm(closest - C)
                        edge_len = np.sqrt(v_len_sq)
                        is_short = (dist_to_center < 0.25) and (edge_len > 0.4)

                    if is_short:
                        short_x_est.extend([A_pt[0], B_pt[0], None])
                        short_y_est.extend([A_pt[1], B_pt[1], None])
                        if dim == 3:
                            short_z_est.extend([A_pt[2], B_pt[2], None])
                    else:
                        line_x_est.extend([A_pt[0], B_pt[0], None])
                        line_y_est.extend([A_pt[1], B_pt[1], None])
                        if dim == 3:
                            line_z_est.extend([A_pt[2], B_pt[2], None])

            # Visibilidad por defecto para este k_val
            is_k_default = (k_val == k_default)
            vis_state_est = True if is_k_default else False
            vis_state_gt = 'legendonly' if is_k_default else False

            # Traces de Modelo
            if dim == 2:
                t_m_knn = go.Scatter(
                    x=line_x_est, y=line_y_est, mode='lines',
                    line=dict(color='rgba(52, 211, 153, 0.25)', width=1),
                    name=f'Conexiones K-NN (Modelo, k={k_val})', visible=vis_state_est
                )
                t_m_short = go.Scatter(
                    x=short_x_est, y=short_y_est, mode='lines',
                    line=dict(color='rgba(239, 68, 68, 0.65)', width=1.5),
                    name=f'Cortocircuitos (Modelo, k={k_val})', visible=vis_state_est
                )
            else:
                t_m_knn = go.Scatter3d(
                    x=line_x_est, y=line_y_est, z=line_z_est, mode='lines',
                    line=dict(color='rgba(52, 211, 153, 0.2)', width=1),
                    name=f'Conexiones K-NN (Modelo, k={k_val})', visible=vis_state_est
                )
                t_m_short = go.Scatter3d(
                    x=short_x_est, y=short_y_est, z=short_z_est, mode='lines',
                    line=dict(color='rgba(239, 68, 68, 0.6)', width=1.5),
                    name=f'Cortocircuitos (Modelo, k={k_val})', visible=vis_state_est
                )

            dynamic_idx_start = len(dynamic_traces)
            dynamic_traces.append(t_m_knn)
            dynamic_traces.append(t_m_short)

            # --- 2. Calcular K-NN para el Ground Truth ---
            t_g_knn = None
            t_g_short = None
            if X_true is not None:
                nbrs_true = NearestNeighbors(n_neighbors=k_val + 1, algorithm='ball_tree').fit(X_true)
                _, indices_true = nbrs_true.kneighbors(X_true)

                line_x_true, line_y_true, line_z_true = [], [], []
                short_x_true, short_y_true, short_z_true = [], [], []

                for i in range(X_true.shape[0]):
                    for neighbor_idx in indices_true[i, 1:]:
                        A_pt = X_true[i]
                        B_pt = X_true[neighbor_idx]

                        v = B_pt - A_pt
                        w = C - A_pt
                        v_len_sq = np.dot(v, v)
                        is_short = False
                        if v_len_sq > 0:
                            t = np.dot(w, v) / v_len_sq
                            t = np.clip(t, 0.0, 1.0)
                            closest = A_pt + t * v
                            dist_to_center = np.linalg.norm(closest - C)
                            edge_len = np.sqrt(v_len_sq)
                            is_short = (dist_to_center < 0.25) and (edge_len > 0.4)

                        if is_short:
                            short_x_true.extend([A_pt[0], B_pt[0], None])
                            short_y_true.extend([A_pt[1], B_pt[1], None])
                            if dim == 3:
                                short_z_true.extend([A_pt[2], B_pt[2], None])
                        else:
                            line_x_true.extend([A_pt[0], B_pt[0], None])
                            line_y_true.extend([A_pt[1], B_pt[1], None])
                            if dim == 3:
                                line_z_true.extend([A_pt[2], B_pt[2], None])

                if dim == 2:
                    t_g_knn = go.Scatter(
                        x=line_x_true, y=line_y_true, mode='lines',
                        line=dict(color='rgba(99, 102, 241, 0.25)', width=1),
                        name=f'Conexiones K-NN (GT, k={k_val})', visible=vis_state_gt
                    )
                    t_g_short = go.Scatter(
                        x=short_x_true, y=short_y_true, mode='lines',
                        line=dict(color='rgba(239, 68, 68, 0.5)', width=1.5),
                        name=f'Cortocircuitos (GT, k={k_val})', visible=vis_state_gt
                    )
                else:
                    t_g_knn = go.Scatter3d(
                        x=line_x_true, y=line_y_true, z=line_z_true, mode='lines',
                        line=dict(color='rgba(99, 102, 241, 0.2)', width=1),
                        name=f'Conexiones K-NN (GT, k={k_val})', visible=vis_state_gt
                    )
                    t_g_short = go.Scatter3d(
                        x=short_x_true, y=short_y_true, z=short_z_true, mode='lines',
                        line=dict(color='rgba(239, 68, 68, 0.5)', width=1.5),
                        name=f'Cortocircuitos (GT, k={k_val})', visible=vis_state_gt
                    )
                dynamic_traces.append(t_g_knn)
                dynamic_traces.append(t_g_short)

            dynamic_idx_end = len(dynamic_traces)
            knn_traces_info.append((k_val, dynamic_idx_start, dynamic_idx_end))

        pixel_traces.extend(dynamic_traces)

    # 2. Vértices del Simplex
    vertex_traces = []

    # Preparar nombres y hovers de acuerdo al mapeo col_ind
    if col_ind is not None:
        est_labels = [f"E{i+1}: {class_names[col_ind[i]]}" if col_ind[i] < len(class_names) else f"E{i+1}" for i in range(M)]
        est_hover = [f"Estimado {i+1} (Asignado a: {class_names[col_ind[i]]})" if col_ind[i] < len(class_names) else f"Estimado {i+1}" for i in range(M)]
    else:
        est_labels = [f"E{i+1}" for i in range(M)]
        est_hover = [f"Estimado {i+1}" for i in range(M)]

    gt_labels = [f"GT{j+1}: {class_names[j]}" if j < len(class_names) else f"GT{j+1}" for j in range(M)]
    gt_hover = [f"Ground Truth {j+1}: {class_names[j]}" if j < len(class_names) else f"Ground Truth {j+1}" for j in range(M)]

    if vertices_true is not None:
        # Vértices estimados (Rojos)
        if dim == 2:
            vertex_trace_est = go.Scatter(
                x=vertices[:, 0], y=vertices[:, 1], mode='markers+text',
                marker=dict(size=12, color='rgb(220, 50, 50)', symbol='triangle-up', line=dict(color='black', width=2)),
                text=est_labels, hovertext=est_hover, hoverinfo='text+name', textposition="top center",
                textfont=dict(size=12, color='rgb(180, 20, 20)', family='sans-serif'),
                name=f'Endmembers Est. ({method_name})'
            )
        else:
            vertex_trace_est = go.Scatter3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], mode='markers+text',
                marker=dict(size=8, color='rgb(220, 50, 50)', symbol='diamond', line=dict(color='black', width=2)),
                text=est_labels, hovertext=est_hover, hoverinfo='text+name', textposition="top center",
                textfont=dict(size=12, color='rgb(180, 20, 20)', family='sans-serif'),
                name=f'Endmembers Est. ({method_name})'
            )

        # Vértices reales (Azules)
        if dim == 2:
            vertex_trace_true = go.Scatter(
                x=vertices_true[:, 0], y=vertices_true[:, 1], mode='markers+text',
                marker=dict(size=12, color='rgb(50, 50, 220)', symbol='circle', line=dict(color='black', width=2)),
                text=gt_labels, hovertext=gt_hover, hoverinfo='text+name', textposition="bottom center",
                textfont=dict(size=12, color='rgb(20, 20, 180)', family='sans-serif'),
                name='Endmembers GT (Real)'
            )
        else:
            vertex_trace_true = go.Scatter3d(
                x=vertices_true[:, 0], y=vertices_true[:, 1], z=vertices_true[:, 2], mode='markers+text',
                marker=dict(size=8, color='rgb(50, 50, 220)', symbol='circle', line=dict(color='black', width=2)),
                text=gt_labels, hovertext=gt_hover, hoverinfo='text+name', textposition="bottom center",
                textfont=dict(size=12, color='rgb(20, 20, 180)', family='sans-serif'),
                name='Endmembers GT (Real)'
            )
        vertex_traces.extend([vertex_trace_est, vertex_trace_true])
    else:
        # Fallback (Gris oscuro)
        fallback_labels = est_labels if col_ind is not None else class_names
        fallback_hover = est_hover if col_ind is not None else [f"Vértice: {name}" for name in class_names]
        if dim == 2:
            vertex_trace = go.Scatter(
                x=vertices[:, 0], y=vertices[:, 1], mode='markers+text',
                marker=dict(size=12, color='rgb(30, 30, 30)', symbol='circle', line=dict(color='black', width=2)),
                text=fallback_labels, hovertext=fallback_hover, hoverinfo='text+name', textposition="top center",
                textfont=dict(size=12, color='black', family='sans-serif'),
                name='Vértices (Materiales Puros)'
            )
        else:
            vertex_trace = go.Scatter3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], mode='markers+text',
                marker=dict(size=8, color='rgb(30, 30, 30)', symbol='circle', line=dict(color='black', width=2)),
                text=fallback_labels, hovertext=fallback_hover, hoverinfo='text+name', textposition="top center",
                textfont=dict(size=12, color='black', family='sans-serif'),
                name='Vértices (Materiales Puros)'
            )
        vertex_traces.append(vertex_trace)

    # 3. Aristas del Simplex (Esqueleto)
    edge_traces = []
    for edge in edges:
        start_idx, end_idx = edge

        # Aristas estimadas (Rojas si hay GT, de lo contrario Gris)
        edge_x = [vertices[start_idx, 0], vertices[end_idx, 0]]
        edge_y = [vertices[start_idx, 1], vertices[end_idx, 1]]
        color_est = 'rgb(220, 50, 50)' if vertices_true is not None else 'rgb(50, 50, 50)'

        if dim == 2:
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, mode='lines',
                line=dict(color=color_est, width=3),
                showlegend=False, hoverinfo='none'
            )
        else:
            edge_z = [vertices[start_idx, 2], vertices[end_idx, 2]]
            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z, mode='lines',
                line=dict(color=color_est, width=3, dash='solid'),
                showlegend=False, hoverinfo='none'
            )
        edge_traces.append(edge_trace)

        # Aristas reales (Azules con guiones)
        if vertices_true is not None:
            edge_x_true = [vertices_true[start_idx, 0], vertices_true[end_idx, 0]]
            edge_y_true = [vertices_true[start_idx, 1], vertices_true[end_idx, 1]]

            if dim == 2:
                edge_trace_true = go.Scatter(
                    x=edge_x_true, y=edge_y_true, mode='lines',
                    line=dict(color='rgb(50, 50, 220)', width=2, dash='dash'),
                    showlegend=False, hoverinfo='none'
                )
            else:
                edge_z_true = [vertices_true[start_idx, 2], vertices_true[end_idx, 2]]
                edge_trace_true = go.Scatter3d(
                    x=edge_x_true, y=edge_y_true, z=edge_z_true, mode='lines',
                    line=dict(color='rgb(50, 50, 220)', width=2, dash='dash'),
                    showlegend=False, hoverinfo='none'
                )
            edge_traces.append(edge_trace_true)

    # Crear figura y armar layout
    fig = go.Figure(data=pixel_traces + vertex_traces + edge_traces)

    # Construir el slider de Plotly si estamos en modo K-NN
    if effective_mode == "knn":
        steps = []
        for step_idx, k_val in enumerate(k_list):
            vis_list = []

            # 1. Píxeles Estimados (siempre visibles)
            vis_list.append(True)
            # 2. Píxeles Reales (siempre visible como legendonly, si existe)
            if X_true is not None:
                vis_list.append('legendonly')

            # 3. Dynamic Traces (K-NN lines)
            for info in knn_traces_info:
                curr_k, start_idx, end_idx = info
                if curr_k == k_val:
                    vis_list.append(True)          # model knn
                    vis_list.append(True)          # model short
                    if X_true is not None:
                        vis_list.append('legendonly')  # gt knn
                        vis_list.append('legendonly')  # gt short
                else:
                    vis_list.append(False)
                    vis_list.append(False)
                    if X_true is not None:
                        vis_list.append(False)
                        vis_list.append(False)

            # 4. Vértices (siempre visibles)
            for _ in vertex_traces:
                vis_list.append(True)
            # 5. Aristas del simplex (siempre visibles)
            for _ in edge_traces:
                vis_list.append(True)

            step = dict(
                method="restyle",
                args=["visible", vis_list],
                label=f"k={k_val}"
            )
            steps.append(step)

        default_active_idx = k_list.index(k_default) if k_default in k_list else 3

        sliders = [dict(
            active=default_active_idx,
            currentvalue={"prefix": "Vecinos K-NN: "},
            pad={"t": 50},
            steps=steps
        )]

    shape_name = "Tetraedro 3D" if M == 4 else "Triángulo 2D"
    mode_titles = {
        "basic": f"Espacio Latente ({shape_name})",
        "compare": f"Comparación Estimado vs Real ({shape_name})",
        "error_heatmap": f"Mapa de Calor de Error Local ({shape_name})",
        "vectors": f"Vectores de Error de Abundancia ({shape_name})",
        "knn": f"Grafo K-NN sobre Símplex ({shape_name})"
    }

    title_text = f"<b>{mode_titles.get(effective_mode, effective_mode)}</b><br>Dataset: {ds_name} | Reducción: {method_name}"

    if dim == 2:
        layout_kwargs = dict(
            title=dict(
                text=title_text, x=0.5, y=0.95, xanchor='center', yanchor='top',
                font=dict(size=15, family="sans-serif", color='black')
            ),
            xaxis=dict(title='X (Proyección)', showgrid=True, gridcolor='rgb(230, 230, 230)', zeroline=False),
            yaxis=dict(title='Y (Proyección)', showgrid=True, gridcolor='rgb(230, 230, 230)', zeroline=False, scaleanchor="x", scaleratio=1),
            plot_bgcolor='rgb(248, 248, 248)',
            margin=dict(l=40, r=40, b=40, t=80),
            legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.02, bgcolor='rgba(255, 255, 255, 0.75)')
        )
        if effective_mode == "knn":
            layout_kwargs["sliders"] = sliders
        fig.update_layout(**layout_kwargs)
    else:
        layout_kwargs = dict(
            title=dict(
                text=title_text, x=0.5, y=0.95, xanchor='center', yanchor='top',
                font=dict(size=15, family="sans-serif", color='black')
            ),
            scene=dict(
                xaxis=dict(title='X (Proyección)', gridcolor='rgb(220, 220, 220)', backgroundcolor='rgb(245, 245, 245)', showbackground=True),
                yaxis=dict(title='Y (Proyección)', gridcolor='rgb(220, 220, 220)', backgroundcolor='rgb(245, 245, 245)', showbackground=True),
                zaxis=dict(title='Z (Proyección)', gridcolor='rgb(220, 220, 220)', backgroundcolor='rgb(245, 245, 245)', showbackground=True),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=80),
            legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.02, bgcolor='rgba(255, 255, 255, 0.75)')
        )
        if effective_mode == "knn":
            layout_kwargs["sliders"] = sliders
        fig.update_layout(**layout_kwargs)

    fig.write_html(out_html)
    print(f"       [OK] Modo '{mode}' guardado en: {out_html}")


def main():
    parser = argparse.ArgumentParser(description="Visualización interactiva 2D/3D del Simplex de Abundancias.")
    parser.add_argument("--file", type=str, default=None, help="Ruta al archivo de abundancias (.npy o .mat).")
    parser.add_argument("--dataset", type=str, default="samson", help="Nombre del dataset (samson, jasper, urban, etc.) para cargar etiquetas de clases.")
    parser.add_argument("--method", type=str, default="FCLS", help="Nombre del método de unmixing (ej. FCLS).")
    parser.add_argument("--out", type=str, default="simplex.html", help="Nombre o ruta del archivo HTML de salida.")
    parser.add_argument("--mode", type=str, default="basic", choices=["basic", "compare", "error_heatmap", "vectors", "knn"],
                        help="Modo de visualización comparativo con el Ground Truth.")
    parser.add_argument("--k-neighbors", type=int, default=5, help="Número de vecinos para el grafo K-NN.")
    parser.add_argument("--latent-file", type=str, default=None, help="Ruta al archivo de espacio latente (latent_Z.npy).")
    parser.add_argument("--latent-est-file", type=str, default=None, help="Ruta al archivo de endmembers estimados en espacio latente (latent_Z_est.npy).")
    parser.add_argument("--latent-true-file", type=str, default=None, help="Ruta al archivo de endmembers reales en espacio latente (latent_Z_true.npy).")

    args = parser.parse_args()

    if args.latent_file is not None:
        if not os.path.exists(args.latent_file):
            raise FileNotFoundError(f"Archivo de espacio latente no encontrado: {args.latent_file}")

        # Cargar espacio latente de píxeles
        z_pixels_raw = np.load(args.latent_file)
        # Asegurar forma (dim, N)
        if z_pixels_raw.ndim == 2:
            d1, d2 = z_pixels_raw.shape
            if d1 < d2:
                z_pixels = z_pixels_raw.astype(np.float64)
            else:
                z_pixels = z_pixels_raw.T.astype(np.float64)
        else:
            raise ValueError(f"Dimensiones de espacio latente inesperadas: {z_pixels_raw.ndim}D")

        dim = z_pixels.shape[0]
        N = z_pixels.shape[1]

        # Resolver ruta de endmembers estimados en espacio latente
        if args.latent_est_file is None:
            dir_path = os.path.dirname(args.latent_file)
            args.latent_est_file = os.path.join(dir_path, 'latent_Z_est.npy')

        if not os.path.exists(args.latent_est_file):
            raise FileNotFoundError(f"Archivo de endmembers en espacio latente no encontrado: {args.latent_est_file}")

        vertices = np.load(args.latent_est_file)
        M = vertices.shape[0]

        # Resolver ruta de endmembers reales en espacio latente
        if args.latent_true_file is None:
            dir_path = os.path.dirname(args.latent_file)
            args.latent_true_file = os.path.join(dir_path, 'latent_Z_true.npy')

        z_true_vertices = None
        if os.path.exists(args.latent_true_file):
            z_true_vertices = np.load(args.latent_true_file)

        # Cargar Ground Truth (A_true) para evaluar pixeles
        A_true = None
        if args.mode in ["compare", "error_heatmap", "vectors", "knn"]:
            A_true = load_ground_truth(args.dataset)

        z_true = None
        if A_true is not None and z_true_vertices is not None:
            # Validar dimensiones
            if A_true.shape[0] == z_true_vertices.shape[0]:
                z_true = np.dot(z_true_vertices.T, A_true)
            else:
                print(f"       [WARNING] Dimensión de clases A_true {A_true.shape} no coincide con vertices GT {z_true_vertices.shape}. Desactivando modo comparativo.")
                if args.mode != "knn":
                    args.mode = "basic"
        else:
            if args.mode != "knn" and args.mode != "basic":
                print("       [WARNING] No se pudo calcular la posición real (GT) de los píxeles. Volviendo a modo 'basic'.")
                args.mode = "basic"
    else:
        if args.file is None:
            raise ValueError("Debe proporcionar --file o --latent-file.")

        if not os.path.exists(args.file):
            raise FileNotFoundError(f"Archivo no encontrado: {args.file}")

        # 1. Cargar abundancias estimadas
        A, M, N = load_abundance_data(args.file)

        # 2. Cargar Ground Truth si es necesario
        A_true = None
        if args.mode in ["compare", "error_heatmap", "vectors", "knn"]:
            A_true = load_ground_truth(args.dataset)
            if A_true is not None:
                # Validar formas
                if A_true.shape != A.shape:
                    print(f"       [WARNING] Dimensiones de A_true {A_true.shape} no coinciden con A {A.shape}. Desactivando modo comparativo/referencial.")
                    if args.mode != "knn":
                        args.mode = "basic"
                    A_true = None
            else:
                if args.mode != "knn":
                    print("       [WARNING] No se pudo cargar el Ground Truth. Volviendo a modo 'basic'.")
                    args.mode = "basic"

        # 3. Definir coordenadas geométricas del simplex estándar
        if M == 3:
            # Triángulo equilátero en 2D (centrado en el origen)
            vertices = np.array([
                [0.0, 1.0],
                [-np.sqrt(3)/2, -0.5],
                [np.sqrt(3)/2, -0.5]
            ])
        elif M == 4:
            # Tetraedro regular estándar en 3D
            vertices = np.array([
                [1.0,  1.0,  1.0],
                [1.0, -1.0, -1.0],
                [-1.0,  1.0, -1.0],
                [-1.0, -1.0,  1.0]
            ])
        else:
            print(f"       [WARNING] M={M} no estándar. Generando simplex base en 3D.")
            np.random.seed(42)
            vertices = np.random.randn(M, 3)
            norms = np.linalg.norm(vertices, axis=1, keepdims=True)
            vertices = vertices / (norms + 1e-10)

        # Proyectar
        z_pixels = project_to_simplex(A, vertices)
        z_true = project_to_simplex(A_true, vertices) if A_true is not None else None

    # Definir aristas del simplex según el número de vértices M
    if M == 3:
        edges = [(0, 1), (1, 2), (2, 0)]
    elif M == 4:
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    else:
        edges = []
        for i in range(M):
            for j in range(i+1, M):
                edges.append((i, j))

    # Definir etiquetas de clases
    ds_key = args.dataset.lower()
    if 'samson' in ds_key:
        ds_key = 'samson'
    elif 'jasper' in ds_key:
        ds_key = 'jasper'
    elif 'urban' in ds_key:
        ds_key = 'urban'
    elif 'apex' in ds_key:
        ds_key = 'apex'

    if ds_key in DATASET_CLASSES:
        class_names = DATASET_CLASSES[ds_key]
        if len(class_names) < M:
            class_names = class_names + [f"Material {i+1}" for i in range(len(class_names), M)]
        else:
            class_names = class_names[:M]
    else:
        class_names = [f"Material {i+1}" for i in range(M)]

    # 6. Graficar
    vertices_true = z_true_vertices if args.latent_file is not None else None
    col_ind = None
    if vertices_true is not None:
        try:
            from scipy.optimize import linear_sum_assignment
            cost_matrix = np.zeros((M, M))
            for i in range(M):
                for j in range(M):
                    cost_matrix[i, j] = np.linalg.norm(vertices[i] - vertices_true[j])
            _, col_ind = linear_sum_assignment(cost_matrix)
        except Exception as e:
            print(f"       [WARNING] No se pudo calcular el alineamiento húngaro latente: {e}")

    plot_latent_simplex(z_pixels, vertices, edges, class_names, args.method, args.dataset, args.out, args.mode, z_true, args.k_neighbors, vertices_true=vertices_true, col_ind=col_ind)


if __name__ == "__main__":
    main()
