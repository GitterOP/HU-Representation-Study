import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .loaders import DatasetLoader, DATASET_REGISTRY
from .dim_reduction import DimensionalityReducer
from .unmixing import (
    turbo_extract_nfindr,
    turbo_fcls_map,
    reconstruct_and_compute_metrics,
)
from .common import hyperHfcVd_Adjusted
from .metrics import compute_aRMSE, compute_SID, compute_ERGAS
from .visuals import (
    generate_spectral_comparison,
    generate_visual_comparison,
    generate_simplex_plot,
)
# Lazy loading for DL unmixers to avoid failures if TensorFlow is not installed
# they are imported inside Pipeline.run() when needed


class PipelineResult(list):
    """
    A custom list subclass that allows dictionary-style lookup for backward compatibility.
    If the list contains only one result, indexing it with a string (e.g., results['rmse'])
    will return the value of that key from the single result dictionary.
    If the list contains multiple results, it returns the value from the last result.
    """
    def __getitem__(self, key):
        if isinstance(key, str):
            if len(self) == 1:
                return self[0][key]
            elif len(self) > 1:
                return self[-1][key]
            else:
                raise KeyError("PipelineResult is empty and cannot be accessed by key.")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if isinstance(key, str):
            if len(self) == 1:
                return self[0].get(key, default)
            elif len(self) > 1:
                return self[-1].get(key, default)
            else:
                return default
        return default


class Pipeline:
    DEFAULT_N_CLASSES = {
        'Pavia': 9,
        'Salinas': 16,
        'Indian': 16
    }

    def __init__(self, config):
        self.config = config

    def run(self):
        import builtins
        _orig_print = builtins.print
        is_verbose = bool(self.config.get('verbose', 1))
        
        if not is_verbose:
            def _custom_print(*args, **kwargs):
                pass
            builtins.print = _custom_print
            
        try:
            ds_config = self.config.get('dataset', {})
            dr_config = self.config.get('dimensionality_reduction', {})
            
            dataset_names = ds_config.get('names', [ds_config.get('name')] if 'name' in ds_config else [])
            dr_methods = dr_config.get('methods', [dr_config.get('method')] if 'method' in dr_config else ["PCA"])
            
            # Prepare combinations for the progress bar
            combinations = []
            for ds_name in dataset_names:
                for dr_method in dr_methods:
                    combinations.append((ds_name, dr_method))
            
            results = PipelineResult()
            
            if not is_verbose:
                from tqdm import tqdm
                pbar = tqdm(combinations, desc="Ejecutando EMBase", unit="exp")
            else:
                pbar = combinations
                
            for ds_name, dr_method in pbar:
                if not is_verbose:
                    pbar.set_postfix(dataset=ds_name, metodo=dr_method)
                else:
                    _orig_print(f"\n{'='*70}")
                    _orig_print(f"[+] INICIANDO PIPELINE: Dataset={ds_name} | Metodo={dr_method}")
                    _orig_print(f"{'='*70}\n")
                
                try:
                    res = self._run_single(ds_name, dr_method)
                    results.append(res)
                except Exception as e:
                    import traceback
                    _orig_print(f"\n[!] ERROR FATAL ejecutando {ds_name} con {dr_method}: {e}")
                    if is_verbose:
                        traceback.print_exc()
                finally:
                    import gc
                    gc.collect()
                    try:
                        import tensorflow as tf
                        tf.keras.backend.clear_session()
                    except ImportError:
                        pass
            
            _orig_print("\n[!] BATERIA DE EXPERIMENTOS FINALIZADA. Revisa results_cl/Global_Summary/")
            return results
            
        finally:
            if not is_verbose:
                builtins.print = _orig_print

    @staticmethod
    def _compute_simplex_volume(vertices):
        """Calcula el volumen del simplex en dimension N-1 a partir de vertices (N, N-1).
        Formula: Volumen = |det(D)| / (N-1)!
        donde D es la matriz de vectores diferencias con respecto a un vertice.
        """
        N = vertices.shape[0]
        if N < 2:
            return 0.0
        D = vertices[:-1] - vertices[-1]
        import math
        from scipy.linalg import det
        try:
            vol = abs(det(D)) / math.factorial(N - 1)
        except Exception:
            vol = 0.0
        return float(vol)

    def _run_single(self, ds_name, dr_method):
        ds     = self.config['dataset']
        dr     = self.config['dimensionality_reduction']
        um     = self.config['unmixing']
        
        # Override the current loop values
        ds['name'] = ds_name
        dr['method'] = dr_method
        method = um.get('method', 'FCLS')

        execution_log = []
        def log_time(stage_name, start_t):
            duration = time.time() - start_t
            msg = f"{stage_name}: {duration:.4f}s"
            print(f"       [TIME] {msg}")
            execution_log.append(msg)
            return time.time()

        t_total = time.time()
        t_start = t_total

        ds_name = ds['name']
        dr_method = dr['method'].upper() if method == 'FCLS' else method
        
        # Definir estructura de carpetas
        base_results = self.config.get('results_dir', 'results_cl')
        reports_dir = os.path.join(base_results, 'Reports', dr_method, ds_name)
        models_dir = os.path.join(base_results, 'Models', dr_method, ds_name)
        latent_dir = os.path.join(base_results, 'Latent', dr_method, ds_name)
        unmixing_dir = os.path.join(base_results, 'Unmixing', dr_method, ds_name)
        
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(latent_dir, exist_ok=True)
        os.makedirs(unmixing_dir, exist_ok=True)

        # [0/6] Load
        print("[0/6] Cargando dataset...")
        dataset_key = None
        for k, v in self.config.get('datasets', {}).items():
            if v['name'] == ds['name']:
                dataset_key = k
                break
        
        is_gt_dataset = dataset_key is not None
        A_true = None
        M_true = None

        if is_gt_dataset:
            from .dataset_loaders_gt import GroundTruthDataset
            dataset = GroundTruthDataset(self.config['datasets'][dataset_key])
            Y, M_true, A_true, meta = dataset.load()
            rows, cols, bands = meta['shape']['rows'], meta['shape']['cols'], meta['shape']['bands']
            image = Y.T.reshape(rows, cols, bands)
            del Y
            gt = None
            print(f"       {meta['name']}: {rows}x{cols}x{bands} (GT Dataset)")
        else:
            loader = DatasetLoader(ds['name'], ds['path'])
            image, gt, meta = loader.load()
            rows, cols, bands = meta['rows'], meta['cols'], meta['bands']
            print(f"       {meta['dataset']}: {rows}x{cols}x{bands}")

        t_start = log_time("Carga de Datos", t_start)

        # [1/6] Define/Estimate N
        if is_gt_dataset and M_true is not None:
            N = M_true.shape[1]
            print(f"[1/6] Usando número de endmembers del Ground Truth para {ds['name']}: N={N}")
        elif ds.get('estimate_n_endmembers', False):
            print("[1/6] Estimando número de endmembers (HFC Ajustado)...")
            N = hyperHfcVd_Adjusted(image, far=1e-5)
        else:
            N = self.DEFAULT_N_CLASSES.get(ds['name'], 9)
            print(f"[1/6] Usando número de endmembers predefinido para {ds['name']}: N={N}")

        n_endmembers = int(N)
        n_components = N - 1
        t_start = log_time("Estimación de N", t_start)

        # [2/6] Dimensionality reduction — only required for FCLS/NFINDR
        reducer = None
        if method == 'FCLS':
            print(f"[2/6] Reducción dimensional ({dr['method']}: {bands} -> {n_components})...")
            reducer    = DimensionalityReducer.create(dr['method'], n_components, **dr)
            reduced_2d = reducer.fit_transform(image, n_components)
            print(f"       Datos reducidos: {reduced_2d.shape}")
            
            np.save(os.path.join(latent_dir, 'latent_Z.npy'), reduced_2d)
            if hasattr(reducer, 'model') and reducer.model is not None:
                reducer.model.save(os.path.join(models_dir, 'model.keras'))
                print(f"       [OK] Modelo guardado en: {models_dir}")
        else:
            print(f"[2/6] Reducción dimensional omitida (no requerida por {method}).")
            reduced_2d = None
        t_start = log_time("Reducción Dimensional", t_start)

        # [3/6] & [4/6] — método de unmixing
        M_2d     = image.reshape(-1, bands)
        data_max = float(image.max()) or 1.0

        if method != 'FCLS':
            print(f"       [WARNING] Método de unmixing '{method}' no soportado o removido. Usando FCLS.")
            method = 'FCLS'

        print("[3/6] Extrayendo endmembers (N-FINDR Turbo)...")
        endmembers = turbo_extract_nfindr(
            image, n_endmembers,
            transform=reduced_2d,
            maxit=um.get('nfindr_maxit', 5),
        )
        print(f"       Endmembers: {endmembers.shape}")

        print("[4/6] Calculando abundancias (FCLS Turbo)...")
        abundances = turbo_fcls_map(
            image, endmembers,
            iterations=um.get('fcls_iterations', 10),
        )
        print(f"       Abundancias: {abundances.shape}")
        t_start = log_time("Proceso de Unmixing", t_start)

        # [4.5/6] Validate Endmembers against Ground Truth References
        val = self.validate_endmembers(endmembers, ds['name'], method, reports_dir, M_true)
        t_start = log_time("Validación de Endmembers", t_start)

        # aRMSE: construir H_true one-hot desde GT map usando el matching húngaro, o usar A_true directamente
        aRMSE = float('nan')
        if val is not None:
            if A_true is not None:
                H_true = A_true.T
                H_est = abundances.reshape(-1, n_endmembers)
                H_true_matched = np.zeros_like(H_est)
                for i, j in zip(val['row_ind'], val['col_ind']):
                    H_true_matched[:, i] = H_true[:, j]
                aRMSE = compute_aRMSE(H_true_matched, H_est)
                print(f"       aRMSE (vs A_true continuo): {aRMSE:.4f}")
            elif gt is not None:
                aRMSE = self._compute_aRMSE_from_gt(
                    gt, abundances, val['row_ind'], val['col_ind'], n_endmembers
                )
                print(f"       aRMSE (vs GT one-hot): {aRMSE:.4f}")

        sad_mean = val['sad_mean'] if val else float('nan')
        sid_mean = val['sid_mean'] if val else float('nan')

        # [4.6/6] Spectral Signature Comparison Plot
        if val:
            print("[4.6/6] Generando comparativa de firmas espectrales...")
            generate_spectral_comparison(endmembers, val['pairs'], ds['name'], method, reports_dir, M_true)
        t_start = log_time("Gráfico Espectral", t_start)

        # [4.7/6] Simplex Visualization Plot
        generate_simplex_plot(self.config, image, endmembers, reducer, M_true, method, dr['method'], reports_dir, A_true, abundances)
        t_start = log_time("Visualización de Simplex", t_start)


        # [5/6] Reconstruct — modelo lineal de mezcla para todos los métodos
        print("[5/6] Reconstruyendo imagen...")
        reconstructed = np.dot(
            abundances.reshape(-1, n_endmembers), endmembers
        ).reshape(rows, cols, bands)
        t_start = log_time("Reconstrucción de Imagen", t_start)

        # [4.8/6] Visual Validation Dashboard
        print("[4.8/6] Generando dashboard visual de reconstrucción...")
        generate_visual_comparison(image, reconstructed, meta, method, reports_dir)
        t_start = log_time("Generación de Dashboard", t_start)

        # [6/6] Metrics
        print("[6/6] Calculando metricas...")
        rmse, sam = reconstruct_and_compute_metrics(image, reconstructed)
        ergas = compute_ERGAS(image, reconstructed)

        # Nuevas métricas en espacio reducido usando el matching Húngaro
        latent_dist_mean = None
        simplex_volume_diff = None

        if M_true is not None and reducer is not None and val is not None:
            try:
                # Proyectar endmembers estimados y de Ground Truth al espacio reducido
                z_est = reducer.transform(endmembers)
                z_true = reducer.transform(M_true.T)
                
                # Alinear z_true al orden de los estimados
                z_true_matched = np.zeros_like(z_est)
                for i, j in zip(val['row_ind'], val['col_ind']):
                    z_true_matched[i] = z_true[j]
                
                # Distancia Euclídea promedio
                dists = np.linalg.norm(z_est - z_true_matched, axis=1)
                latent_dist_mean = float(np.mean(dists))
                
                # Diferencia de volúmenes de símplex
                vol_est = self._compute_simplex_volume(z_est)
                vol_true = self._compute_simplex_volume(z_true)
                simplex_volume_diff = float(abs(vol_est - vol_true))
            except Exception as e:
                print(f"       [!] Error al calcular métricas en espacio reducido: {e}")
        
        # Guardar matrices resultantes
        np.save(os.path.join(unmixing_dir, f"{ds_name}_{method}_endmembers.npy"), endmembers)
        np.save(os.path.join(unmixing_dir, f"{ds_name}_{method}_abundances.npy"), abundances)
        t_start = log_time("Cálculo de Métricas y Guardado", t_start)

        # Explicitly delete bulky arrays to free memory
        if 'image' in locals():
            del image
        if 'M_2d' in locals():
            del M_2d
        if 'M_true' in locals():
            del M_true

        total_duration = time.time() - t_total
        print(f"\n[!] Tiempo total de ejecución: {total_duration:.2f}s")

        # Guardar log de tiempos
        log_path = os.path.join(reports_dir, f"timing.log")
        with open(log_path, 'w') as f_log:
            f_log.write(f"LOG DE TIEMPOS - {ds['name']} - {method}\n")
            f_log.write("="*40 + "\n")
            for line in execution_log:
                f_log.write(line + "\n")
            f_log.write("-" * 40 + "\n")
            f_log.write(f"TIEMPO TOTAL: {total_duration:.4f}s\n")
            
        import json
        import json
        metrics_dict = {
            "Dataset": ds_name,
            "Method_DimRed": dr_method,
            "Method_Unmixing": method,
            "Metrics": {
                "aRMSE": float(aRMSE) if not np.isnan(aRMSE) else None,
                "rRMSE": float(rmse),
                "SAM": float(sam),
                "mSAD": float(sad_mean),
                "mSID": float(sid_mean),
                "ERGAS": float(ergas),
                "Latent_Dist_Mean": latent_dist_mean,
                "Simplex_Volume_Diff": simplex_volume_diff
            },
            "Performance": {
                "Time_Total": float(total_duration)
            }
        }
        with open(os.path.join(reports_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=4)
            
        self._update_global_summary(base_results, metrics_dict)

        W = 42
        print(f"\n{'='*W}")
        print(f"  INFORME FINAL DE CALIDAD - {method}")
        print(f"{'='*W}")
        print(f"  {'Metrica':<22} {'Valor':>10}")
        print(f"  {'-'*32}")
        print(f"  {'SAD medio':<22} {sad_mean:>10.4f}  <-- endmembers vs GT")
        print(f"  {'SID medio':<22} {sid_mean:>10.6f}  <-- endmembers vs GT")
        print(f"  {'aRMSE':<22} {aRMSE:>10.4f}  <-- abundancias vs GT")
        print(f"  {'rRMSE':<22} {rmse:>10.4f}  <-- reconstruccion")
        print(f"  {'SAM global':<22} {sam:>10.4f}  <-- reconstruccion")
        print(f"  {'ERGAS':<22} {ergas:>10.4f}  <-- reconstruccion")
        if latent_dist_mean is not None:
            print(f"  {'Dist Latente m.':<22} {latent_dist_mean:>10.4f}  <-- espacio reducido")
        if simplex_volume_diff is not None:
            print(f"  {'Dif Vol Simplex':<22} {simplex_volume_diff:>10.6f}  <-- espacio reducido")
        print(f"{'='*W}")

        ret_dict = {
            'reconstructed': None,
            'abundances':    None,
            'endmembers':    endmembers,
            'rmse':          rmse,
            'sam':           sam,
            'ergas':         ergas,
            'sad':           sad_mean,
            'sid':           sid_mean,
            'aRMSE':         aRMSE,
            'N':             N,
            'metadata':      meta,
            'gt':            None,
        }
        
        # Clean up remaining local large variables
        del reconstructed, abundances
        if 'gt' in locals() and gt is not None:
            del gt
        if 'reduced_2d' in locals() and reduced_2d is not None:
            del reduced_2d
        if 'reducer' in locals() and reducer is not None:
            del reducer

        return ret_dict

    def _update_global_summary(self, base_dir, metrics_dict):
        import pandas as pd
        
        global_dir = os.path.join(base_dir, 'Global_Summary')
        os.makedirs(global_dir, exist_ok=True)
        master_csv = os.path.join(global_dir, 'master_metrics.csv')
        
        row = {
            'Dataset': metrics_dict['Dataset'],
            'Method_DimRed': metrics_dict['Method_DimRed'],
            'Method_Unmixing': metrics_dict['Method_Unmixing'],
            'aRMSE': metrics_dict['Metrics']['aRMSE'],
            'rRMSE': metrics_dict['Metrics']['rRMSE'],
            'SAM': metrics_dict['Metrics']['SAM'],
            'mSAD': metrics_dict['Metrics']['mSAD'],
            'Latent_Dist_Mean': metrics_dict['Metrics'].get('Latent_Dist_Mean'),
            'Simplex_Volume_Diff': metrics_dict['Metrics'].get('Simplex_Volume_Diff'),
            'Time_Total': metrics_dict['Performance']['Time_Total']
        }
        
        df_new = pd.DataFrame([row])
        if os.path.exists(master_csv):
            df = pd.read_csv(master_csv)
            df = pd.concat([df, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(master_csv, index=False)
        
        # 1. Top 5 Global (Mejor aRMSE general)
        top_global = df.sort_values(by='aRMSE', ascending=True).head(5)
        top_global.to_csv(os.path.join(global_dir, 'leaderboard_global_top5.csv'), index=False)
        
        # 2. Top 5 por Dataset
        top_ds = df.sort_values(by='aRMSE', ascending=True).groupby('Dataset').head(5)
        top_ds.to_csv(os.path.join(global_dir, 'leaderboard_dataset_top5.csv'), index=False)
        
        # 3. Top 5 por Modelo
        top_model = df.sort_values(by='aRMSE', ascending=True).groupby('Method_DimRed').head(5)
        top_model.to_csv(os.path.join(global_dir, 'leaderboard_models_top5.csv'), index=False)

    @staticmethod
    def _compute_aRMSE_from_gt(gt, abundances, row_ind, col_ind, n_endmembers):
        """
        Construye H_true one-hot desde el mapa de etiquetas GT y calcula aRMSE.

        gt:            (rows, cols) mapa de clases con valores 1..n_classes (0=sin etiqueta)
        abundances:    (rows, cols, n_endmembers) abundancias estimadas
        row_ind:       asignación húngara → índice de endmember extraído (0-based)
        col_ind:       asignación húngara → índice de clase de referencia (0-based)
        n_endmembers:  número de endmembers del pipeline
        Returns: float — aRMSE sobre los píxeles etiquetados
        """
        gt_flat = gt.ravel()

        # Mapear clase GT (1-indexed) → endmember (0-indexed) según el matching
        class_to_em = {int(col_ind[k]) + 1: int(row_ind[k]) for k in range(len(row_ind))}

        rows, cols = gt.shape
        H_true = np.zeros((rows * cols, n_endmembers), dtype=np.float64)
        for class_id, em_idx in class_to_em.items():
            H_true[gt_flat == class_id, em_idx] = 1.0

        labeled = gt_flat > 0
        if not np.any(labeled):
            return float('nan')

        H_true_lbl = H_true[labeled]
        H_est_lbl  = abundances.reshape(-1, n_endmembers)[labeled]
        return compute_aRMSE(H_true_lbl, H_est_lbl)

    def validate_endmembers(self, extracted, ds_name, method, out_dir, M_true=None):
        """
        Compara los endmembers extraidos con las firmas de referencia del Ground Truth.
        Realiza un matching por cercania espectral (SAM) y guarda los resultados.
        """
        if M_true is not None:
            references = M_true.T  # Transponer a (N_ref, Bands)
        else:
            ref_path = os.path.join('Spectral_signatures', ds_name, f'signatures_{ds_name.lower()}.npy')
            
            if not os.path.exists(ref_path):
                print(f"\n[!] Advertencia: No se encontró archivo de referencia en {ref_path}")
                return None

            references = np.load(ref_path).T  # Transponer a (N_ref, Bands)
        n_extracted = extracted.shape[0]
        n_ref = references.shape[0]

        print(f"\n[4.5/6] Validando {n_extracted} endmembers contra {n_ref} referencias...")
        
        # Construir matriz de costos (SAM) para el algoritmo húngaro
        cost_matrix = np.zeros((n_extracted, n_ref))
        rmse_matrix = np.zeros((n_extracted, n_ref))

        for i in range(n_extracted):
            ext_sig = extracted[i]
            for j in range(n_ref):
                ref_sig = references[j]
                
                # SAM
                dot = np.dot(ext_sig, ref_sig)
                norm = (np.linalg.norm(ext_sig) * np.linalg.norm(ref_sig)) + 1e-10
                sam = np.arccos(np.clip(dot / norm, -1.0, 1.0)) * (180 / np.pi)
                
                # RMSE
                rmse = np.sqrt(np.mean((ext_sig - ref_sig)**2))
                
                cost_matrix[i, j] = sam
                rmse_matrix[i, j] = rmse

        # Resolver el problema de asignación lineal (Algoritmo Húngaro)
        # Esto minimiza la suma total de SAMs asegurando emparejamientos únicos
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        results = []
        total_sam = 0
        total_rmse = 0

        for i, j in zip(row_ind, col_ind):
            best_sam = cost_matrix[i, j]
            best_rmse = rmse_matrix[i, j]

            results.append({
                'Extracted_ID': i + 1,
                'Matched_Class_ID': j + 1,
                'SAM_deg': best_sam,
                'RMSE': best_rmse
            })

            total_sam += best_sam
            total_rmse += best_rmse

        n_matched = len(results)
        avg_sam  = total_sam  / n_matched if n_matched > 0 else 0.0
        avg_rmse = total_rmse / n_matched if n_matched > 0 else 0.0

        # SID por par emparejado (shapes ya son (N, bands) en ambos arrays)
        s_ref  = references[col_ind]   # (n_matched, bands)
        s_test = extracted[row_ind]    # (n_matched, bands)
        sid_mean, sid_per_pair = compute_SID(s_ref, s_test)
        for k, r in enumerate(results):
            r['SID'] = float(sid_per_pair[k])
        avg_sid = float(sid_mean)

        # Imprimir tabla en terminal
        print(f"{'='*70}")
        print(f"{'ID Ext.':<8} | {'ID Ref.':<8} | {'SAD (°)':<12} | {'RMSE':<12} | {'SID':<12}")
        print(f"{'-'*70}")
        for r in results:
            print(f"{r['Extracted_ID']:<8} | {r['Matched_Class_ID']:<8} | "
                  f"{r['SAM_deg']:<12.4f} | {r['RMSE']:<12.4f} | {r['SID']:<12.6f}")
        print(f"{'-'*70}")
        print(f"{'PROMEDIO':<19} | {avg_sam:<12.4f} | {avg_rmse:<12.4f} | {avg_sid:<12.6f}")
        print(f"{'='*70}")

        # Guardar en CSV
        os.makedirs(out_dir, exist_ok=True)
        csv_file = os.path.join(out_dir, f"{ds_name}_N-FINDR_validation.csv")

        with open(csv_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Extracted_ID', 'Matched_Class_ID', 'SAM_deg', 'RMSE', 'SID'])
            writer.writeheader()
            writer.writerows(results)
            writer.writerow({
                'Extracted_ID': 'AVERAGE',
                'Matched_Class_ID': '-',
                'SAM_deg': avg_sam,
                'RMSE': avg_rmse,
                'SID': avg_sid,
            })

        print(f"       [OK] Detalle guardado en: {csv_file}")
        return {
            'pairs':     results,
            'row_ind':   row_ind,
            'col_ind':   col_ind,
            'sad_mean':  avg_sam,
            'rmse_mean': avg_rmse,
            'sid_mean':  avg_sid,
        }
