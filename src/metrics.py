import numpy as np


def compute_aRMSE(H_true, H_estimated):
    """
    Abundance Root Mean Squared Error (aRMSE).

    Mide el error de estimación de abundancias comparando contra el Ground Truth.
    Es la métrica más importante para validar si el unmixing fue exitoso.

    H_true:      (n_pixels, n_endmembers) abundancias ground truth (ej. one-hot desde GT map)
    H_estimated: (n_pixels, n_endmembers) abundancias estimadas por el pipeline
    Returns: float — promedio global del RMSE por píxel

    Fórmula: (1/R) * Σ_r sqrt( (1/N) * Σ_n (H_true[r,n] - H_est[r,n])² )

    Interpretación:
        < 0.1   → Muy bueno
        0.1-0.2 → Bueno
        0.2-0.3 → Aceptable
        > 0.3   → Pobre
    """
    squared_errors = (H_true - H_estimated) ** 2
    rmse_per_pixel = np.sqrt(np.mean(squared_errors, axis=1))
    return float(np.mean(rmse_per_pixel))


def compute_SID(s_ref, s_test):
    """
    Spectral Information Divergence (SID) — divergencia KL simétrica.

    Más robusta que SAD para variabilidad espectral: captura diferencias en la
    distribución de energía de la firma, no solo el ángulo vectorial.

    s_ref:  (N, bands) firmas espectrales de referencia — una firma por fila
    s_test: (N, bands) firmas estimadas, emparejadas fila-a-fila con s_ref
    Returns: (float, ndarray) — SID medio y SID por par

    Fórmula (simétrica): SID(p,q) = D(p||q) + D(q||p)
        donde D(p||q) = Σ_i p_i * log(p_i / q_i)  (KL divergencia)
    Normalización: cada firma se normaliza a distribución de probabilidad sobre bandas.

    Interpretación:
        < 0.01  → Muy similar
        0.01-0.05 → Similar
        0.05-0.1  → Diferencia moderada
        > 0.1   → Muy diferente
    """
    eps = 1e-10
    p = s_ref  / (np.sum(s_ref,  axis=1, keepdims=True) + eps)
    q = s_test / (np.sum(s_test, axis=1, keepdims=True) + eps)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    sid_per_pair = (
        np.sum(p * np.log(p / q), axis=1) +
        np.sum(q * np.log(q / p), axis=1)
    )
    return float(np.mean(sid_per_pair)), sid_per_pair


def compute_ERGAS(Y_original, X_reconstructed, d=1):
    """
    Erreur Relative Globale Adimensionelle de Synthèse (ERGAS).

    Combina error espacial y espectral en una sola métrica radiométrica global.
    Normaliza el RMSE de cada banda por su media, evitando sesgo hacia bandas
    con mayor energía absoluta.

    Y_original:      (rows, cols, bands) imagen original
    X_reconstructed: (rows, cols, bands) imagen reconstruida
    d: factor de resolución espacial (default=1, sin contexto pan-sharpening)
    Returns: float — menor es mejor

    Fórmula: 100 * d * sqrt( (1/L) * Σ_l (RMSE_l / μ_l)² )

    Interpretación:
        < 3   → Excelente
        3-5   → Muy bueno
        5-8   → Bueno
        > 8   → Pobre
    """
    eps = 1e-10
    mse_per_band  = np.mean((Y_original - X_reconstructed) ** 2, axis=(0, 1))
    rmse_per_band = np.sqrt(mse_per_band)
    mean_per_band = np.mean(Y_original, axis=(0, 1))
    relative_errors = (rmse_per_band / (mean_per_band + eps)) ** 2
    return float(100.0 * d * np.sqrt(np.mean(relative_errors)))
