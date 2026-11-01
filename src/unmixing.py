import numpy as np

# Compatibility patches for pysptools on modern NumPy (removes np.float/np.int deprecations)
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int


def estimate_n_endmembers(M):
    """Estimate number of endmembers using HySime."""
    import pysptools.material_count as mc
    hysime = mc.HySime()
    p, _ = hysime.count(M)
    return p


def turbo_extract_nfindr(M, q, transform=None, maxit=3):
    """
    Vectorized N-FINDR endmember extraction.
    transform: 2D array (n_pixels, n_components) from prior dim reduction, or None.
    Returns endmembers array of shape (q, bands).
    """
    from pysptools.eea import eea as eea_mod

    rows, cols, bands = M.shape
    M_2d = M.reshape(-1, bands)

    if transform is not None:
        if transform.ndim == 3:
            transform = transform.reshape(-1, transform.shape[2])
    else:
        from sklearn.decomposition import PCA as _PCA
        transform = _PCA(n_components=q - 1).fit_transform(M_2d)

    nsamples = transform.shape[0]
    transform_h = np.column_stack((np.ones(nsamples), transform))

    _, idx = eea_mod.ATGP(M_2d, q)

    E = transform_h[idx, :].T
    for _ in range(maxit):
        for i in range(q):
            inv_E = np.linalg.pinv(E)
            vol_ratios = np.abs(np.dot(inv_E, transform_h.T))
            
            # Evitar seleccionar un índice que ya esté en idx (para no colapsar el simplex)
            row_ratios = vol_ratios[i, :].copy()
            row_ratios[list(idx)] = -1.0
            best = np.argmax(row_ratios)
            
            if row_ratios[best] > 1.0001:
                idx[i] = best
                E[:, i] = transform_h[best, :]

    return M_2d[idx, :]


def turbo_fcls_map(M, U, iterations=15):
    """
    Vectorized FCLS abundance estimation with ANC+ASC constraints.
    Returns abundances of shape (rows, cols, q).
    """
    rows, cols, bands = M.shape
    M_2d = M.reshape(-1, bands)
    q = U.shape[0]

    inv_UtU = np.linalg.pinv(np.dot(U, U.T))
    abundances = np.dot(inv_UtU, np.dot(U, M_2d.T)).T

    for _ in range(iterations):
        abundances = np.maximum(abundances, 0)
        sums = np.sum(abundances, axis=1, keepdims=True)
        sums[sums == 0] = 1
        abundances = abundances / sums

    return abundances.reshape(rows, cols, q)


from .metrics import compute_aRMSE, compute_SID, compute_ERGAS  # noqa: F401 — re-exported


def reconstruct_and_compute_metrics(original, reconstructed):
    """Returns (rmse, sam_degrees)."""
    rmse = float(np.sqrt(np.mean((original - reconstructed) ** 2)))

    orig_norm = np.linalg.norm(original, axis=2)
    rec_norm = np.linalg.norm(reconstructed, axis=2)
    orig_norm[orig_norm == 0] = 1e-10
    rec_norm[rec_norm == 0] = 1e-10

    dot_product = np.sum(original * reconstructed, axis=2)
    cos_theta = np.clip(dot_product / (orig_norm * rec_norm), -1.0, 1.0)
    sam_deg = float(np.mean(np.arccos(cos_theta)) * (180 / np.pi))

    return rmse, sam_deg
