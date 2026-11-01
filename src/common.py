import numpy as np
from scipy.stats import norm

def hyperHfcVd_Adjusted(M, far=1e-5):
    """
    Estimación de la Dimensionalidad Virtual (VD) mediante algoritmo HFC.
    Versión ajustada para datos normalizados [0, 1].
    M: Matriz de datos (Bands x Pixels) o (Rows x Cols x Bands)
    far: False Alarm Rate (Tasa de falsa alarma)
    """
    if len(M.shape) == 3:
        rows, cols, bands = M.shape
        M_2d = M.reshape(-1, bands).T
    else:
        M_2d = M
        bands, N_pixels = M_2d.shape

    numBands, N = M_2d.shape
    
    # Cálculo de matrices de Covarianza y Correlación
    cov_m = np.cov(M_2d)
    corr_m = (M_2d @ M_2d.T) / N
    
    # Autovalores reales ordenados desc
    lambdaCov = np.sort(np.real(np.linalg.eigvals(cov_m)))[::-1]
    lambdaCorr = np.sort(np.real(np.linalg.eigvals(corr_m)))[::-1]
    
    numEndmembers = 0
    for x in range(numBands):
        # Ajuste para datos normalizados: uso de cuadrados de autovalores
        s2 = (2*(lambdaCov[x]**2)/N) + (2*(lambdaCorr[x]**2)/N)
        sigma = np.sqrt(max(s2, 1e-15))
        
        # Test estadístico (Neyman-Pearson)
        tau = -norm.ppf(far, 0, sigma)
        if (lambdaCorr[x] - lambdaCov[x]) > tau:
            numEndmembers += 1
            
    return numEndmembers
