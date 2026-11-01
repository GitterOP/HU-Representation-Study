"""
Generic Ground Truth Dataset Loader for hyperspectral unmixing.

Supports any dataset with (Y, M, A) structure stored in .mat files.
New datasets require only config.yaml edits — no code changes.

Supported datasets (out of the box):
    jasper  → Jasper Ridge (100×100, 198 bands, 4 materials)
    urban   → Urban        (307×307, 162 bands, 4 materials)
    samson  → Samson       (95×95,   156 bands, 3 materials)
    apex    → APEX         (111×122, 285 bands, 4 materials, no endmember GT)

Usage
-----
    from src.dataset_loaders_gt import GroundTruthDataset
    from src.config import load_config

    config = load_config('config.yaml')
    dataset = GroundTruthDataset(config['datasets']['jasper'])
    Y, M, A, metadata = dataset.load()

    # Y : (bands, pixels)       float64, range [0, 1]
    # M : (bands, materials)    float64 or None if unavailable
    # A : (materials, pixels)   float64 or None if unavailable
    # metadata : dict with name, shape, paths, normalization info
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import scipy.io as sio


# ---------------------------------------------------------------------------
# Standalone helper
# ---------------------------------------------------------------------------

def reshape_abundances(A: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Reshape abundance matrix to a spatial 3-D array.

    Parameters
    ----------
    A : ndarray of shape (materials, pixels)
        Flat abundance matrix, one row per material.
    rows : int
        Number of image rows.
    cols : int
        Number of image columns.

    Returns
    -------
    A_3d : ndarray of shape (rows, cols, materials)
        Spatially arranged abundances suitable for imshow.

    Examples
    --------
    >>> A_3d = reshape_abundances(A, rows=100, cols=100)
    >>> plt.imshow(A_3d[:, :, 0], cmap='jet')
    """
    materials = A.shape[0]
    return A.reshape(materials, rows, cols).transpose(1, 2, 0)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GroundTruthDataset:
    """Generic loader for hyperspectral datasets with ground-truth (Y, M, A).

    Parameters
    ----------
    dataset_config : dict
        Configuration block from config.yaml.  Expected structure::

            name       : 'Jasper Ridge'
            image_path : 'data/processed_data/jasper/jasperRidge2_R198.mat'
            gt_path    : 'data/processed_data/jasper/end4.mat'   # optional
            mat_keys:
              Y : 'Y'    # key for image inside image_path file
              M : 'M'    # key for endmembers inside gt_path file (null → not available)
              A : 'A'    # key for abundances inside gt_path file (null → not available)
            shape:
              rows      : 100
              cols      : 100
              bands     : 198
              materials : 4

    Notes
    -----
    - Y is normalised to [0, 1] by dividing by its maximum value.
    - 3-D arrays (rows, cols, bands) are automatically reshaped to 2-D.
    - M or A may be ``None`` when the key is ``null`` in config.yaml.
    """

    def __init__(self, dataset_config: dict) -> None:
        self._cfg = dataset_config
        self.name: str = dataset_config['name']
        self.image_path: str = dataset_config['image_path']
        self.gt_path: Optional[str] = dataset_config.get('gt_path')
        self.mat_keys: dict = dataset_config['mat_keys']
        self.shape: dict = dataset_config['shape']

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], dict]:
        """Load dataset and return (Y, M, A, metadata).

        Returns
        -------
        Y : ndarray, shape (bands, pixels)
            Hyperspectral image, normalised to [0, 1].
        M : ndarray, shape (bands, materials), or None
            Ground-truth endmember signatures.
        A : ndarray, shape (materials, pixels), or None
            Ground-truth abundance fractions.
        metadata : dict
            Keys: ``name``, ``shape``, ``image_path``, ``gt_path``,
            ``Y_max_before_norm``.

        Raises
        ------
        FileNotFoundError
            If image_path or gt_path does not exist.
        KeyError
            If a requested mat key is not present in the file.
        ValueError
            If array shapes do not match the declared config dimensions.

        Examples
        --------
        >>> dataset = GroundTruthDataset(config['datasets']['jasper'])
        >>> Y, M, A, meta = dataset.load()
        >>> print(Y.shape)   # (198, 10000)
        >>> print(M.shape)   # (198, 4)
        >>> print(A.shape)   # (4, 10000)
        """
        # --- Image (Y) ------------------------------------------------
        img_mat = self._load_mat(self.image_path)
        Y_raw = self._extract_array(img_mat, self.mat_keys['Y'])
        Y_raw = self._to_2d_Y(Y_raw)

        y_max = float(Y_raw.max())
        Y = Y_raw / y_max if y_max > 1.0 else Y_raw.copy()

        # --- Ground Truth (M, A) -------------------------------------
        M: Optional[np.ndarray] = None
        A: Optional[np.ndarray] = None

        if self.gt_path is not None:
            gt_mat = self._load_mat(self.gt_path)

            m_key = self.mat_keys.get('M')
            if m_key:
                M = self._extract_array(gt_mat, m_key)
                M = self._to_2d_M(M)

            a_key = self.mat_keys.get('A')
            if a_key:
                A = self._extract_array(gt_mat, a_key)
                A = self._to_2d_A(A)

        # --- Validate -------------------------------------------------
        self._validate_shapes(Y, M, A)

        metadata = {
            'name': self.name,
            'shape': self.shape,
            'image_path': self.image_path,
            'gt_path': self.gt_path,
            'Y_max_before_norm': y_max,
        }
        return Y, M, A, metadata

    def validate_shapes(
        self,
        Y: np.ndarray,
        M: Optional[np.ndarray],
        A: Optional[np.ndarray],
    ) -> None:
        """Public wrapper — raise ValueError if shapes are wrong."""
        self._validate_shapes(Y, M, A)

    def reshape_abundances(self, A: Optional[np.ndarray] = None) -> np.ndarray:
        """Reshape A to (rows, cols, materials) using this dataset's dimensions.

        Parameters
        ----------
        A : ndarray of shape (materials, pixels), optional
            If omitted you must pass it explicitly; this is a convenience
            wrapper around the module-level :func:`reshape_abundances`.

        Returns
        -------
        ndarray of shape (rows, cols, materials)
        """
        if A is None:
            raise ValueError("Pass the abundance matrix A as argument.")
        rows = self.shape['rows']
        cols = self.shape['cols']
        return reshape_abundances(A, rows, cols)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_mat(self, path: str) -> dict:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"Dataset file not found: {abs_path}\n"
                f"Check 'image_path' / 'gt_path' in config.yaml for '{self.name}'."
            )
        return sio.loadmat(abs_path)

    def _extract_array(self, mat_dict: dict, key: str) -> np.ndarray:
        if key not in mat_dict:
            available = sorted(k for k in mat_dict if not k.startswith('__'))
            raise KeyError(
                f"Key '{key}' not found in .mat file for dataset '{self.name}'.\n"
                f"Available keys: {available}\n"
                f"Update 'mat_keys' in config.yaml."
            )
        return np.array(mat_dict[key], dtype=np.float64)

    @staticmethod
    def _to_2d_Y(Y: np.ndarray) -> np.ndarray:
        """Ensure Y has shape (bands, pixels)."""
        if Y.ndim == 3:
            # Assume (rows, cols, bands)
            rows, cols, bands = Y.shape
            return Y.reshape(rows * cols, bands).T          # → (bands, pixels)
        if Y.ndim == 2:
            return Y                                        # already (bands, pixels)
        raise ValueError(f"Unexpected Y dimensionality: {Y.ndim}D ({Y.shape}).")

    @staticmethod
    def _to_2d_M(M: np.ndarray) -> np.ndarray:
        """Ensure M has shape (bands, materials)."""
        if M.ndim == 2:
            return M
        raise ValueError(f"Unexpected M dimensionality: {M.ndim}D ({M.shape}).")

    @staticmethod
    def _to_2d_A(A: np.ndarray) -> np.ndarray:
        """Ensure A has shape (materials, pixels)."""
        if A.ndim == 3:
            # Assume (rows, cols, materials)
            rows, cols, mats = A.shape
            return A.reshape(rows * cols, mats).T           # → (materials, pixels)
        if A.ndim == 2:
            return A                                        # already (materials, pixels)
        raise ValueError(f"Unexpected A dimensionality: {A.ndim}D ({A.shape}).")

    def _validate_shapes(
        self,
        Y: np.ndarray,
        M: Optional[np.ndarray],
        A: Optional[np.ndarray],
    ) -> None:
        rows = self.shape['rows']
        cols = self.shape['cols']
        bands = self.shape['bands']
        materials = self.shape['materials']
        n_pixels = rows * cols

        expected_Y = (bands, n_pixels)
        if Y.shape != expected_Y:
            raise ValueError(
                f"[{self.name}] Y shape mismatch: "
                f"expected {expected_Y}, got {Y.shape}.\n"
                f"Check 'shape' in config.yaml."
            )

        if M is not None:
            expected_M = (bands, materials)
            if M.shape != expected_M:
                raise ValueError(
                    f"[{self.name}] M shape mismatch: "
                    f"expected {expected_M}, got {M.shape}."
                )

        if A is not None:
            expected_A = (materials, n_pixels)
            if A.shape != expected_A:
                raise ValueError(
                    f"[{self.name}] A shape mismatch: "
                    f"expected {expected_A}, got {A.shape}."
                )
