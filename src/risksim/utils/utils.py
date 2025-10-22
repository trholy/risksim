import numpy as np
from typing import Sequence


def var_covar_matrix(
        std: Sequence[float],
        corr: Sequence[float]
) -> np.ndarray:
    """Construct a variance-covariance matrix for 2-dim case.

    The original code assumed a bivariate correlation list containing a single
    correlation coefficient. This function raises a ValueError for incorrect
    shapes and returns a 2x2 covariance matrix.
    """
    std = np.asarray(std, dtype=float)
    if std.shape != (2,):
        raise ValueError("std must be a sequence of length 2 for bivariate case")

    if len(corr) != 1:
        raise ValueError("corr must be a sequence with exactly one correlation value")

    rho = float(corr[0])
    corr_matrix = np.array([[1.0, rho], [rho, 1.0]])
    cov = np.diag(std) @ corr_matrix @ np.diag(std)
    return cov


def cholesky_decomp(matrix: np.ndarray) -> np.ndarray:
    """Return the lower-triangular Cholesky factor of a positive-definite matrix.

    Raises a informative error if the matrix is not positive-definite.
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "matrix is not positive-definite for Cholesky decomposition") from exc
