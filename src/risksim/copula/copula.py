from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from risksim.utils import var_covar_matrix, cholesky_decomp

logger = logging.getLogger(__name__)


@dataclass
class CopulaConfig:
    """Configuration for a bivariate Gaussian copula simulation.

    Attributes:
        n_runs: number of Monte-Carlo samples
        x_range: (min, max) for uniform marginal of X
        y_range: (min, max) for uniform marginal of Y
        mu: location shift applied to the correlated normals (length 2)
        std: standard deviations for the two normals (length 2)
        corr: sequence with single correlation coefficient (rho)
    """

    n_runs: int
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    mu: Sequence[float]
    std: Sequence[float]
    corr: Sequence[float]


class CopulaSimulator:
    """Simulate dependent marginals via a Gaussian copula.

    The class is intentionally minimal and focuses on the bivariate case because
    the original code was bivariate. It exposes a reproducible `run` method that
    returns a dictionary with requested outputs.
    """

    def __init__(
            self,
            config: CopulaConfig,
            rng: Optional[np.random.Generator] = None
    ):
        self.config = config
        self.rng = rng or np.random.default_rng()

        # validate shapes
        self.mu = np.asarray(config.mu, dtype=float)
        self.std = np.asarray(config.std, dtype=float)
        if self.mu.shape != (2,) or self.std.shape != (2,):
            raise ValueError("mu and std must be sequences of length 2")

        # prepare covariance and decomposition
        self._cov = var_covar_matrix(self.std, config.corr)
        self._chol = cholesky_decomp(self._cov)

    def run(
            self,
            *,
            full_log: bool = False,
            debug: bool = False
    ) -> Mapping[str, np.ndarray]:
        """Execute the Monte Carlo simulation and return a results dictionary.

        Returned keys:
            - 'sum': X + Y array
            - 'xy': stacked columns [X, Y]
        Additional keys when full_log=True:
            - 'standard_norm_ab': correlated normals after Cholesky
            - 'copula_realisation': correlated uniforms from the Gaussian copula
        """
        n = int(self.config.n_runs)

        # 1) independent uniforms
        u = self.rng.random((n, 2))
        if debug:
            logger.debug(
                "independent uniforms sample[0]=%s", u[0].tolist())

        # 2) transform to independent standard normals
        z_indep = norm.ppf(u)
        if debug:
            logger.debug(
                "independent normals sample[0]=%s", z_indep[0].tolist())

        # 3) correlated standard normals
        z_corr = z_indep @ self._chol.T + self.mu
        if debug:
            logger.debug(
                "correlated normals sample[0]=%s", z_corr[0].tolist())

        # 4) copula uniforms
        var_list = np.square(self.std)
        u_copula = norm.cdf((z_corr - self.mu) / np.sqrt(var_list))
        if debug:
            logger.debug(
                "copula uniforms sample[0]=%s", u_copula[0].tolist())

        # 5) apply uniform marginals
        x = (self.config.x_range[1] - self.config.x_range[0]) * u_copula[:, 0] + self.config.x_range[0]
        y = (self.config.y_range[1] - self.config.y_range[0]) * u_copula[:, 1] + self.config.y_range[0]
        total = x + y

        results: MutableMapping[str, np.ndarray] = {
            "sum": total,
            "xy": np.column_stack((x, y))
        }
        if full_log:
            results.update(
                {"standard_norm_ab": z_corr,
                 "copula_realisation": u_copula})

        return results
