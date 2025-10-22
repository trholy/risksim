import numpy as np
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class RiskMeasure:
    """
    A class for computing common financial risk measures.

    Parameters
    ----------
    data : Sequence[float]
        The sample of returns (losses or gains).
    alpha : float, optional
        Confidence level for Value-at-Risk and Conditional VaR (default = 0.1).
    gamma : float, optional
        Exponent parameter for the Power Spectral Risk Measure (default = 0.5).
    """

    data: Sequence[float]
    alpha: float = 0.1
    gamma: float = 0.5
    _sorted_data: np.ndarray = field(init=False, repr=False)
    _n: int = field(init=False, repr=False)

    def __post_init__(self):
        self._sorted_data = np.sort(np.asarray(self.data))
        self._n = len(self._sorted_data)

    @property
    def expected_value(self) -> float:
        """Expected (mean) value of the dataset."""
        return float(np.mean(self._sorted_data))

    def var(self) -> float:
        """Variance of the dataset."""
        return float(np.var(self._sorted_data, ddof=0))

    def std(self) -> float:
        """Standard deviation of the dataset."""
        return float(np.std(self._sorted_data, ddof=0))

    def VaR(self) -> float:
        """
        Compute Value-at-Risk (VaR) at level `alpha`.

        Returns
        -------
        float
            Value-at-Risk (negative quantile).
        """
        idx = int(np.floor(self.alpha * self._n)) - 1
        idx = np.clip(idx, 0, self._n - 1)
        var_value = -self._sorted_data[idx]
        return float(var_value)

    def CVaR(self) -> float:
        """
        Compute Conditional Value-at-Risk (CVaR, a.k.a. Expected Shortfall).

        Returns
        -------
        float
            Conditional Value-at-Risk (negative mean of the worst α fraction).
        """
        cutoff = int(np.floor(self.alpha * self._n))
        cutoff = max(cutoff, 1)
        tail_losses = self._sorted_data[:cutoff]
        cvar_value = -float(np.mean(tail_losses))
        return cvar_value

    def power(self) -> float:
        """
        Compute Power Spectral Risk Measure (PSRM).

        Returns
        -------
        float
            Power spectral risk measure value.
        """
        n = self._n
        weights = np.power(np.arange(1, n + 1) / n, self.gamma) - np.power(
            np.arange(0, n) / n, self.gamma)
        # Ensure weights sum to 1 (numerically stable)
        weights /= np.sum(weights)
        return float(np.dot(self._sorted_data, weights))
