from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


class MonteCarloPlotUtils:
    """Small collection of plotting helper functions used by the example script.

    These are intentionally simple wrappers around matplotlib to keep the
    simulation logic free from plotting concerns (so plotting can be easily
    removed or replaced in tests).
    """

    @staticmethod
    def scatter(
            x: Iterable[float],
            y: Iterable[float],
            *,
            title: str,
            xlabel: str = "X",
            ylabel: str = "Y",
            grid: bool = True
    ) -> None:
        plt.figure(figsize=(6, 5))
        plt.scatter(list(x), list(y), s=10, alpha=0.6)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if grid:
            plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def histogram(
            data: Iterable[float],
            *,
            bins: int = 50,
            title: str = "Histogram",
            xlabel: str = "Value",
            ylabel: str = "Count",
            density: bool = False
    ) -> None:
        plt.figure(figsize=(6, 4))
        plt.hist(
            list(data), bins=bins,
            alpha=0.7, edgecolor="black", density=density)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_cdf_from_hist(hist_values: np.ndarray, bins: np.ndarray) -> None:
        dx = bins[1] - bins[0]
        cdf = np.cumsum(hist_values) * dx
        plt.plot(bins[1:], cdf)
        plt.xlabel("Value")
        plt.ylabel("Cumulative Probability")
        plt.grid(True)
        plt.show()

    @staticmethod
    def gauss_copula_realisation(
            u_copula: np.ndarray,
            *,
            title: str = "Gaussian Copula Realisations") -> None:
        """
        Plot the correlated uniform variables from a Gaussian copula simulation.

        Args:
            u_copula: np.ndarray of shape (n_samples, 2) — the copula realisations.
            title: plot title (defaults to German-style name used in the original script).
        """
        if u_copula.ndim != 2 or u_copula.shape[1] != 2:
            raise ValueError(
                "u_copula must be a 2D array with shape (n_samples, 2)")

        plt.figure(figsize=(6, 5))
        plt.scatter(u_copula[:, 0], u_copula[:, 1], s=10, alpha=0.6)
        plt.title(title)
        plt.xlabel("U₁ (copula realisation)")
        plt.ylabel("U₂ (copula realisation)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()


class PortfolioPlotter:
    """Handles all plotting functionality for portfolio analysis."""

    @staticmethod
    def scatter_returns(
            series_x: pd.Series,
            series_y: pd.Series,
            label_x: str,
            label_y: str
    ):
        """Plot joint distribution of two return series aligned by date index."""
        aligned = pd.concat([series_x, series_y], axis=1, join="inner").dropna()
        aligned.columns = [label_x, label_y]

        if aligned.empty:
            raise ValueError("No overlapping data to plot.")

        plt.scatter(aligned[label_x], aligned[label_y], alpha=0.7)
        plt.xlabel(f"Return {label_x}")
        plt.ylabel(f"Return {label_y}")
        plt.title(f"Joint Distribution: {label_x} vs {label_y}")
        PortfolioPlotter._add_grid()

    @staticmethod
    def compare_distributions(values_pf, mu_pf, std_pf, bins=50):
        """Plot empirical (historical) vs parametric (normal) CDFs."""
        hist, bin_edges = np.histogram(values_pf, bins=bins, density=True)
        dx = bin_edges[1] - bin_edges[0]
        F_hist = np.cumsum(hist) * dx
        plt.plot(bin_edges[1:], F_hist, label="Historical Simulation")

        x_range = np.linspace(values_pf.min(), values_pf.max(), bins)
        plt.plot(
            x_range,
            stats.norm.cdf(x_range, mu_pf, std_pf),
            label="Variance-Covariance Method")

        plt.xlabel("Return")
        plt.ylabel("Cumulative Probability")
        plt.title("Portfolio Return Distribution Comparison")
        plt.legend()
        PortfolioPlotter._add_grid()

    @staticmethod
    def _add_grid():
        plt.grid(True)
        plt.axhline(0, color="black")
        plt.axvline(0, color="black")
        plt.show()
