from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


class PlotUtils:
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
