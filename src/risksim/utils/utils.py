from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def _var_covar_matrix(
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


def _cholesky_decomp(matrix: np.ndarray) -> np.ndarray:
    """Return the lower-triangular Cholesky factor of a positive-definite matrix.

    Raises a informative error if the matrix is not positive-definite.
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "matrix is not positive-definite for Cholesky decomposition") from exc


@dataclass
class StockDataLoader:
    """Loads and prepares stock price data from CSV files."""

    tickers: list[str]
    data_dir: str | Path = field(default=Path("../datasets"))

    def __post_init__(self):
        """Ensure path is a Path object and exists."""
        self.data_dir = Path(self.data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}")

    def load_data(self) -> dict[str, pd.DataFrame]:
        """Load and clean data for all tickers."""
        data = {}
        for ticker in self.tickers:
            file_path = self.data_dir / f"{ticker}.csv"
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Missing data file for ticker: {ticker}")

            df = pd.read_csv(
                file_path,
                sep=",",
                decimal=".",
                usecols=["Date", "Adj Close"])
            df["Date"] = pd.to_datetime(df["Date"])
            df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
            data[ticker] = df.dropna(subset=["Adj Close"])
        return data


@dataclass
class PortfolioAnalyzer:
    """Performs return calculations, correlation, and portfolio statistics."""

    price_data: dict[str, pd.DataFrame]
    return_df: pd.DataFrame = field(init=False)

    def __post_init__(self):
        """Compute combined return chart after data is loaded."""
        self.return_df = self._build_price_return_df()

    def _build_price_return_df(self) -> pd.DataFrame:
        """Combine prices and returns into one DataFrame indexed by Date."""
        return_df = pd.DataFrame()
        for ticker, df in self.price_data.items():
            return_df[f"Price {ticker}"] = df["Adj Close"]
            return_df[f"Return {ticker}"] = df["Adj Close"].pct_change()
        return_df["Date"] = next(iter(self.price_data.values()))["Date"]
        return_df.set_index("Date", inplace=True)
        return_df["Return PF"] = return_df.filter(like="Return").mean(axis=1)
        return_df = return_df.multiply(100)
        return return_df

    def get_returns(self, ticker: str) -> pd.Series:
        """Return aligned and cleaned pandas Series of returns (no zero filtering)."""
        return self.return_df[f"Return {ticker}"].dropna()

    def get_portfolio_returns(self) -> pd.Series:
        """Return cleaned pandas Series of portfolio returns."""
        return self.return_df["Return PF"].dropna()

    def calculate_statistics(self, values: np.ndarray) -> tuple[float, float]:
        """Return mean and standard deviation."""
        return float(np.mean(values)), float(np.std(values))

    def calculate_correlation(self, ticker1: str, ticker2: str) -> float:
        """Compute correlation between two assets based on overlapping dates."""
        s1 = self.get_returns(ticker1)
        s2 = self.get_returns(ticker2)

        aligned = pd.concat([s1, s2], axis=1, join="inner").dropna()
        aligned.columns = [ticker1, ticker2]

        if aligned.empty:
            raise ValueError(
                f"No overlapping data between {ticker1} and {ticker2}.")

        return float(aligned.corr().iloc[0, 1])
