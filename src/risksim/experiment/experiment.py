from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from risksim.risk import RiskMeasure
from risksim.copula import CopulaConfig, CopulaSimulator


@dataclass
class RiskExperimentConfig:
    runs: int = 10
    sim_samples: int = 10000
    alpha: float = 0.1
    gamma: float = 0.5


class RiskExperiment:
    """Run repeated copula simulations and compute risk measures using RiskMeasure.

    The class returns a pandas DataFrame with VaR, CVaR and Power across runs.
    """

    def __init__(
            self,
            copula_config: CopulaConfig,
            experiment_config: Optional[RiskExperimentConfig] = None,
            rng: Optional[np.random.Generator] = None
    ):
        self.copula_config = copula_config
        self.ex_config = experiment_config or RiskExperimentConfig()
        self.rng = rng or np.random.default_rng()

    def run(self) -> pd.DataFrame:
        vaR_list = []
        cvaR_list = []
        psrm_list = []
        all_sums = []

        for i in range(self.ex_config.runs):
            cfg = CopulaConfig(
                n_runs=self.ex_config.sim_samples,
                x_range=self.copula_config.x_range,
                y_range=self.copula_config.y_range,
                mu=self.copula_config.mu,
                std=self.copula_config.std,
                corr=self.copula_config.corr,
            )
            sim = CopulaSimulator(cfg, rng=self.rng)
            results = sim.run(full_log=False)
            sums = results["sum"]
            all_sums.append(sums)

            rm = RiskMeasure(
                sums,
                alpha=self.ex_config.alpha,
                gamma=self.ex_config.gamma)
            vaR_list.append(rm.VaR())
            cvaR_list.append(rm.CVaR())
            psrm_list.append(rm.power())

        df = pd.DataFrame(
            {"VaR": vaR_list, "CVaR": cvaR_list, "Power": psrm_list})

        # store some aggregates on the returned frame for convenience
        df.attrs["all_sums"] = all_sums
        return df
