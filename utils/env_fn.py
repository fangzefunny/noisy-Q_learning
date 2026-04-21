"""Task generators for reward-guided learning simulations.

The primary task here is the restless two-armed bandit from Findling et al.
(2019): each option's latent mean payoff follows a beta random walk, and trial
payoffs are sampled from beta distributions around those drifting means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

Condition = Literal["partial", "complete"]
TaskType = Literal["restless", "reversal"]

@dataclass(frozen=True)
class restless_bandit_config:
    condition: Condition="partial"
    n_trials: int=56
    n_blocks: int=1
    reward_min: float=0.01
    reward_max: float=0.99
    tau: float=3.0
    omega: float=1.5
    mode: str="for_fit" # for_fit or for_analysis 
    seed: Optional[int]=42

class restless_bandit:
    """Generate drifting reward schedules for a two-armed bandit task.
    
        This environment generates a restless bandit task with two bandit arms
        whose reward means drift over time. The drift is generated according
        to the following equation:
            alpha = 1 + r_means[t-1]*np.exp(tau)
            beta  = 1 + (1-r_means[t-1])*np.exp(tau)
            r_mean[t] = Beta(alpha, beta) 

        The sampled reward is generated based on a Beta distribution 
        based on the current r_mean.
            alpha = 1 + r_means[t]*np.exp(omega)
            beta  = 1 + (1-r_means[t])*np.exp(omega)
            reward[t] = Beta(alpha, beta)

        The reward is then normalized to the range of [.01, .99].
    """

    def __init__(self, config: restless_bandit_config) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def instan(self) -> pd.DataFrame:
        # generate r_mean [n_trials, 2]
        r_means = self._random_walk_means(
            self.config.n_trials,
            self.config.tau,
            self.rng,
        )
        # generate r_sampled [n_trials, 2]
        r_sampled = self._sample_rewards(
            r_means, 
            self.config.omega, 
            self.rng, 
        )
        # generate reward [n_trials, 2]
        df ={
            "r_mean_0": (r_means[:, 0]*100).astype(int),
            "r_mean_1": (r_means[:, 1]*100).astype(int),
            "r_sampled_0": (r_sampled[:, 0]*100).astype(int),
            "r_sampled_1": (r_sampled[:, 1]*100).astype(int),
            "trial": np.arange(0, self.config.n_trials),
            "condition": self.config.condition,
            "stage": "train"
        }
        if self.config.mode == "for_fit":
            for_fit = {}
            for i in range(self.config.n_trials):
                for_fit[i] = {
                    "r_mean_0": float(df["r_mean_0"][i]),
                    "r_mean_1": float(df["r_mean_1"][i]),
                    "r_sampled_0": float(df["r_sampled_0"][i]),
                    "r_sampled_1": float(df["r_sampled_1"][i]),
                    "trial": i,
                }
            return for_fit        
        else:
            return pd.DataFrame(df)

    @staticmethod
    def _random_walk_means(
        n_trials: int, 
        tau: float,
        rng: np.random.Generator,
        r_min: float=0.01,
        r_max: float=0.99,
        ) -> np.ndarray:
        
        # generate the trajectory of r_mean 
        r_mean0 = 0.5
        r_means = np.empty((n_trials, 2), dtype=float)
        for t in range(0, n_trials):
            for j in range(r_means.shape[1]):
                prev_r_mean = r_mean0 if t==0 else r_means[t-1, j]
                alpha = 1 + prev_r_mean*np.exp(tau)
                beta  = 1 + (1-prev_r_mean)*np.exp(tau)
                r_means[t, j] = rng.beta(alpha, beta)

        # clip the reward 
        r_means = np.clip(r_means, r_min, r_max)
        return r_means

    @staticmethod
    def _sample_rewards(
        r_means: np.ndarray,
        omega: float,
        rng: np.random.Generator,
        r_min: float=0.01,
        r_max: float=0.99,
    ) -> np.ndarray:

        # sample the reward based on the current r_mean 
        r_sampled = np.empty((r_means.shape[0], r_means.shape[1]), dtype=float)
        for t in range(0, r_means.shape[0]):
            for j in range(r_means.shape[1]):
                alpha = 1 + r_means[t, j]*np.exp(omega)
                beta  = 1 + (1-r_means[t, j])*np.exp(omega)
                r_sampled[t, j] = rng.beta(alpha, beta)

        # clip the reward 
        r_sampled = np.clip(r_sampled, r_min, r_max)
        return r_sampled

if __name__ == "__main__":

    env = restless_bandit(restless_bandit_config)
    task = env.instan()
