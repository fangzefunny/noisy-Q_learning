"""Noisy Q-learning model for volatile bandit simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class model_config:
    alpha: float
    eta: float
    beta: float
    sigma: float

class noisy_q_learning:
    name = "noisy_q_learning"
    p_names = ["alpha", "eta", "beta", "sigma"]
    p_bnds  = [(0.0, .99), (0.0, 50.0), (0.0, 50.0), (0.0, 1.0)]
    p_pbnds = [(0.1, 0.5), (0.5, 2.0), (0.5, 5.0), (0.1, 0.5)]
    n_param = len(p_names)

    def __init__(self, ) -> None:
        pass 

    @staticmethod
    def load_configs(params) -> model_config:
        
        """Load the model configurations from a dictionary."""
        
        return model_config(
            alpha = params[0],
            eta   = params[1],
            beta  = params[2],
            sigma = params[3],
        )

    @staticmethod
    def _init_model(config: model_config) -> np.ndarray:

        """init the model to output the latent variable"""

        Q = np.ones([2,])*50
        return Q


    @staticmethod
    def policy(
        Q: np.ndarray,
        prev_a: int,
        config: model_config, 
        ) -> np.ndarray:

        """The policy: p(a|z, theta), 
        
            The action distribution given the latent variable.
            Here the latent is z. We use a sigmoid function with perseveration
            to generate the probabability of choosing B. 

            logit = beta * (qB - qA) + eta * per
            prob  = 1 / (1 + np.exp(-logit))
            policy = [1-prob, prob]
        """
        # calculate the logits
        q_diff = Q[1] - Q[0]

        # calculate the perseveration 
        if prev_a != 99: # if there is a previous action
            # choose A: act = 0, choose B: action = 1
            # choose A: then perseverate -1
            # choose B: then perseverate +1
            per = -1 if prev_a==0 else 1  
        else:
            per = 0 
        
        # get the probability of choosing B
        logit = config.beta * q_diff + config.eta*per
        prob  = 1 / (1 + np.exp(-logit))
        
        # build a bernoulli distribution 
        return np.array([1-prob, prob])

    @staticmethod
    def log_policy(
        Q: np.ndarray,
        prev_a: int,
        config: model_config,
        ) -> np.ndarray:

        """Stable log probabilities for the two actions.

            This computes log P(action|Q, theta) directly from the logit,
            avoiding the numerical issue where sigmoid(logit) underflows to
            exactly 0 and log(0) becomes -inf.
        """

        # calculate the logits
        q_diff = Q[1] - Q[0]

        # calculate the perseveration 
        if prev_a != 99: # if there is a previous action
            # choose A: act = 0, choose B: action = 1
            # choose A: then perseverate -1
            # choose B: then perseverate +1
            per = -1 if prev_a==0 else 1  
        else:
            per = 0 

        logit = config.beta * q_diff + config.eta * per

        # P(B) = sigmoid(logit), P(A) = sigmoid(-logit).
        log_p_b = -np.logaddexp(0.0, -logit)
        log_p_a = -np.logaddexp(0.0, logit)

        # build a bernoulli distribution 
        return np.array([log_p_a, log_p_b])

    @staticmethod
    def update_latent(
        Q: np.ndarray,
        r_sampled: np.ndarray,
        config: model_config,
        rng: np.random.Generator,
        ) -> np.ndarray:

        """Update the latent variable based on the action and reward.
        
            p(Z_new|Z, a, r, theta). The update uses noisy Q leanring rule 
            Q_new = Q + alpha * (r - Q) + noise
            noise ~ N(0, sigma)
            sigma = |r_sampled - Q| * config.sigma
        """
        sigma = np.abs(r_sampled - Q) * config.sigma
        noise = np.array([
            rng.normal(0.0, sigma[0]),
            rng.normal(0.0, sigma[1]),
        ])
        Q_new = Q + config.alpha * (r_sampled - Q) + noise 
        return Q_new 

def simulate(
    agent: object,
    params: list, 
    prb_data: dict, 
    seed: Optional[int]=42,
    ) -> dict:

    # load the agent as a subject
    subj = agent()
    config = subj.load_configs(params)
    Q = subj._init_model(config)
    rng = np.random.default_rng(seed)

    # simulate the agent    
    trials = sorted(prb_data.keys())
    prb_data_new = {}
    prev_a, r_sample = 99, None
    for t in trials:

        # get the trial data 
        seg_data = prb_data[t]

        # if not the first trial, then update the latent
        # variable, Q 
        if t > 0: 
            Q = subj.update_latent(
                Q, 
                r_sample, # do not exists in the first trial  
                config, 
                rng)
        
        # get the policy
        log_policy = subj.log_policy(Q, prev_a, config)
        a = rng.choice(2, p=np.exp(log_policy))
        # shallow copy: only "a" is new; other values share references with prb_data[t]
        seg_data_new = {**seg_data, "a": a, "qA": float(Q[0]), "qB": float(Q[1])}
        prev_a = a

        # get the feedback for next iteration update
        r_sample_0 = seg_data["r_sampled_0"]
        r_sample_1 = seg_data["r_sampled_1"]
        r_sample = np.array([r_sample_0, r_sample_1])     

        # store the data
        prb_data_new[t] = seg_data_new
    
    return prb_data_new


def prb_data_to_dataframe(prb_data: dict) -> pd.DataFrame:
    """Convert trial-keyed segment dicts (e.g. simulate output) to one row per trial."""
    trials = sorted(prb_data.keys())
    return pd.DataFrame([prb_data[t] for t in trials])
