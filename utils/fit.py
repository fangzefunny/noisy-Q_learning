from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

def systematic_resampling(
    weights: np.ndarray,
    rng: np.random.Generator
    ) -> np.ndarray:

    # calculate the cumsum for the original weights
    n = weights.shape[0]
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0 

    # propose a new cumsum 
    u0 = rng.random() / n 
    proposal = u0 + np.arange(n) / n 

    # reject the bad particles that worse
    # than proposal 
    ind = np.zeros([n,], dtype=np.intc)
    i, j = 0, 0 
    while i < n: 
        # if the proposal is less than the cumsum, 
        # then accept the particle
        if proposal[i] < cumsum[j]:
            ind[i] = j
            i += 1
        else:
            j += 1

    return ind

def estimate_nll_pf(
    prb_data: dict,
    agent_cls,
    params: list,
    n_particles: int=200,
    seed: Optional[int]=42,
    ) -> float:
    
    """Estimation the negative log likelihood using Particle filter

        This implementation is based on:
        https://github.com/csmfindling/learning_variability/blob/master/lib_c/state_estimation/smc_functions.cpp

        I summarize the fitting process as:
            1. initialize the particles and weights
            for each trial:
                2. update the latent for each particles (propagate)
                3. compute log p(a|Z), and use it as the weight of the particle (normalize)
                4. calculate the marginal likelihood of the trial (mean of the weights)
                5. resample the particles to avoid degeneracy
    """

    # init the model and particles
    model = agent_cls()
    config = model.load_configs(params)
    particles = []
    for _ in range(n_particles):
        particle = {"q": model._init_model(config)}
        particles.append(particle)
    log_weights = np.zeros(n_particles)

    seg_keys = sorted(prb_data.keys())
    rng = np.random.default_rng(seed)
    prev_a, r_sample = 99, None 
    ll = 0.0
    for seg_key in seg_keys:
        
        # obtain the trial data
        seg_data = prb_data[seg_key]
        trial = seg_data["trial"]
        a = int(seg_data["a"])
        r_sampled_0 = seg_data["r_sampled_0"]
        r_sampled_1 = seg_data["r_sampled_1"]
      
        # for each particle, calculate the log-likelihood
        # as the weight of the particle
        for i, particle in enumerate(particles):

            # if not the first trial, then update the latent variable
            if trial > 0: 
                particle["q"] = model.update_latent(
                    particle["q"],
                    r_sample,
                    config,
                    rng,
                )
            # get stable log policy for the observed action
            log_policy = model.log_policy(particle["q"], prev_a, config)
            log_weights[i] = log_policy[a]
            
        # exp and normalize the weights
        b = np.max(log_weights)
        unnorm_weights = np.exp(log_weights - b)
        weights_sum = np.sum(unnorm_weights)
        ll += b + np.log(weights_sum) - np.log(n_particles)
        # normalize the weights
        weights = unnorm_weights / weights_sum

        # update the prev_a
        prev_a = a 
        r_sample = np.array([r_sampled_0, r_sampled_1])

        # resample the particles
        ind = systematic_resampling(weights, rng)
        particles = [deepcopy(particles[i]) for i in ind]

    return -ll 

def loss_ln_pf(
    sub_data: dict,
    agent_cls,
    params: list,
    n_particles: int=200,
    seed: Optional[int]=42,
    ) -> float:

    """Loss function for the particle filter.

        This function calculates the negative log likelihood of the data
        using the particle filter.
    """
    loss = 0.0 
    for _, prb_data in sub_data.items():
        nll = estimate_nll_pf(
            prb_data, 
            agent_cls, 
            params, 
            n_particles, 
            seed)
        loss += nll
    return loss

