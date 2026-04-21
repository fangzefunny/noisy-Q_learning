import os
import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt 

# my visualize package 
from viz import viz 
viz.get_style()
from utils.env_fn import *

pth = os.path.dirname(os.path.abspath(__file__))
figures_pth = os.path.join(pth, "figures")
if not os.path.exists(figures_pth): os.makedirs(figures_pth)

"""Replicate the task setting: Trying to replicate the results in Fig. 1B."""

# load the environemnt 
exp_config = restless_bandit_config(
    condition = "complete",
    n_trials = 56,
    n_blocks = 1,
    reward_min = 0.01,
    reward_max = 0.99,
    tau = 3.0,
    omega = 1.5,
    seed = 2,
    mode = "for_anlaysis",
)

# generate the task 
env = restless_bandit(exp_config)
task_df = env.instan()

# show the reward random walk,
# replicate experiment 1 like Fig. 1B in the paper 
fig, axs = plt.subplots(1, 1, figsize=(5.2, 2))
ax = axs 
# plot the r_mean
sns.lineplot(data=task_df, 
    x="trial", 
    y="r_mean_0", 
    label="r_mean_A",
    color=viz.Blue, 
    linewidth=2.5,
    ax=ax)
sns.lineplot(data=task_df, 
    x="trial", 
    y="r_mean_1", 
    label="r_mean_B",
    color=viz.Red, 
    linewidth=2.5,
    ax=ax)
# plot the sampled reward
sns.scatterplot(data=task_df, 
    x="trial", 
    y="r_sampled_0", 
    label="r_sampled_A",
    color=viz.Blue, 
    alpha=0.6,
    edgecolor="none",
    s=10,
    ax=ax)
sns.scatterplot(data=task_df, 
    x="trial", 
    y="r_sampled_1", 
    label="r_sampled_B",
    color=viz.Red, 
    alpha=0.6,
    edgecolor="none",
    s=10,
    ax=ax)
ax.set_ylabel("Reward")
ax.set_xlabel("Trial")
# add legend 
ax.legend(loc="upper right", bbox_to_anchor=(1.6, 0.9))
fig.tight_layout(rect=[0, 0, 0.6, 0.9])  
fig.subplots_adjust(right=0.6)
ax.set_title("Reward Random Walk")
plt.savefig(f"figures/fig1_restless_bandit_task.pdf", dpi=300)

