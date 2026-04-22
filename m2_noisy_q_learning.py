import os

from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt 

# my visualize package 
from viz import viz 
viz.get_style()
from utils.env_fn import *
from utils.model import *

pth = os.path.dirname(os.path.abspath(__file__))
figures_pth = os.path.join(pth, "figures")
if not os.path.exists(figures_pth): os.makedirs(figures_pth)

# load the environment
env = restless_bandit(restless_bandit_config(
    condition = "complete",
    n_trials = 56,
    n_blocks = 1,
    reward_min = 0.01,
    reward_max = 0.99,
    tau = 3.0,
    omega = 1.5,
    seed = 2,
    mode = "for_fit",
))
prb_data = env.instan()

# simulate the model 
agent = noisy_q_learning
params = [.2, .1, 4, .25]
seed = 1234
prb_data_sim = prb_data_to_dataframe(simulate(
    agent, 
    params, 
    prb_data,
    seed,
))

# plot the results  
fig, axs = plt.subplots(1, 1, figsize=(5.2, 2))
ax = axs 
# plot the r_mean (errorbar=None: CI ribbon adds extra legend handles and breaks label order)
sns.lineplot(data=prb_data_sim, 
    x="trial", 
    y="r_mean_0", 
    label="r_mean_A",
    color=viz.Blue, 
    linewidth=4,
    alpha=0.6,
    errorbar=None,
    ax=ax)
sns.lineplot(data=prb_data_sim, 
    x="trial", 
    y="r_mean_1", 
    label="r_mean_B",
    color=viz.Red, 
    linewidth=4,
    alpha=0.6,
    errorbar=None,
    ax=ax)
# plot the sampled reward
sns.scatterplot(data=prb_data_sim, 
    x="trial", 
    y="r_sampled_0", 
    label="r_sampled_A",
    color=viz.Blue, 
    alpha=0.6,
    edgecolor="none",
    s=10,
    ax=ax)
sns.scatterplot(data=prb_data_sim, 
    x="trial", 
    y="r_sampled_1", 
    label="r_sampled_B",
    color=viz.Red, 
    alpha=0.6,
    edgecolor="none",
    s=10,
    ax=ax)
# plot the predicted Q values
sns.lineplot(data=prb_data_sim, 
    x="trial", 
    y="qA", 
    label="qA",
    color=viz.Blue, 
    linewidth=2.5,
    errorbar=None,
    ax=ax)
sns.lineplot(data=prb_data_sim, 
    x="trial", 
    y="qB", 
    label="qB",
    color=viz.Red, 
    linewidth=2.5,
    errorbar=None,
    ax=ax)
ax.set_ylabel("Reward")
ax.set_xlabel("Trial")
ax.set_ylim([-0.05, 1.05])
ax.legend(loc="upper right", bbox_to_anchor=(1.6, 0.9))
fig.tight_layout(rect=[0, 0, 0.6, 0.9])  
fig.subplots_adjust(right=0.6)
ax.set_title("Noisy Q-learning")
plt.savefig(f"figures/fig2_noisy_q_learning.pdf", dpi=300)
