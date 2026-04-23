import os
import time 
import argparse
import pickle
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns 

# my visualize package 
from viz import viz
from utils.model import *
import stats

# get the pth 
pth = os.path.dirname(os.path.abspath(__file__))
os.chdir(pth)

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="Fit the model to the data")
parser.add_argument("--data_set",   "-d", help="choose data name", default="restless_bandit-noisy_q_learning_weber")
parser.add_argument("--agent_name", "-n", help="choose agent", default="noisy_q_learning_weber")
parser.add_argument("--job_id",     "-j", help="job id", type=int, default=0)
parser.add_argument("--seed",       "-s", help="random seed", type=int, default=420)
args = parser.parse_args()
agent  = eval(args.agent_name)

# load the original parameter 
df_original = pd.read_csv(f"data/sub_params-{args.data_set}.csv", index_col=0)

# get the data for fit
fname = f"data/{args.data_set}.pkl"
with open(fname, "rb")as handle: data_for_fit = pickle.load(handle) 
sub_lst = list(data_for_fit.keys())

# load the recovered parameter 
n_rep = 10
cols = ["sub_id", "random_init_id", "alpha", "eta", "beta", "sigma", "nll"]
df_recovered = {col: [] for col in cols}
for sub_id in sub_lst:
    for i in range(n_rep):

        # get the subject fiting information
        fname = f"fits/{args.data_set}/{args.agent_name}/fit_sub_info-{sub_id}-{i}-mle-pf.pkl"
        with open(fname, "rb")as handle: fit_sub_info = pickle.load(handle)

        # get the subject id 
        df_recovered["sub_id"].append(sub_id)
        df_recovered["random_init_id"].append(i)
        df_recovered["nll"].append(-fit_sub_info["log_like"])

        # get the recovered parameter
        for j, p_name in enumerate(["alpha", "eta", "beta", "sigma"]):
            df_recovered[p_name].append(float(fit_sub_info["param"][j]))

df_recovered = pd.DataFrame(df_recovered)

# keep the best fit (lowest nll) for each subject across random initializations
best_idx = df_recovered.groupby("sub_id")["nll"].idxmin()
df_recovered_best = (
    df_recovered.loc[best_idx]
    .sort_values("sub_id")
    .reset_index(drop=True)
)

# combine the original and recovered dfs
df_merged = df_original.merge(
    df_recovered_best,  
    on="sub_id",
    suffixes=("_true", "_recovered"),
)

# visualize 
fig, axs = plt.subplots(1, 4, figsize=(9, 2.5))
axs = axs.flatten()
for i, p_name in enumerate(["alpha", "eta", "beta", "sigma"]):
    sns.scatterplot(data=df_merged, x=f"{p_name}_true", y=f"{p_name}_recovered", ax=axs[i])
    stats.corr(df_merged[f"{p_name}_true"], df_merged[f"{p_name}_recovered"], method="spearman", title=f"{p_name}", p_bar=1)
    low = min(df_merged[f"{p_name}_true"].min(), df_merged[f"{p_name}_recovered"].min())
    high = max(df_merged[f"{p_name}_true"].max(), df_merged[f"{p_name}_recovered"].max())
    axs[i].set_xlim(low*0.9, high*1.1)
    axs[i].set_ylim(low*0.9, high*1.1)
    ticks = np.round(np.linspace(low, high, 3), 2)
    axs[i].set_xticks(ticks)
    axs[i].set_yticks(ticks)
    axs[i].plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
    axs[i].set_xlabel(f"{p_name}_true")
    axs[i].set_ylabel(f"{p_name}_recovered")
    axs[i].set_title(f"{p_name}")
    axs[i].set_box_aspect(1)
fig.tight_layout()

plt.savefig(f"figures/fig3_recovery_visualize-{args.agent_name}.pdf", dpi=300)





