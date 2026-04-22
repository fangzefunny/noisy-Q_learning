import os
import pickle
import time 
import argparse
from pybads import BADS

# my visualize package 
from viz import viz

viz.get_style()
from utils.env_fn import *
from utils.model import *
from utils.fit import loss_fn_pf

pth = os.path.dirname(os.path.abspath(__file__))
os.chdir(pth)

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="Fit the model to the data")
parser.add_argument("--data_set",   "-d", help="choose data name", default="restless_bandit")
parser.add_argument("--agent_name", "-n", help="choose agent", default="noisy_q_learning")
parser.add_argument("--job_id",     "-j", help="job id", type=int, default=0)
parser.add_argument("--seed",       "-s", help="random seed", type=int, default=420)
args = parser.parse_args()
agent  = eval(args.agent_name)

dirs = [f"{pth}/fits", f"{pth}/fits/{args.data_set}", f"{pth}/fits/{args.data_set}/{args.agent_name}"]
for d in dirs: 
    if not os.path.exists(d): os.mkdir(d)


 ## get sub_id and fit sample 
fname = f'{pth}/data/{args.data_set}.pkl'
with open(fname, "rb") as handle: data_for_fit = pickle.load(handle)
# get sub_id and fit sample 
n_sub = len(data_for_fit.keys())
i_sub, i_fit = args.job_id%n_sub, args.job_id//n_sub
sub_id = list(data_for_fit.keys())[i_sub]

# get the subject data 
sub_data = data_for_fit[sub_id]
key = f'{sub_id}-{i_fit}'
print(f"Fitting parameter for subject {sub_id}, the {i_fit}th fit sample")

# fit back the parameters 
print("Start model fitting...", flush=True)
target = lambda params: loss_fn_pf(
    sub_data, 
    agent, 
    params, 
    n_particles=200,
    seed=args.seed+args.job_id)
lb =  np.array([bnd[0] for bnd in agent.p_bnds])
ub =  np.array([bnd[1] for bnd in agent.p_bnds])
plb = np.array([pbnd[0] for pbnd in agent.p_pbnds])
pub = np.array([pbnd[1] for pbnd in agent.p_pbnds])
rng = np.random.default_rng(args.seed+args.job_id)
param0 = plb + rng.random(len(agent.p_names)) * (pub - plb)

# set up bads optimizer 
bads_opt = {
    'uncertainty_handling': True,
    'noise_final_samples': 0,
}
start_caption = f"""
Starting bads option with paramters:
    {[f"{p:.4f}" for p in param0]}
"""
print(start_caption, flush=True)
bads = BADS(target, 
            param0, 
            lb,
            ub,
            plb,
            pub,
            options=bads_opt)
start_time = time.time()
result = bads.optimize()
end_time = time.time()
n_data = sum(len(prb_data.keys()) for prb_data in sub_data.values())
final_result = {
    "log_post": -result["fval"],
    "log_like": -result["fval"],
    "param": result["x"],
    "param_name": agent.p_names,
    "n_param": len(agent.p_names),
    "aic": result["fval"] + 2*len(agent.p_names),
    "bic": result["fval"] + np.log(n_data)*len(agent.p_names),
}
opt_loss_val = -final_result["log_post"]
opt_params = final_result["param"]
print(f"lowest loss: {opt_loss_val:.4f}, using {(end_time - start_time):.2f} seconds", flush=True)
print(f"opt_params: {opt_params}", flush=True)

# save the result
fname = f'{pth}/fits/{args.data_set}/{args.agent_name}/fit_sub_info-{key}-mle-pf.pkl'
with open(fname, 'wb')as handle: pickle.dump(final_result, handle)