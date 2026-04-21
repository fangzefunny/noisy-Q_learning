import os
import time
from pybads import BADS

# my visualize package 
from viz import viz

from utils.fit import loss_ln_pf 
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
params = [.2, .1, 8, .1]
seed = 1234
n_prbs = 10
sub_data = {}
for prb_id in range(n_prbs):
    prb_data_sim = simulate(
        agent, 
        params, 
        prb_data,
        seed,
    )
    sub_data[f"prb_{prb_id}"] = prb_data_sim

# fit back the parameters 
seed = 42
agent = noisy_q_learning
target = lambda params: loss_ln_pf(
    sub_data, 
    agent, 
    params, 
    n_particles=200,
    seed=seed)
lb =  np.array([bnd[0] for bnd in agent.p_bnds])
ub =  np.array([bnd[1] for bnd in agent.p_bnds])
plb = np.array([pbnd[0] for pbnd in agent.p_pbnds])
pub = np.array([pbnd[1] for pbnd in agent.p_pbnds])
param0 = plb + np.random.rand(len(agent.p_names)) * (pub - plb)

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

