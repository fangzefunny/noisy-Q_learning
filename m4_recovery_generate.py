import os
import pickle

# my visualize package 
from viz import viz

viz.get_style()
from utils.env_fn import *
from utils.model import *

pth = os.path.dirname(os.path.abspath(__file__))
pth = os.path.join(pth, "data")
if not os.path.exists(pth): os.makedirs(pth)

# simulate the model 
bnds = (
    (0., .9),
    (0., 2),
    (0., 4),
    (0., 1),
)
agent = noisy_q_learning
n_subs = 20
n_prbs = 5
seed = 420
all_data = {}
p_names = agent.p_names
sub_params = {p: [] for p in p_names}
sub_params["sub_id"] = []
for sub_id in range(n_subs):
    sub_data = {}
    for prb_id in range(n_prbs):

        # load environment 
        env = restless_bandit(restless_bandit_config(  
                seed = 2+sub_id*n_prbs+prb_id,
                mode = "for_fit",
            ))
        prb_data = env.instan()

        # get parameter 
        params = [
            bnds[0][0] + np.random.rand() * (bnds[0][1] - bnds[0][0]),
            bnds[1][0] + np.random.rand() * (bnds[1][1] - bnds[1][0]),
            bnds[2][0] + np.random.rand() * (bnds[2][1] - bnds[2][0]),
            bnds[3][0] + np.random.rand() * (bnds[3][1] - bnds[3][0]),
        ]

        # generate subject data 
        prb_data = simulate(
            agent, 
            params, 
            prb_data,
            seed+sub_id*n_prbs+prb_id,
        )
        sub_data[f"prb_{prb_id}"] = prb_data
    all_data[f"sub_{sub_id}"] = sub_data

    # save the subject parameters 
    for i, p_name in enumerate(p_names):
        sub_params[p_name].append(params[i])
    sub_params["sub_id"].append(f"sub_{sub_id}")
    
# save the data 
fname = f"{pth}/restless_bandit.pkl"
with open(fname, "wb") as handle: pickle.dump(all_data, handle)
fname = f"{pth}/sub_params.csv"
pd.DataFrame(sub_params).to_csv(fname)
