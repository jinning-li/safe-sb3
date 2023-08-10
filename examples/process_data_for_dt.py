import h5py

data_dir = 'data/sac-Safexp-CarButton1-v0_es3_lam0.1.h5py'

with h5py.File(data_dir, 'r') as f:
    obs = f['observation'][:]
    ac = f['action'][:]
    reward = f['reward'][:]
    terminal = f['terminal'][:]
    cost = f['cost'][:]

trajectories = []
start = 0
for i in range(obs.shape[0]):
    if terminal[i]:
        end = i + 1
        traj = {
            "observations": obs[start:end],
            "actions": ac[start:end],
            "rewards": reward[start:end],
            "dones": terminal[start:end],
            "costs": cost[start:end],
        }
        trajectories.append(traj)
        start = end
import pdb;pdb.set_trace()