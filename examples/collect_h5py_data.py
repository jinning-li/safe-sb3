import gym
import h5py
import numpy as np
import safety_gym
import bullet_safety_gym
import tqdm

from stable_baselines3 import SAC

from wrappers import AddCostToRewardEnv


def main(args):
    env_name = args["env"]
    env = gym.make(env_name)
    env.seed(args["env_seed"])

    model_dir = args["policy_load_dir"]

    model = SAC("MlpPolicy", env)
    model.set_parameters(model_dir)

    total_timesteps = int(args["steps"])

    obs_rec = np.ndarray((0, ) + env.observation_space.shape)
    ac_rec = np.ndarray((0, ) + env.action_space.shape)
    re_rec = np.ndarray((0, ))
    terminal_rec = np.ndarray((0, ), dtype=bool)
    cost_rec = np.ndarray((0, ))
    cost_hazards_rec = np.ndarray((0, ))

    f = h5py.File(args["output_dir"], 'w')
    obs = env.reset()
    for _ in tqdm.tqdm(range(total_timesteps)):
        ac, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, info = env.step(ac)
        obs_rec = np.concatenate((obs_rec, obs[None]))
        ac_rec = np.concatenate((ac_rec, ac[None]))
        re_rec = np.concatenate((re_rec, np.array([reward])))
        terminal_rec = np.concatenate((terminal_rec, np.array([done])))
        cost_rec = np.concatenate((cost_rec, np.array([info['cost']])))
        # cost_hazards_rec = np.concatenate((cost_hazards_rec, np.array([info['cost_hazards']])))

        if done:
            obs = env.reset()
            print(f"Mean Episode Return: {re_rec.sum()/terminal_rec.sum()}")
        else:
            obs = next_obs

    f.create_dataset("observation", data=obs_rec)
    f.create_dataset("action", data=ac_rec)
    f.create_dataset("reward", data=re_rec)
    f.create_dataset("terminal", data=terminal_rec)
    f.create_dataset("cost", data=cost_rec)
    # f.create_dataset("cost_hazards", data=cost_hazards_rec)

    f.close()


if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, default='Safexp-CarButton1-v0')
    parser.add_argument('--env_seed', '-es', type=int, default=3)
    parser.add_argument('--steps', type=int, default=int(1e5))
    parser.add_argument('--policy_load_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()
    args = vars(args)

    main(args)
