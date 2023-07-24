from stable_baselines3 import IQL
import gym
import safety_gym
import bullet_safety_gym
from wrappers import AddCostToRewardEnv
import os
from datetime import datetime


def main(args):
    env_name = args["env"]
    env = gym.make(env_name)
    env.seed(args["env_seed"])
    lamb = args["lambda"]
    env = AddCostToRewardEnv(env, lamb=lamb)

    root_dir = "tensorboard_logs"
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_name = (
        "iql-" + env_name + "_es" + str(args["env_seed"]) 
        # + "_lam" + str(lamb) + '_' + date)
        + "_lam" + str(lamb))
    tensorboard_log = os.path.join(root_dir, experiment_name)

    model = IQL("MlpPolicy", env, tensorboard_log=tensorboard_log, verbose=1, device="cpu")
    model.learn(total_timesteps=args["steps"])
    model.save("iql-" + env_name + "-es" + str(args["env_seed"]))

    del model

    env.close()


if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, default='Safexp-CarButton1-v0')
    parser.add_argument('--env_seed', '-es', type=int, default=3)
    parser.add_argument('--lambda', '-lam', type=float, default=1.)
    parser.add_argument('--steps', '-st', type=int, default=int(1e7))
    args = parser.parse_args()
    args = vars(args)

    main(args)