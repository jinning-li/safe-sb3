import os
from datetime import datetime

import bullet_safety_gym
import gym
import safety_gym
from wrappers import AddCostToRewardEnv

from stable_baselines3 import JumpStartSAC
from stable_baselines3.js_sac import utils as js_utils


def main(args):
    device = args["device"]
    env_name = args["env"]
    env = gym.make(env_name)
    env.seed(args["env_seed"])
    if args["random"]:
        env.set_num_different_layouts(100)
    lamb = args["lambda"]
    env = AddCostToRewardEnv(env, lamb=lamb)

    root_dir = "tensorboard_logs"
    # date = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_name = (
        "js-sac-" + env_name + "_es" + str(args["env_seed"])
        # + "_lam" + str(lamb) + '_' + date)
        + "_lam" + str(lamb))
    if args["suffix"]:
        experiment_name += f'_{args["suffix"]}'
    tensorboard_log = os.path.join(root_dir, experiment_name)

    expert_policy = js_utils.load_expert_policy(
        model_dir=args['expert_model_dir'], env=env, device=device
    )

    model = JumpStartSAC(
        "MlpPolicy",
        expert_policy,
        env,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device=device,
    )
    model.learn(total_timesteps=args["steps"])

    del model

    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, default='Safexp-CarButton1-v0')
    parser.add_argument('--env_seed', '-es', type=int, default=3)
    parser.add_argument('--device', '-d', type=str, default="cpu")

    # E.g., expert_model_dir: 'sac-Safexp-CarButton1-v0_es3_lam0.1/SAC_6'
    parser.add_argument('--expert_model_dir', '-emd', type=str, required=True)

    parser.add_argument('--lambda', '-lam', type=float, default=1.)
    parser.add_argument('--steps', '-st', type=int, default=int(1e7))
    parser.add_argument('--random', '-r', action='store_true', default=False)
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()
    args = vars(args)

    main(args)