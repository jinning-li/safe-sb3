# Example usage:
# The expert model dir for transformers should be the base wandb experiment
# logdir.
# python examples/run_js_iql.py -e Safexp-CarButton1-v0 -es 3 -d cuda:1 \
# -emd /home/lijinning/decision-transformer/gym/wandb/run-20230810_155710-1gi0bods \
# --use_transformer_expert -lam 0.1
import os
from datetime import datetime

import bullet_safety_gym
import gym
import safety_gym
from wrappers import AddCostToRewardEnv

from stable_baselines3 import JumpStartIQL
from stable_baselines3.js_sac import utils as js_utils


def main(args):
    device = args["device"]
    use_transformer_expert = args["use_transformer_expert"]
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
        "js-iql-" + env_name + "_es" + str(args["env_seed"])
        # + "_lam" + str(lamb) + '_' + date)
        + "_lam" + str(lamb))
    if args["suffix"]:
        experiment_name += f'_{args["suffix"]}'
    if use_transformer_expert:
        experiment_name += '_transformer'
    tensorboard_log = os.path.join(root_dir, experiment_name)

    if use_transformer_expert:
        loaded_stats = js_utils.load_demo_stats(
            path=args["expert_model_dir"]
        )
        obs_mean, obs_std, reward_scale, target_return = loaded_stats
        expert_policy = js_utils.load_transformer(
            model_dir=args['expert_model_dir'], device=device
        )
    else:
        obs_mean, obs_std = None, None
        expert_policy = js_utils.load_expert_policy(
            model_dir=args['expert_model_dir'], env=env, device=device
        )

    model = JumpStartIQL(
        "MlpPolicy",
        expert_policy,
        env,
        use_transformer_expert=use_transformer_expert,
        target_return=target_return,
        reward_scale=reward_scale,
        obs_mean=obs_mean,
        obs_std=obs_std,
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
    parser.add_argument(
        '--use_transformer_expert', action='store_true', default=False
    )

    parser.add_argument('--lambda', '-lam', type=float, default=1.)
    parser.add_argument('--steps', '-st', type=int, default=int(1e7))
    parser.add_argument('--random', '-r', action='store_true', default=False)
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()
    args = vars(args)

    main(args)