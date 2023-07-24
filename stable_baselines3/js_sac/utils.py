import os
import re

import numpy as np

from stable_baselines3.bc import BC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.sac.sac import SAC


def infer_algo_cls_from_dir_name(dir):
    tmp = dir.lower()
    tmp = re.split('_|-|/', tmp)
    if "bc" in tmp:
        return BC
    elif "sac" in tmp:
        return SAC
    else:
        raise NotImplementedError(
            "The expert policy class is not found! Please follow the"
            " convention to name the model directory!"
        )

def load_expert_policy(model_dir, env, device="cpu"):
    algo_cls = infer_algo_cls_from_dir_name(model_dir)
    model = algo_cls("MlpPolicy", env, device=device)
    model_dir = os.path.join("tensorboard_logs", model_dir, "model.pt")
    model.set_parameters(model_dir)
    expert_policy = model.policy
    return expert_policy

def check_expert_policy(expert_policy, algorithm):
    if not hasattr(expert_policy, "predict"):
        raise NotImplementedError(
            "Expert policy must implement a method 'perdict'!"
            " It predicts next actions from current observations."
            " Please follow the definition of `BaseAlgorithm.predict`."
        )
    if not isinstance(algorithm, BaseAlgorithm):
        raise ValueError(
            "`algorithm` must be inherited from `BaseAlgorithm!`"
        )
    expected_shape = (algorithm.n_envs, ) + algorithm.action_space.shape
    obs_shape = (algorithm.n_envs, ) + algorithm.observation_space.shape
    dummy_obs = np.random.rand(*obs_shape)
    dummy_action, _ = expert_policy.predict(dummy_obs)
    if dummy_action.shape != expected_shape:
        raise ValueError(
            "The output shape of `expert_policy.predict` should be"
            f" {expected_shape}! Got {dummy_action.shape} instead."
        )