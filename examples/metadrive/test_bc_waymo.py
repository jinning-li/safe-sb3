import metadrive
import sys
sys.path.append("/home/xinyi/Software/miniconda3/lib/python3.8/site-packages/stable_baselines3/")
from stable_baselines3 import SAC
from stable_baselines3 import BC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from metadrive.engine.asset_loader import AssetLoader
from metadrive.manager.waymo_traffic_manager import WaymoTrafficManager
from metadrive.policy.idm_policy import WaymoIDMPolicy
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from stable_baselines3.common.buffers import ReplayBuffer
import tensorboard
import matplotlib.pyplot as plt
import numpy as np
import gym
import os

from metadrive import MetaDriveEnv
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy
from metadrive.policy.expert_policy import ExpertPolicy
from sp_utils import check_observation_action_space, map_discrete_to_continuous, map_continuous_to_discrete


if __name__ == "__main__":
    # FROM metadrive/metadrive/envs/real_data_envs/waymo_env.py
    asset_path = AssetLoader.asset_path
    env_name = "waymo"
    # created a waymoEnv
    env = WaymoEnv(
            {
                "use_render": False,
                # "agent_policy": ExpertPolicy, agent_policy is for ego veh
                "manual_control": False,
                "no_traffic": False,
                "start_scenario_index": 0,
                "show_coordinates": True,
                "num_scenarios": 3,
                "data_directory": AssetLoader.file_path(asset_path, "waymo", return_raw_style=False),
                "show_policy_mark": False,
                "no_static_vehicles": True,
                "reactive_traffic": False,
                # "vehicle_config": dict(
                #     show_lidar=True,
                #     # no_wheel_friction=True,
                #     lidar=dict(num_lasers=0))
                "vehicle_config": dict(
                # no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
            }
        ) 

    check_observation_action_space(env)



    # env = Monitor(DummyVecEnv([lambda: env]))
    # env = DummyVecEnv([lambda: env])
    
    # check_observation_action_space(env)
    root_dir = "tensorboard_log"
    exp_name = ("sac_" + env_name + 
                "_replay" ) # other config
    tensorboard_log = os.path.join(root_dir,exp_name)
    # create a buffer
    buffer_size = 100000
    
    # Initialize the SAC agent, passing the MetaDrive environment and the replay buffer to the agent
    agent = SAC("MlpPolicy", env, 
                # buffer_size=buffer_size,
                tensorboard_log=tensorboard_log,
                # optimize_memory_usage=True,
                verbose=1)
    total_timesteps = buffer_size * 4
    agent.learn(total_timesteps=total_timesteps)
    agent.save(exp_name)
    loaded_agent = SAC.load(exp_name, env)

    mean_reward, _ = evaluate_policy(agent, env, n_eval_episodes=2)
    print(f"Mean reward: {mean_reward:.2f}")

    agent.save(exp_name)
    loaded_agent = SAC.load(exp_name)



