from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv

 from metadrive.engine.asset_loader import AssetLoader
import os
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
import numpy as np

# class safeProxEnv(ScenarioEnv):
#     # FOR TEST ONLY: replace this when the safeProxEnv is implemented
#     def __init__(self, config=None):
#         super(WaymoEnv, self).__init__(config) 

#     def get_cost(self, action):
#         #### dummy return .1234 everytime
#         cost = 0.1234
#         return cost

def check_observation_action_space(env):
    # Access the observation space
    observation_space = env.observation_space

    # Print information about the observation space
    print("Observation space:", observation_space)
    print("Observation space shape:", observation_space.shape)
    print("Observation space high:", observation_space.high)
    print("Observation space low:", observation_space.low)

    # Access the action space
    action_space = env.action_space

    # Print information about the action space
    print("Action space:", action_space)
    print("Action space shape:", action_space.shape)
    print("Action space high:", action_space.high)
    print("Action space low:", action_space.low)



class SeparateCostRewardWaymoEnv(WaymoEnv):
    
    def __init__(self, config=None, lamb = 1.):
        super(SeparateCostRewardWaymoEnv, self).__init__(config)
        self.lamb = lamb
    

    def step(self, action):
        """ step return new_reward that has the cost seperated, with updated info
        """
        state, reward, done, info = super().step(action)
        cost = info['cost']
        new_reward = reward - self.lamb*info['cost']
        info["re"] = new_reward
        return state, new_reward, cost, done, info
    

class SeparateCostRewardScenarioEnv(ScenarioEnv):
    
    def __init__(self, config=None, lamb = 1.):
        super(SeparateCostRewardScenarioEnv, self).__init__(config)
        self.lamb = lamb
    

    def step(self, action):
        """ step return new_reward that has the cost seperated, with updated info
        """
        state, reward, done, info = super().step(action)
        cost = info['cost']
        new_reward = reward - self.lamb*info['cost']
        info["re"] = new_reward
        return state, new_reward, cost, done, info
    

def main():
    

    # FROM metadrive/metadrive/envs/real_data_envs/waymo_env.py
    asset_path = AssetLoader.asset_path
    env_name = "waymo" 


    env = SeparateCostRewardWaymoEnv(
            {
                "use_render": False,
                # "agent_policy": ExpertPolicy, agent_policy is for ego veh
                "manual_control": False,
                "no_traffic": False,
                "start_scenario_index": 0,
                "show_coordinates": True,
                "num_scenarios": 3,
                "h5py_directory": AssetLoader.file_path(asset_path, "waymo", return_raw_style=False),
                "show_policy_mark": False,
                "no_static_vehicles": True,
                "reactive_traffic": True,
                # "vehicle_config": dict(
                #     show_lidar=True,
                #     # no_wheel_friction=True,
                #     lidar=dict(num_lasers=0))
                # "vehicle_config": dict(
                # # no_wheel_friction=True,
                # lidar=dict(num_lasers=120, distance=50, num_others=4),
                # lane_line_detector=dict(num_lasers=12, distance=50),
                # side_detector=dict(num_lasers=160, distance=50)
            # ),
            }
        ) 

    
    env_cost_reward = env


    check_observation_action_space(env)
    root_dir = "tensorboard_log"
    exp_name = ("sac_" + env_name + 
                "_default" ) # other config
    
    loaded_agent = SAC.load(exp_name,env = env_cost_reward)
    # mean_reward, _ = evaluate_policy(agent, env, n_eval_episodes=2)
    # print(f"Mean reward: {mean_reward:.2f}")

    obs = env_cost_reward.reset()
    rewards = []
    costs = []
    for _ in range(1000):
        action, _states = loaded_agent.predict(obs, deterministic=True)
        state, reward, cost, done, info = env_cost_reward.step(action)
        rewards.append(reward)
        costs.append(cost)
        print(reward, cost)
        if done:
            obs = env_cost_reward.reset()
            acc_rew = np.cumsum(rewards)
            acc_cost = np.cumsum(costs)
            plt.figure()
            plt.plot(np.arange(len(acc_rew)), acc_rew, label = "acc rewards")
            plt.plot(np.arange(len(acc_cost)), acc_cost, label = "acc costs")
            plt.legend() 
            plt.xlabel("timestamp")
            plt.ylabel("acc reward or cost")
            plt.figure()
            plt.plot(np.arange(len(rewards)), rewards, label = "acc rewards")
            plt.plot(np.arange(len(costs)), costs, label = "acc costs")
            plt.legend() 
            plt.xlabel("timestamp")
            plt.ylabel("reward or cost")
            plt.show()    

    # load.save(exp_name)
    # loaded_agent = SAC.load(exp_name)

    
    


if __name__ == "__main__":
    
    main()