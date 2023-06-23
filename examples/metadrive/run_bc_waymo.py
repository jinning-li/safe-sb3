from stable_baselines3 import BC
import gym

import h5py
import os
from datetime import datetime

from trafficgen.utils.typedef import AgentType, RoadLineType, RoadEdgeType
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

from metadrive.engine.asset_loader import AssetLoader
from stable_baselines3 import BC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


WAYMO_SAMPLING_FREQ = 10




class AddCostToRewardEnv(WaymoEnv):
    def __init__(self, wrapped_env, lamb=1.):
        """Initialize the class.
        
        Args: 
            wrapped_env: the env to be wrapped
            lamb: new_reward = reward + lamb * cost_hazards
        """
        super().__init__(wrapped_env)
        self._lamb = lamb

    def set_lambda(self, lamb):
        self._lamb = lamb

    def step(self, action):
        state, reward, done, info = super().step(action)
        new_reward = reward - self._lamb * info['cost']
        info["re"] = reward
        return state, new_reward, done, info

def main(args):

    
    file_list = os.listdir(args['pkl_dir'])
    num_scenarios = len(file_list)
    print("num of scenarios: ", num_scenarios)
    env = AddCostToRewardEnv(
    {
        "manual_control": False,
        "no_traffic": False,
        "agent_policy":ReplayEgoCarPolicy,
        "waymo_data_directory":args['pkl_dir'],
        "case_num": num_scenarios,
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        
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
    }, lamb=5.
    )


    env.seed(0)

    exp_name = "bc-waymo-es" + str(args["env_seed"])
    root_dir = "tensorboard_log"
    tensorboard_log = os.path.join(root_dir, exp_name)

    model = BC("MlpPolicy", env, tensorboard_log=tensorboard_log, verbose=1)
    model.learn(total_timesteps=args['steps'], data_dir=args['h5py_path'])

    model.save(os.path.join(args['output_dir'], exp_name))
    # loaded_agent = BC.load(exp_name)

    
    del model
    env.close()

    # done = False
    # while not done:
    #     env.render()
    #     action = env.action_space.sample()  # Replace with your agent's action selection logic
    #     obs, reward, done, info = env.step(action)

def test(args):
    file_list = os.listdir(args['pkl_dir'])
    num_scenarios = len(file_list)
    print("num of scenarios: ", num_scenarios)
    env = AddCostToRewardEnv(
    {
        "manual_control": False,
        "no_traffic": False,
        # "agent_policy":ReplayEgoCarPolicy,
        "waymo_data_directory":args['pkl_dir'],
        "case_num": num_scenarios,
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        "use_render": True,
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


    env.seed(0)
    
    exp_name = "bc-waymo-es" + str(args["env_seed"])
    
    model = BC.load('examples/metadrive/saved_bc_policy/bc-waymo-es3.zip')
    for seed in range(0, num_scenarios):
            o = env.reset(force_seed=seed)
            
            for i in range(199):
                action, _ = model.predict(o, deterministic = True)
                o, r, d, info = env.step(action)

                # env.render(mode="rgb_array")
                print('action:', action, 'reward: ', r, 'cost: ',info['cost'], 'done:', d)
                if d:
                    print('seed '+str(seed)+' is over!')
                    break

                
            
    del model
    env.close()





if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5py_path', '-h5', type=str, default='examples/metadrive/h5py/one_pack.h5py')
    parser.add_argument('--pkl_dir', '-pkl', type=str, default='examples/metadrive/pkl_20')
    parser.add_argument('--output_dir', '-out', type=str, default='examples/metadrive/saved_bc_policy')
    parser.add_argument('--env_seed', '-es', type=int, default=0)
    parser.add_argument('--lambda', '-lam', type=float, default=1.)
    parser.add_argument('--steps', '-st', type=int, default=int(1e5))
    args = parser.parse_args()
    args = vars(args)

    main(args)
    # test(args)