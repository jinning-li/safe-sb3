
import gym
from gym import spaces




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




# Define the mapping function from discrete to continuous actions
def map_discrete_to_continuous(value, num_values, lb, hb):
    return [(value[i]/num_values)*(hb-lb) + lb for i in range(len(value))]

# Define the mapping function from continuous to discrete values
def map_continuous_to_discrete(value,num_values, lb, hb):
    return [int((value[i] - lb) * (num_values) / (hb-lb)) for i in range(len(value))]

if __name__ == "__main__":
    # Define the number of discrete values for each dimension
    num_values = 100

    # Define the continuous value space
    continuous_value_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    
    # Define the discrete value space
    discrete_value_space = spaces.MultiDiscrete([num_values] * continuous_value_space.shape[0])
    # Print information about the observation space
    print("discrete space:", discrete_value_space)
    print("discrete space shape:", discrete_value_space.shape)
    # Test the discretization
    continuous_value = [0.54, -0.76]
    discrete_value = map_continuous_to_discrete(continuous_value, num_values, -1.0, 1.0)
    continuous_value_restored = map_discrete_to_continuous(discrete_value, num_values, -1.0, 1.0)

    print("Continuous value:", continuous_value)
    print("Discrete value:", discrete_value)
    print("Restored Continuous value:", continuous_value_restored)