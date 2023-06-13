from stable_baselines3.common.policies import ActorCriticPolicy
# from stable_baselines3.bc.bc_old import BC
from stable_baselines3.bc.bc import BC
MlpPolicy = ActorCriticPolicy


__all__ = ["MlpPolicy", "BC"]
