from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.js_sac import utils as js_utils
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.sac.sac import SAC


SelfJumpStartSAC = TypeVar("SelfJumpStartSAC", bound="JumpStartSAC")


class JumpStartSAC(SAC):
    """
    Jump Start Soft Actor-Critic (SAC).

    It has the same backbone as SAC. The only difference is that there is a guide (expert) policy,
    which leads the exploration policy (the one being trained) to high reward areas.
    The action sampling process is overwritten to adopt the guide policy.
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        expert_policy: Any,
        env: Union[GymEnv, str],
        data_collection_env: GymEnv,
        guidance_timesteps: int = 500_000,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            _init_setup_model=_init_setup_model,
        )
        self.data_collection_env = data_collection_env
        self.guidance_timesteps = guidance_timesteps
        js_utils.check_expert_policy(expert_policy, self)
        self.expert_policy = expert_policy

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            guide_prob = self.get_guide_probability()
            use_guide = np.random.choice([False, True], p=[1-guide_prob, guide_prob])
            if use_guide:
                unscaled_action, _ = self.expert_policy.predict(self._last_obs, deterministic=False)
            else:
                # Note: when using continuous actions,
                # we assume that the policy uses tanh to scale the action
                # We use non-deterministic action in the case of SAC, for TD3, it does not matter
                unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def get_guide_probability(self):
        if self.num_timesteps > self.guidance_timesteps:
            return 0.
        prob_start = 0.9
        prob = prob_start * np.exp(-5. * self.num_timesteps / self.guidance_timesteps)
        return prob
