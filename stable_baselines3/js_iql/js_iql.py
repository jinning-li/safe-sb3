from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.iql.iql import IQL
from stable_baselines3.iql.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy, ValueNet
from stable_baselines3.js_sac import utils as js_utils

SelfJumpStartIQL = TypeVar("SelfJumpStartIQL", bound="JumpStartIQL")


class JumpStartIQL(IQL):
    """
    Jump Start Implicit Q Learning (js-IQL).
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
    value: ValueNet

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        expert_policy: Any,
        env: Union[GymEnv, str],
        # data_collection_env: GymEnv,
        use_transformer_expert: bool,
        target_return: Optional[float] = None,
        reward_scale: Optional[float] = None,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
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
        # self.data_collection_env = data_collection_env
        self.guidance_timesteps = guidance_timesteps
        js_utils.check_expert_policy(expert_policy, self)
        self.expert_policy = expert_policy
        self.use_transformer_expert = use_transformer_expert
        if self.use_transformer_expert:
            assert target_return is not None
            assert reward_scale is not None
            assert obs_mean is not None
            assert obs_std is not None
        self.target_return_init = target_return
        self.reward_scale = reward_scale
        self.obs_mean = th.from_numpy(obs_mean).to(device=device)
        self.obs_std = th.from_numpy(obs_std).to(device=device)

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
        self.hist_ac = np.concatenate(
            [self.hist_ac, np.zeros((1, self.ac_dim))], axis=0
        )
        self.hist_re = np.concatenate([self.hist_re, np.zeros(1)])
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            guide_prob = self.get_guide_probability()
            use_guide = np.random.choice([False, True], p=[1-guide_prob, guide_prob])
            if use_guide:
                if self.use_transformer_expert:
                    hist_obs = th.tensor(
                        self.hist_obs, dtype=th.float32, device=self.device
                    )
                    hist_ac = th.tensor(
                        self.hist_ac, dtype=th.float32, device=self.device
                    )
                    hist_re = th.tensor(
                        self.hist_re, dtype=th.float32, device=self.device
                    )
                    target_return = th.tensor(
                        self.target_return, dtype=th.float32, device=self.device
                    )
                    timesteps = th.tensor(
                        self.timesteps, dtype=th.long, device=self.device
                    )
                    unscaled_action = self.expert_policy.get_action(
                        (hist_obs - self.obs_mean) / self.obs_std,
                        hist_ac,
                        hist_re,
                        target_return,
                        timesteps,
                    )
                    unscaled_action = unscaled_action.detach().cpu().numpy()
                else:
                    unscaled_action, _ = self.expert_policy.predict(
                        self._last_obs, deterministic=False
                    )
            else:
                # Note: when using continuous actions,
                # we assume that the policy uses tanh to scale the action
                # We use non-deterministic action in the case of SAC, for TD3, it does not matter
                unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
        self.hist_ac[-1] = unscaled_action
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
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.hist_obs = np.concatenate([self.hist_obs, new_obs], axis=0)
            assert len(self.hist_obs.shape) == 2
            self.hist_re[-1] = rewards
            pred_return = self.target_return[0,-1] - (rewards/self.reward_scale)
            self.target_return = np.concatenate(
                [self.target_return, pred_return.reshape(1, 1)], axis=1)
            t = self.timesteps[0, -1] + 1
            self.timesteps = np.concatenate(
                [self.timesteps, np.ones((1, 1)) * t], axis=1)

            assert dones.shape == (1, )
            if dones:
                self.hist_obs = self._last_obs.reshape(1, self.obs_dim)
                self.hist_ac= np.zeros((0, self.ac_dim))
                self.hist_re = np.zeros(0)
                if self.use_transformer_expert:
                    self.target_return = np.array([[self.target_return_init]])
                self.timesteps = np.zeros((1, 1))

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            from stable_baselines3.bc.policies import BCPolicy
            if not isinstance(self.policy, BCPolicy):
                self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def get_guide_probability(self):
        if self.num_timesteps > self.guidance_timesteps:
            return 0.
        prob_start = 0.9
        prob = prob_start * np.exp(-5. * self.num_timesteps / self.guidance_timesteps)
        return prob
