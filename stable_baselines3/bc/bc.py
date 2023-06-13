from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.policies import ActorCriticPolicy as BCPolicy
import h5py

SelfBC = TypeVar("SelfBC", bound="BC")


class BC(OffPolicyAlgorithm):
    """Behavior Cloning Algorithm.
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": BCPolicy,
    }
    policy: BCPolicy

    def __init__(
        self,
        policy: Union[str, Type[BCPolicy]],
        env: Union[GymEnv, str],
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
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )
        if _init_setup_model:
            self._setup_model()

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.policy.optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        actor_losses = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.policy.reset_noise()

            # Select action according to policy
            # actions = self.policy(replay_data.observations, deterministic=True)
            actions, _, _ = self.policy(replay_data.observations)
            target = replay_data.actions

            assert actions.shape == target.shape
            actor_loss = F.mse_loss(actions, target)

            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.policy.optimizer.zero_grad()
            actor_loss.backward()
            self.policy.optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))

    def learn(
        self: SelfBC,
        total_timesteps: int,
        data_dir: str,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "BC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfBC:
        self.load_replay_buffer(data_dir)
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def load_replay_buffer(self, data_dir) -> None:
        with h5py.File(data_dir, 'r') as f:
            obs = f['observation'][:]
            ac = f['action'][:]
            reward = f['reward'][:]
            terminal = f['terminal'][:]
            cost = f['cost'][:]
        size = min(obs.shape[0], self.replay_buffer.buffer_size) - 1
        for i in range(size):
            self.replay_buffer.add(
                obs[i][None],
                obs[i+1][None],
                ac[i],
                reward[i],
                terminal[i],
                [{"cost": cost[i]}],
            )
        print("Loaded datasets to Replay Buffer!\n")

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params()  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []
