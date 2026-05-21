"""Vectorized Tower Defense environment for multi-container training."""

from typing import Optional, List

import gymnasium
from gymnasium.vector import AsyncVectorEnv


def make_td_vector_env(
    num_envs: int,
    save_path: Optional[str] = None,
    **env_kwargs,
) -> AsyncVectorEnv:
    """Create a vectorized Tower Defense environment.

    Each sub-environment connects to a separate Factorio container.
    Containers must already be running (via `fle cluster`).

    Args:
        num_envs: Number of parallel environments.
        save_path: Optional path to a prebuilt save file.
        **env_kwargs: Additional kwargs passed to TowerDefenseEnv.

    Returns:
        An AsyncVectorEnv wrapping multiple TowerDefenseEnv instances.
    """

    def _make_env(idx: int):
        def _init():
            from fle.env.gym_env.registry import make_td_env

            return make_td_env(run_idx=idx, save_path=save_path, **env_kwargs)

        return _init

    env_fns = [_make_env(i) for i in range(num_envs)]
    return AsyncVectorEnv(env_fns)
