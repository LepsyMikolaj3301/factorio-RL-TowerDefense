"""Action masking wrapper for Tower Defense environments."""

import numpy as np
import gymnasium
from gymnasium import spaces

from fle.env.gym_env.td_spaces import (
    ACTION_NOOP,
    ACTION_PLACE_WALL,
    ACTION_PLACE_TURRET,
    ACTION_REFILL_TURRET,
    ACTION_SHOOT,
    NUM_ACTION_TYPES,
    TRACKED_ITEMS,
)


class ActionMaskWrapper(gymnasium.Wrapper):
    """Adds an action_mask to observations based on inventory state.

    The mask is a binary vector of shape (NUM_ACTION_TYPES,) where 1 means
    the action is valid and 0 means it should not be taken.

    Compatible with SB3's MaskablePPO via sb3-contrib.
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        # Extend observation space with action_mask
        self.observation_space = spaces.Dict(
            {
                **env.observation_space.spaces,
                "action_mask": spaces.MultiBinary(NUM_ACTION_TYPES),
            }
        )

    def _compute_mask(self, obs: dict) -> np.ndarray:
        """Compute action mask from current observation."""
        mask = np.ones(NUM_ACTION_TYPES, dtype=np.int8)

        inv = obs.get("inventory", np.zeros(len(TRACKED_ITEMS)))

        # Can't place wall if no walls in inventory
        wall_idx = TRACKED_ITEMS.index("stone-wall")
        if inv[wall_idx] <= 0:
            mask[ACTION_PLACE_WALL] = 0

        # Can't place turret if no turrets in inventory
        turret_idx = TRACKED_ITEMS.index("gun-turret")
        if inv[turret_idx] <= 0:
            mask[ACTION_PLACE_TURRET] = 0

        # Can't refill if no ammo
        ammo_idx_1 = TRACKED_ITEMS.index("firearm-magazine")
        ammo_idx_2 = TRACKED_ITEMS.index("piercing-rounds-magazine")
        if inv[ammo_idx_1] <= 0 and inv[ammo_idx_2] <= 0:
            mask[ACTION_REFILL_TURRET] = 0

        # Noop is always valid
        mask[ACTION_NOOP] = 1

        return mask

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["action_mask"] = self._compute_mask(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs["action_mask"] = self._compute_mask(obs)
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return current action mask (for SB3 MaskablePPO compatibility)."""
        obs = self.env.unwrapped._last_obs
        if obs is None:
            return np.ones(NUM_ACTION_TYPES, dtype=np.int8)
        return self._compute_mask(obs)
