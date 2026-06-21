"""Action masking wrapper for Tower Defense environments."""

import numpy as np
import gymnasium
from gymnasium import spaces

from fle.env.gym_env.td_spaces import (
    ACTION_MOVE_ANCHOR,
    ACTION_NOOP,
    ACTION_PLACE_TURRET,
    ACTION_REFILL_TURRET,
    MAX_ANCHORS,
    MAX_SLOTS,
    NUM_ACTION_TYPES,
    TRACKED_ITEMS,
)


class ActionMaskWrapper(gymnasium.Wrapper):
    """Adds action masks to observations based on inventory and slot state.

    Two complementary masks are exposed:

    - ``action_type_mask`` (shape ``(NUM_ACTION_TYPES,)``): whether each action
      *type* is worth taking at all. PLACE_TURRET needs an empty slot and a
      turret in inventory; REFILL_TURRET needs an occupied slot and ammo; NOOP
      is always valid.
    - ``slot_mask`` (shape ``(MAX_SLOTS,)``): real slots. Because standard
      MaskablePPO masks each discrete factor independently and cannot condition
      the slot mask on the chosen action type, the per-action validity
      (empty vs occupied) is enforced at execution time via the invalid-action
      penalty. The observation also carries ``place_slot_mask`` /
      ``refill_slot_mask`` for callers that do custom action-conditioned masking.

    Compatible with SB3's MaskablePPO via sb3-contrib.
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self._obs: dict = {}
        # Extend observation space with the masks the policy can consume.
        self.observation_space = spaces.Dict(
            {
                **env.observation_space.spaces,
                "action_type_mask": spaces.MultiBinary(NUM_ACTION_TYPES),
                "slot_mask": spaces.MultiBinary(MAX_SLOTS),
                "anchor_mask": spaces.MultiBinary(MAX_ANCHORS),
            }
        )

    def _compute_action_type_mask(self, obs: dict) -> np.ndarray:
        """Compute the per-action-type mask from inventory and slot state."""
        mask = np.zeros(NUM_ACTION_TYPES, dtype=np.int8)
        mask[ACTION_NOOP] = 1  # Noop always valid

        inv = obs.get("inventory", np.zeros(len(TRACKED_ITEMS)))
        place_slots = obs.get("place_slot_mask", np.zeros(MAX_SLOTS))
        refill_slots = obs.get("refill_slot_mask", np.zeros(MAX_SLOTS))

        # Can place a turret only if we have one and there is an empty slot.
        turret_idx = TRACKED_ITEMS.index("gun-turret")
        if inv[turret_idx] > 0 and np.any(place_slots):
            mask[ACTION_PLACE_TURRET] = 1

        # Can refill only if we have ammo and there is an occupied slot.
        ammo_idx_1 = TRACKED_ITEMS.index("firearm-magazine")
        ammo_idx_2 = TRACKED_ITEMS.index("piercing-rounds-magazine")
        has_ammo = inv[ammo_idx_1] > 0 or inv[ammo_idx_2] > 0
        if has_ammo and np.any(refill_slots):
            mask[ACTION_REFILL_TURRET] = 1

        # Can move only if the map has at least one anchor to move to.
        anchors = obs.get("anchor_valid_mask", np.zeros(MAX_ANCHORS))
        if np.any(anchors):
            mask[ACTION_MOVE_ANCHOR] = 1

        return mask

    def _slot_mask(self, obs: dict) -> np.ndarray:
        valid = obs.get("slot_valid_mask", np.zeros(MAX_SLOTS, dtype=np.int8))
        return np.asarray(valid, dtype=np.int8)

    def _anchor_mask(self, obs: dict) -> np.ndarray:
        valid = obs.get("anchor_valid_mask", np.zeros(MAX_ANCHORS, dtype=np.int8))
        return np.asarray(valid, dtype=np.int8)

    def _augment(self, obs: dict) -> dict:
        obs["action_type_mask"] = self._compute_action_type_mask(obs)
        obs["slot_mask"] = self._slot_mask(obs)
        obs["anchor_mask"] = self._anchor_mask(obs)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._augment(obs)
        self._obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._augment(obs)
        self._obs = obs
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return the action-type mask (for SB3 MaskablePPO compatibility)."""
        if not self._obs:
            mask = np.zeros(NUM_ACTION_TYPES, dtype=np.int8)
            mask[ACTION_NOOP] = 1
            return mask
        return self._compute_action_type_mask(self._obs)
