"""Action masking wrapper for Tower Defense environments."""

import numpy as np
import gymnasium
from gymnasium import spaces

from fle.env.gym_env.td_spaces import (
    ACTION_MOVE_ANCHOR,
    ACTION_NOOP,
    ACTION_PICK_TURRET,
    ACTION_PLACE_TURRET,
    ACTION_REFILL_TURRET,
    MAX_ANCHORS,
    MAX_SLOTS,
    NUM_ACTION_TYPES,
    TRACKED_ITEMS,
)

# Number of discrete ammo levels in the flat MultiDiscrete action factor
# (must match td_spaces.make_action_space / flatten_action_space: Discrete(51)).
NUM_AMMO_LEVELS = 51


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
        """Compute the per-action-type mask from transit, reach, and inventory.

        Two structural rules:
          * While in transit (``movement[0] == 1``) only NOOP and MOVE_ANCHOR are
            legal — the agent may re-route but cannot service a turret mid-walk.
          * Turret actions use the action-conditioned *reach* masks, so PLACE /
            REFILL / PICK are only offered when a valid slot is reachable from the
            character's current position (these are already zeroed while moving).
        """
        mask = np.zeros(NUM_ACTION_TYPES, dtype=np.int8)
        mask[ACTION_NOOP] = 1  # Noop always valid

        # Can move whenever the map has at least one anchor to move to.
        anchors = self._anchor_mask(obs)
        if np.any(anchors):
            mask[ACTION_MOVE_ANCHOR] = 1

        movement = obs.get("movement")
        is_moving = (
            movement is not None and len(movement) > 0 and bool(movement[0])
        )
        if is_moving:
            # In transit: only re-route (MOVE_ANCHOR) or wait (NOOP).
            return mask

        inv = obs.get("inventory", np.zeros(len(TRACKED_ITEMS)))
        place_reach = obs.get("place_reach_mask", np.zeros(MAX_SLOTS))
        refill_reach = obs.get("refill_reach_mask", np.zeros(MAX_SLOTS))
        pick_reach = obs.get("pick_reach_mask", np.zeros(MAX_SLOTS))

        # Place: need a turret in inventory AND a reachable empty slot.
        turret_idx = TRACKED_ITEMS.index("gun-turret")
        if inv[turret_idx] > 0 and np.any(place_reach):
            mask[ACTION_PLACE_TURRET] = 1

        # Refill: need ammo AND a reachable occupied slot.
        ammo_idx_1 = TRACKED_ITEMS.index("firearm-magazine")
        ammo_idx_2 = TRACKED_ITEMS.index("piercing-rounds-magazine")
        has_ammo = inv[ammo_idx_1] > 0 or inv[ammo_idx_2] > 0
        if has_ammo and np.any(refill_reach):
            mask[ACTION_REFILL_TURRET] = 1

        # Pick: need a reachable occupied slot (turret to mine back up).
        if np.any(pick_reach):
            mask[ACTION_PICK_TURRET] = 1

        return mask

    def _slot_mask(self, obs: dict) -> np.ndarray:
        valid = obs.get("slot_valid_mask", np.zeros(MAX_SLOTS, dtype=np.int8))
        return np.asarray(valid, dtype=np.int8)

    def _anchor_mask(self, obs: dict) -> np.ndarray:
        valid = obs.get("anchor_valid_mask", np.zeros(MAX_ANCHORS, dtype=np.int8))
        mask = np.asarray(valid, dtype=np.int8).copy()
        try:
            unwrapped = self.env.unwrapped
            target = (
                int(getattr(unwrapped, "_move_target", -1))
                if getattr(unwrapped, "_moving", False)
                else int(getattr(unwrapped, "_current_anchor_index", -1))
            )
            if 0 <= target < len(mask):
                mask[target] = 0
        except Exception:
            pass
        return mask

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

    def _flat_action_mask(self, obs: dict) -> np.ndarray:
        """Full flat mask for the MultiDiscrete action [type, slot, ammo, anchor].

        Length = NUM_ACTION_TYPES + MAX_SLOTS + NUM_AMMO_LEVELS + MAX_ANCHORS.
        MaskablePPO masks each factor independently and requires >= 1 valid option
        per factor, so the slot/anchor factors fall back to index 0 when no
        constrained option exists (the env ignores irrelevant factors for the
        chosen action type, and the invalid-action penalty handles residue).
        """
        type_mask = self._compute_action_type_mask(obs)

        # Slot factor: any slot valid for some currently-legal turret action.
        place = obs.get("place_reach_mask", np.zeros(MAX_SLOTS, dtype=np.int8))
        refill = obs.get("refill_reach_mask", np.zeros(MAX_SLOTS, dtype=np.int8))
        pick = obs.get("pick_reach_mask", np.zeros(MAX_SLOTS, dtype=np.int8))
        slot_mask = (
            np.asarray(place, np.int8)
            | np.asarray(refill, np.int8)
            | np.asarray(pick, np.int8)
        )
        if not slot_mask.any():
            slot_mask[0] = 1  # fallback so the factor is samplable

        # Ammo factor: unconstrained.
        ammo_mask = np.ones(NUM_AMMO_LEVELS, dtype=np.int8)

        # Anchor factor: real anchors, fallback to 0.
        anchor_mask = self._anchor_mask(obs).copy()
        if not anchor_mask.any():
            anchor_mask[0] = 1

        return np.concatenate([type_mask, slot_mask, ammo_mask, anchor_mask]).astype(
            np.int8
        )

    def action_masks(self) -> np.ndarray:
        """Return the full flat MultiDiscrete mask (for SB3 MaskablePPO)."""
        if not self._obs:
            # Pre-reset fallback: only NOOP + slot0 + ammo + anchor0 valid.
            type_mask = np.zeros(NUM_ACTION_TYPES, dtype=np.int8)
            type_mask[ACTION_NOOP] = 1
            slot_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
            slot_mask[0] = 1
            ammo_mask = np.ones(NUM_AMMO_LEVELS, dtype=np.int8)
            anchor_mask = np.zeros(MAX_ANCHORS, dtype=np.int8)
            anchor_mask[0] = 1
            return np.concatenate([type_mask, slot_mask, ammo_mask, anchor_mask])
        return self._flat_action_mask(self._obs)
