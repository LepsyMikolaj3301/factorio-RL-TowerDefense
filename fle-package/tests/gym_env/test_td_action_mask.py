"""Unit tests for Tower Defense action masks."""

import gymnasium
import numpy as np

from fle.env.gym_env.action_mask import ActionMaskWrapper
from fle.env.gym_env.td_spaces import (
    ACTION_MOVE_ANCHOR,
    ACTION_TAKE_AMMO,
    AMMO_AMOUNT_LEVELS,
    MAX_ANCHORS,
    MAX_SLOTS,
    NUM_ACTION_TYPES,
    TRACKED_ITEMS,
    make_action_space,
    make_observation_space,
)


class DummyTDEnv(gymnasium.Env):
    def __init__(self):
        self.observation_space = make_observation_space()
        self.action_space = make_action_space()
        self._current_anchor_index = 1
        self._move_target = -1
        self._moving = False


def _obs_with_anchors():
    obs = {
        "anchor_valid_mask": np.zeros(MAX_ANCHORS, dtype=np.int8),
        "movement": np.zeros(5, dtype=np.float32),
        "inventory": np.zeros(len(TRACKED_ITEMS), dtype=np.int32),
        "place_reach_mask": np.zeros(MAX_SLOTS, dtype=np.int8),
        "refill_reach_mask": np.zeros(MAX_SLOTS, dtype=np.int8),
        "pick_reach_mask": np.zeros(MAX_SLOTS, dtype=np.int8),
        "take_ammo_reach_mask": np.zeros(MAX_SLOTS, dtype=np.int8),
    }
    obs["anchor_valid_mask"][:3] = 1
    return obs


def test_anchor_mask_hides_current_anchor_when_stationary():
    wrapper = ActionMaskWrapper(DummyTDEnv())
    mask = wrapper._anchor_mask(_obs_with_anchors())

    assert mask[0] == 1
    assert mask[1] == 0
    assert mask[2] == 1


def test_action_type_mask_disables_move_if_only_current_anchor_exists():
    env = DummyTDEnv()
    env._current_anchor_index = 0
    wrapper = ActionMaskWrapper(env)
    obs = _obs_with_anchors()
    obs["anchor_valid_mask"][:] = 0
    obs["anchor_valid_mask"][0] = 1

    mask = wrapper._compute_action_type_mask(obs)

    assert mask[ACTION_MOVE_ANCHOR] == 0


def test_action_type_mask_enables_take_ammo_for_reachable_loaded_turret():
    wrapper = ActionMaskWrapper(DummyTDEnv())
    obs = _obs_with_anchors()
    obs["take_ammo_reach_mask"][4] = 1

    mask = wrapper._compute_action_type_mask(obs)

    assert mask[ACTION_TAKE_AMMO] == 1


def test_action_type_mask_disables_take_ammo_without_reachable_loaded_turret():
    wrapper = ActionMaskWrapper(DummyTDEnv())
    obs = _obs_with_anchors()

    mask = wrapper._compute_action_type_mask(obs)

    assert mask[ACTION_TAKE_AMMO] == 0


def test_action_type_mask_disables_take_ammo_while_moving():
    wrapper = ActionMaskWrapper(DummyTDEnv())
    obs = _obs_with_anchors()
    obs["take_ammo_reach_mask"][4] = 1
    obs["movement"][0] = 1.0

    mask = wrapper._compute_action_type_mask(obs)

    assert mask[ACTION_TAKE_AMMO] == 0


def test_flat_action_mask_includes_take_ammo_slots_and_new_width():
    wrapper = ActionMaskWrapper(DummyTDEnv())
    obs = _obs_with_anchors()
    obs["take_ammo_reach_mask"][7] = 1

    mask = wrapper._flat_action_mask(obs)

    assert mask.shape == (NUM_ACTION_TYPES + MAX_SLOTS + AMMO_AMOUNT_LEVELS + MAX_ANCHORS,)
    assert mask[ACTION_TAKE_AMMO] == 1
    slot_offset = NUM_ACTION_TYPES
    assert mask[slot_offset + 7] == 1
