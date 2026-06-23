"""Unit tests for Tower Defense action masks."""

import gymnasium
import numpy as np

from fle.env.gym_env.action_mask import ActionMaskWrapper
from fle.env.gym_env.td_spaces import (
    ACTION_MOVE_ANCHOR,
    MAX_ANCHORS,
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
