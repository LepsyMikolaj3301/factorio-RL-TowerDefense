"""Unit tests for TowerDefenseEnv - space conformance and action encoding."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

import gymnasium

from fle.env.gym_env.td_spaces import (
    NUM_ACTION_TYPES,
    NUM_CHANNELS,
    DEFAULT_GRID_SIZE,
    TRACKED_ITEMS,
    make_observation_space,
    make_action_space,
)


class TestTDEnvironmentSpaces:
    """Test space definitions without requiring a Factorio connection."""

    def test_observation_space_is_gymnasium_dict(self):
        obs_space = make_observation_space()
        assert isinstance(obs_space, gymnasium.spaces.Dict)

    def test_action_space_is_gymnasium_dict(self):
        act_space = make_action_space()
        assert isinstance(act_space, gymnasium.spaces.Dict)

    def test_obs_space_sample_valid(self):
        obs_space = make_observation_space()
        sample = obs_space.sample()
        assert obs_space.contains(sample)

    def test_action_space_sample_valid(self):
        act_space = make_action_space()
        sample = act_space.sample()
        assert act_space.contains(sample)

    def test_map_channel_range(self):
        obs_space = make_observation_space()
        map_space = obs_space["map"]
        assert map_space.low.min() == 0
        assert map_space.high.max() == 255

    def test_inventory_space_shape(self):
        obs_space = make_observation_space()
        inv_space = obs_space["inventory"]
        assert inv_space.shape == (len(TRACKED_ITEMS),)

    def test_custom_grid_size(self):
        gs = 32
        obs_space = make_observation_space(grid_size=gs)
        act_space = make_action_space(grid_size=gs)
        assert obs_space["map"].shape == (NUM_CHANNELS, gs, gs)
        assert act_space["target_x"].n == gs
