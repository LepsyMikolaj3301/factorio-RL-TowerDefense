"""Unit tests for Tower Defense observation and action spaces."""

import numpy as np
import pytest

from fle.env.gym_env.td_spaces import (
    ACTION_NOOP,
    ACTION_PLACE_WALL,
    ACTION_PLACE_TURRET,
    ACTION_MOVE_WALL,
    ACTION_MOVE_TURRET,
    ACTION_REFILL_TURRET,
    ACTION_SHOOT,
    NUM_ACTION_TYPES,
    NUM_CHANNELS,
    DEFAULT_GRID_SIZE,
    TRACKED_ITEMS,
    MAX_TURRETS,
    MAX_WALLS,
    make_observation_space,
    make_action_space,
)


class TestTDSpaces:
    def test_action_type_values_unique(self):
        actions = [
            ACTION_NOOP, ACTION_PLACE_WALL, ACTION_PLACE_TURRET,
            ACTION_MOVE_WALL, ACTION_MOVE_TURRET, ACTION_REFILL_TURRET,
            ACTION_SHOOT,
        ]
        assert len(actions) == NUM_ACTION_TYPES
        assert len(set(actions)) == NUM_ACTION_TYPES

    def test_action_types_sequential(self):
        assert ACTION_NOOP == 0
        assert ACTION_SHOOT == 6

    def test_observation_space_shape(self):
        obs_space = make_observation_space()
        assert "map" in obs_space.spaces
        map_shape = obs_space["map"].shape
        assert map_shape == (NUM_CHANNELS, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)

    def test_observation_space_custom_grid(self):
        obs_space = make_observation_space(grid_size=32)
        assert obs_space["map"].shape == (NUM_CHANNELS, 32, 32)

    def test_action_space_contains_required_keys(self):
        act_space = make_action_space()
        assert "action_type" in act_space.spaces
        assert "target_x" in act_space.spaces
        assert "target_y" in act_space.spaces

    def test_action_space_sample(self):
        act_space = make_action_space()
        sample = act_space.sample()
        assert "action_type" in sample
        assert 0 <= sample["action_type"] < NUM_ACTION_TYPES

    def test_tracked_items_not_empty(self):
        assert len(TRACKED_ITEMS) > 0
        assert "firearm-magazine" in TRACKED_ITEMS
        assert "stone-wall" in TRACKED_ITEMS
        assert "gun-turret" in TRACKED_ITEMS

    def test_constants(self):
        assert NUM_CHANNELS == 8
        assert DEFAULT_GRID_SIZE == 64
        assert MAX_TURRETS == 50
        assert MAX_WALLS == 200
