"""Unit tests for TowerDefenseEnv - space conformance and action encoding."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

import gymnasium

from fle.env.gym_env.td_spaces import (
    ACTION_TAKE_AMMO,
    NUM_ACTION_TYPES,
    NUM_CHANNELS,
    DEFAULT_GRID_SIZE,
    TRACKED_ITEMS,
    make_observation_space,
    make_action_space,
)
from fle.env.gym_env.td_config import TDScenarioConfig
from fle.env.gym_env.td_environment import TowerDefenseEnv


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
        # The action space is slot-based and independent of grid size.
        assert "slot_index" in act_space.spaces

    def test_take_ammo_action_moves_ammo_without_picking_turret(self):
        env = TowerDefenseEnv.__new__(TowerDefenseEnv)
        env.config = TDScenarioConfig(ammo_type="piercing-rounds-magazine")
        env._turret_slots = np.array([[4.0, 5.0]], dtype=np.float32)
        env._slot_epsilon = 0.6
        env._moving = False
        env._last_ammo_taken = 0
        env._find_turret_at = MagicMock(return_value=object())
        env.instance = MagicMock()
        env.instance.rcon_client.send_command.return_value = "7"

        invalid = env._execute_action(
            {
                "action_type": ACTION_TAKE_AMMO,
                "slot_index": 0,
                "ammo_amount": 10,
            }
        )

        assert invalid is False
        assert env._last_ammo_taken == 7
        env._find_turret_at.assert_called_once_with(4.0, 5.0)
        env.instance.first_namespace.pickup_entity.assert_not_called()
        command = env.instance.rcon_client.send_command.call_args.args[0]
        assert "piercing-rounds-magazine" in command
        assert "position={4.0,5.0}" in command

    def test_take_ammo_action_is_invalid_when_no_ammo_moves(self):
        env = TowerDefenseEnv.__new__(TowerDefenseEnv)
        env.config = TDScenarioConfig(ammo_type="piercing-rounds-magazine")
        env._turret_slots = np.array([[4.0, 5.0]], dtype=np.float32)
        env._slot_epsilon = 0.6
        env._moving = False
        env._last_ammo_taken = 3
        env._find_turret_at = MagicMock(return_value=object())
        env.instance = MagicMock()
        env.instance.rcon_client.send_command.return_value = "0"

        invalid = env._execute_action(
            {
                "action_type": ACTION_TAKE_AMMO,
                "slot_index": 0,
                "ammo_amount": 10,
            }
        )

        assert invalid is True
        assert env._last_ammo_taken == 0
