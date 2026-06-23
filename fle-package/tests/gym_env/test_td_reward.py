"""Unit tests for TowerDefenseEnv reward shaping."""

import pytest
import numpy as np

from fle.env.gym_env.td_config import TDScenarioConfig
from fle.env.gym_env.td_environment import TowerDefenseEnv
from fle.env.gym_env.td_spaces import ACTION_MOVE_ANCHOR, ACTION_NOOP


def _reward_env(**cfg_overrides):
    cfg = TDScenarioConfig(
        alpha_survive=0.0,
        beta_kills=0.0,
        delta_damage=0.0,
        epsilon_invalid=0.0,
        p_wall_destroyed=0.0,
        p_turret_destroyed=0.0,
        p_building_destroyed=0.0,
        p_radar_damage=0.0,
        p_move_command=0.0,
        p_move_transit=0.0,
        w_coverage_delta=0.0,
        w_ammo_delta=0.0,
        w_threat=0.0,
        shaping_decay_steps=100,
    )
    for key, value in cfg_overrides.items():
        setattr(cfg, key, value)
    env = TowerDefenseEnv.__new__(TowerDefenseEnv)
    env.config = cfg
    env._last_events = TowerDefenseEnv._zero_events()
    env._prev_health = 100.0
    env._cur_char_hp = 100.0
    env._prev_radar_hp = 250.0
    env._cur_radar_hp = 250.0
    env._total_steps = 0
    env._prev_coverage = 0.0
    env._cur_coverage = 0.0
    env._prev_ammo_frac = 0.0
    env._cur_ammo_frac = 0.0
    env._cur_threat_closeness = 0.0
    env._moving = False
    return env


def test_reward_penalizes_valid_move_command():
    env = _reward_env(p_move_command=0.05)

    reward = env._compute_reward(
        invalid=False,
        action={"action_type": ACTION_MOVE_ANCHOR},
    )

    assert reward == pytest.approx(-0.05)


def test_reward_penalizes_transit_step():
    env = _reward_env(p_move_transit=0.01)
    env._moving = True

    reward = env._compute_reward(
        invalid=False,
        action={"action_type": ACTION_NOOP},
    )

    assert reward == pytest.approx(-0.01)


def test_reward_adds_coverage_delta_shaping():
    env = _reward_env(w_coverage_delta=0.5)
    env._prev_coverage = 0.2
    env._cur_coverage = 0.6

    reward = env._compute_reward(invalid=False, action={"action_type": ACTION_NOOP})

    assert reward == pytest.approx(0.2)
    assert env._prev_coverage == pytest.approx(0.6)


def test_reward_adds_ammo_delta_shaping():
    env = _reward_env(w_ammo_delta=0.2)
    env._prev_ammo_frac = 0.25
    env._cur_ammo_frac = 0.75

    reward = env._compute_reward(invalid=False, action={"action_type": ACTION_NOOP})

    assert reward == pytest.approx(0.1)
    assert env._prev_ammo_frac == pytest.approx(0.75)


def test_redundant_stationary_anchor_move_is_invalid():
    env = _reward_env()
    env._anchor_slots = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    env._current_anchor_index = 0
    env._moving = False

    assert env._move_to_anchor(0) is True


def test_redundant_inflight_anchor_move_is_invalid():
    env = _reward_env()
    env._anchor_slots = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    env._move_target = 1
    env._moving = True

    assert env._move_to_anchor(1) is True


def test_terminates_when_one_third_boilers_destroyed():
    env = _reward_env()
    env._step_count = 2
    env._has_radar = True
    env._cur_radar_hp = 250.0
    env._cur_char_hp = 250.0
    env.config.boiler_loss_fraction = 1.0 / 3.0
    obs = {
        "boiler_valid_mask": np.array([1, 1, 1], dtype=np.int8),
        "boilers": np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [2.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }

    assert env._check_terminated(obs) is True


def test_does_not_terminate_below_boiler_loss_threshold():
    env = _reward_env()
    env._step_count = 2
    env._has_radar = True
    env._cur_radar_hp = 250.0
    env._cur_char_hp = 250.0
    env.config.boiler_loss_fraction = 1.0 / 3.0
    obs = {
        "boiler_valid_mask": np.array([1, 1, 1, 1], dtype=np.int8),
        "boilers": np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [2.0, 0.0, 1.0, 1.0],
                [3.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }

    assert env._check_terminated(obs) is False
