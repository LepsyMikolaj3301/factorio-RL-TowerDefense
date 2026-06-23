"""Unit tests for TowerDefenseEnv reward shaping."""

import pytest
import numpy as np

from fle.env.gym_env.td_config import TDScenarioConfig
from fle.env.gym_env.td_environment import TowerDefenseEnv
from fle.env.gym_env.td_spaces import (
    ACTION_MOVE_ANCHOR,
    ACTION_NOOP,
    ACTION_PICK_TURRET,
    ACTION_PLACE_TURRET,
    ACTION_REFILL_TURRET,
    ACTION_TAKE_AMMO,
    GROUP_FEATURES,
    MAX_GROUPS,
    MAX_NESTS,
    MAX_SLOTS,
    NEST_FEATURES,
    SLOT_FEATURES,
)


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
        p_empty_turret=0.0,
        p_move_command=0.0,
        p_move_transit=0.0,
        w_coverage_delta=0.0,
        w_ammo_delta=0.0,
        w_threatened_turret_delta=0.0,
        w_early_place_turret=0.0,
        w_early_refill_turret=0.0,
        early_turret_setup_steps=200,
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
    env._cur_empty_turrets = 0
    env._cur_threat_closeness = 0.0
    env._prev_threatened_turret_readiness = 0.0
    env._cur_threatened_turret_readiness = 0.0
    env._last_early_setup_reward = 0.0
    env._cur_slot_threat_alignment = np.ones(MAX_SLOTS, dtype=np.float32)
    env._cur_has_threat_direction = False
    env._step_count = 1
    env._moving = False
    env._norm = 32.0
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


def test_reward_penalizes_empty_live_turrets():
    env = _reward_env(p_empty_turret=0.25)
    env._cur_empty_turrets = 3

    reward = env._compute_reward(invalid=False, action={"action_type": ACTION_NOOP})

    assert reward == pytest.approx(-0.75)


def test_reward_penalizes_wall_losses_with_configured_weight():
    env = _reward_env(p_wall_destroyed=12.0)
    env._last_events["walls_lost"] = 2

    reward = env._compute_reward(invalid=False, action={"action_type": ACTION_NOOP})

    assert reward == pytest.approx(-24.0)


def test_reward_adds_valid_early_place_turret_bonus():
    env = _reward_env(w_early_place_turret=0.10)

    reward = env._compute_reward(
        invalid=False,
        action={"action_type": ACTION_PLACE_TURRET},
    )

    assert reward == pytest.approx(0.10)
    assert env._last_early_setup_reward == pytest.approx(0.10)


def test_reward_adds_valid_early_refill_turret_bonus():
    env = _reward_env(w_early_refill_turret=0.20)

    reward = env._compute_reward(
        invalid=False,
        action={"action_type": ACTION_REFILL_TURRET},
    )

    assert reward == pytest.approx(0.20)
    assert env._last_early_setup_reward == pytest.approx(0.20)


def test_reward_scales_early_place_bonus_by_threat_side_alignment():
    env = _reward_env(w_early_place_turret=0.10)
    env._cur_has_threat_direction = True
    env._cur_slot_threat_alignment[0] = 1.0
    env._cur_slot_threat_alignment[1] = 0.0

    same_side = env._compute_reward(
        invalid=False,
        action={"action_type": ACTION_PLACE_TURRET, "slot_index": 0},
    )
    opposite_side = env._compute_reward(
        invalid=False,
        action={"action_type": ACTION_PLACE_TURRET, "slot_index": 1},
    )

    assert same_side == pytest.approx(0.10)
    assert opposite_side == pytest.approx(0.0)
    assert env._last_early_setup_reward == pytest.approx(0.0)


def test_reward_scales_early_refill_bonus_by_threat_side_alignment():
    env = _reward_env(w_early_refill_turret=0.20)
    env._cur_has_threat_direction = True
    env._cur_slot_threat_alignment[0] = 0.25

    reward = env._compute_reward(
        invalid=False,
        action={"action_type": ACTION_REFILL_TURRET, "slot_index": 0},
    )

    assert reward == pytest.approx(0.05)
    assert env._last_early_setup_reward == pytest.approx(0.05)


def test_reward_does_not_reward_valid_early_pick_turret():
    env = _reward_env(w_early_place_turret=0.10, w_early_refill_turret=0.20)

    reward = env._compute_reward(
        invalid=False,
        action={"action_type": ACTION_PICK_TURRET},
    )

    assert reward == pytest.approx(0.0)
    assert env._last_early_setup_reward == pytest.approx(0.0)


def test_reward_does_not_reward_valid_early_take_ammo():
    env = _reward_env(w_early_place_turret=0.10, w_early_refill_turret=0.20)

    reward = env._compute_reward(
        invalid=False,
        action={"action_type": ACTION_TAKE_AMMO},
    )

    assert reward == pytest.approx(0.0)
    assert env._last_early_setup_reward == pytest.approx(0.0)


def test_reward_does_not_add_setup_bonus_for_invalid_actions():
    env = _reward_env(w_early_place_turret=0.10, w_early_refill_turret=0.20)

    place_reward = env._compute_reward(
        invalid=True,
        action={"action_type": ACTION_PLACE_TURRET},
    )
    refill_reward = env._compute_reward(
        invalid=True,
        action={"action_type": ACTION_REFILL_TURRET},
    )

    assert place_reward == pytest.approx(0.0)
    assert refill_reward == pytest.approx(0.0)
    assert env._last_early_setup_reward == pytest.approx(0.0)


def test_reward_does_not_add_setup_bonus_after_warmup():
    env = _reward_env(
        w_early_place_turret=0.10,
        w_early_refill_turret=0.20,
        early_turret_setup_steps=200,
    )
    env._step_count = 201

    place_reward = env._compute_reward(
        invalid=False,
        action={"action_type": ACTION_PLACE_TURRET},
    )
    refill_reward = env._compute_reward(
        invalid=False,
        action={"action_type": ACTION_REFILL_TURRET},
    )

    assert place_reward == pytest.approx(0.0)
    assert refill_reward == pytest.approx(0.0)
    assert env._last_early_setup_reward == pytest.approx(0.0)


def test_reward_adds_threatened_turret_readiness_delta_shaping():
    env = _reward_env(w_threatened_turret_delta=0.8)
    env._prev_threatened_turret_readiness = 0.25
    env._cur_threatened_turret_readiness = 0.75

    reward = env._compute_reward(invalid=False, action={"action_type": ACTION_NOOP})

    assert reward == pytest.approx(0.4)
    assert env._prev_threatened_turret_readiness == pytest.approx(0.75)


def test_threatened_readiness_prefers_turrets_on_incoming_side():
    env = _reward_env()
    slots = np.zeros((MAX_SLOTS, SLOT_FEATURES), dtype=np.float32)
    slot_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
    groups = np.zeros((MAX_GROUPS, GROUP_FEATURES), dtype=np.float32)
    group_mask = np.zeros(MAX_GROUPS, dtype=np.int8)
    nests = np.zeros((MAX_NESTS, NEST_FEATURES), dtype=np.float32)
    nest_mask = np.zeros(MAX_NESTS, dtype=np.int8)

    # East-side attack: the group is east of the radar and moving west.
    groups[0, :10] = [2.0, 0.0, 0.6, 0.0, -1.0, 0.0, 2.0, 0.0, 1.0, 1.0]
    group_mask[0] = 1
    slots[0, :5] = [1.0, 0.0, 1.0, 1.0, 1.0]
    slots[1, :5] = [-1.0, 0.0, 0.0, 0.0, 0.0]
    slot_mask[:2] = 1

    east_score = env._compute_threatened_turret_readiness(
        slots, slot_mask, groups, group_mask, nests, nest_mask
    )

    slots[0, :5] = [1.0, 0.0, 0.0, 0.0, 0.0]
    slots[1, :5] = [-1.0, 0.0, 1.0, 1.0, 1.0]
    west_score = env._compute_threatened_turret_readiness(
        slots, slot_mask, groups, group_mask, nests, nest_mask
    )

    assert east_score == pytest.approx(1.0)
    assert west_score == pytest.approx(0.0)


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
