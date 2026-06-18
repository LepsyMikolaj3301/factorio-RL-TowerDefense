"""Unit tests for Tower Defense observation and action spaces (slot-based)."""

import numpy as np

from fle.env.gym_env.td_spaces import (
    ACTION_NOOP,
    ACTION_PLACE_TURRET,
    ACTION_REFILL_TURRET,
    NUM_ACTION_TYPES,
    NUM_CHANNELS,
    DEFAULT_GRID_SIZE,
    MAX_SLOTS,
    SLOT_FEATURES,
    TRACKED_ITEMS,
    make_observation_space,
    make_action_space,
    flatten_action_space,
)


class TestTDSpaces:
    def test_action_type_values_unique(self):
        actions = [ACTION_NOOP, ACTION_PLACE_TURRET, ACTION_REFILL_TURRET]
        assert len(actions) == NUM_ACTION_TYPES
        assert len(set(actions)) == NUM_ACTION_TYPES

    def test_action_types_sequential(self):
        assert ACTION_NOOP == 0
        assert ACTION_PLACE_TURRET == 1
        assert ACTION_REFILL_TURRET == 2
        assert NUM_ACTION_TYPES == 3

    def test_observation_space_shape(self):
        obs_space = make_observation_space()
        assert obs_space["map"].shape == (
            NUM_CHANNELS,
            DEFAULT_GRID_SIZE,
            DEFAULT_GRID_SIZE,
        )

    def test_observation_has_slot_fields(self):
        obs_space = make_observation_space()
        assert obs_space["turret_slots"].shape == (MAX_SLOTS, SLOT_FEATURES)
        assert obs_space["slot_valid_mask"].shape == (MAX_SLOTS,)
        assert obs_space["place_slot_mask"].shape == (MAX_SLOTS,)
        assert obs_space["refill_slot_mask"].shape == (MAX_SLOTS,)
        # Old free-placement fields are gone.
        assert "turrets" not in obs_space.spaces
        assert "walls" not in obs_space.spaces

    def test_action_space_is_slot_based(self):
        act_space = make_action_space()
        assert set(act_space.spaces) == {"action_type", "slot_index", "ammo_amount"}
        assert act_space["action_type"].n == NUM_ACTION_TYPES
        assert act_space["slot_index"].n == MAX_SLOTS
        # No raw coordinate fields anymore.
        assert "target_x" not in act_space.spaces
        assert "source_x" not in act_space.spaces

    def test_action_space_sample(self):
        act_space = make_action_space()
        sample = act_space.sample()
        assert 0 <= sample["action_type"] < NUM_ACTION_TYPES
        assert 0 <= sample["slot_index"] < MAX_SLOTS

    def test_flatten_action_space_layout(self):
        flat = flatten_action_space()
        assert list(flat.nvec) == [NUM_ACTION_TYPES, MAX_SLOTS, 51]

    def test_tracked_items_not_empty(self):
        assert "firearm-magazine" in TRACKED_ITEMS
        assert "gun-turret" in TRACKED_ITEMS

    def test_constants(self):
        assert NUM_CHANNELS == 8
        assert DEFAULT_GRID_SIZE == 64
        assert MAX_SLOTS == 64
        assert SLOT_FEATURES == 5
