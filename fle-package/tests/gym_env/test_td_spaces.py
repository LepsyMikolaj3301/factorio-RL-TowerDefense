"""Unit tests for Tower Defense observation and action spaces (slot-based)."""

import numpy as np

from fle.env.gym_env.td_spaces import (
    ACTION_NOOP,
    ACTION_PICK_TURRET,
    ACTION_PLACE_TURRET,
    ACTION_REFILL_TURRET,
    ACTION_MOVE_ANCHOR,
    ACTION_TAKE_AMMO,
    AMMO_AMOUNT_LEVELS,
    NUM_ACTION_TYPES,
    NUM_CHANNELS,
    DEFAULT_GRID_SIZE,
    MAX_SLOTS,
    MAX_ANCHORS,
    MAX_BOILERS,
    MAX_GROUPS,
    MAX_NESTS,
    BOILER_FEATURES,
    GROUP_FEATURES,
    NEST_FEATURES,
    MOVEMENT_FEATURES,
    RECENT_LOSS_FEATURES,
    SLOT_FEATURES,
    TRACKED_ITEMS,
    make_observation_space,
    make_action_space,
    flatten_action_space,
)


class TestTDSpaces:
    def test_action_type_values_unique(self):
        actions = [
            ACTION_NOOP,
            ACTION_PICK_TURRET,
            ACTION_PLACE_TURRET,
            ACTION_REFILL_TURRET,
            ACTION_MOVE_ANCHOR,
            ACTION_TAKE_AMMO,
        ]
        assert len(actions) == NUM_ACTION_TYPES
        assert len(set(actions)) == NUM_ACTION_TYPES

    def test_action_types_sequential(self):
        assert ACTION_NOOP == 0
        assert ACTION_PICK_TURRET == 1
        assert ACTION_PLACE_TURRET == 2
        assert ACTION_REFILL_TURRET == 3
        assert ACTION_MOVE_ANCHOR == 4
        assert ACTION_TAKE_AMMO == 5
        assert NUM_ACTION_TYPES == 6

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
        assert obs_space["take_ammo_slot_mask"].shape == (MAX_SLOTS,)
        # Old free-placement fields are gone.
        assert "turrets" not in obs_space.spaces
        assert "walls" not in obs_space.spaces

    def test_observation_has_threat_and_movement_fields(self):
        obs_space = make_observation_space()
        assert obs_space["biter_groups"].shape == (MAX_GROUPS, GROUP_FEATURES)
        assert obs_space["group_valid_mask"].shape == (MAX_GROUPS,)
        assert obs_space["nests"].shape == (MAX_NESTS, NEST_FEATURES)
        assert obs_space["nest_valid_mask"].shape == (MAX_NESTS,)
        assert obs_space["boilers"].shape == (MAX_BOILERS, BOILER_FEATURES)
        assert obs_space["boiler_valid_mask"].shape == (MAX_BOILERS,)
        assert obs_space["movement"].shape == (MOVEMENT_FEATURES,)
        assert obs_space["recent_losses"].shape == (RECENT_LOSS_FEATURES,)
        # Action-conditioned reach masks for the pointer head.
        for k in (
            "place_reach_mask",
            "refill_reach_mask",
            "pick_reach_mask",
            "take_ammo_reach_mask",
        ):
            assert obs_space[k].shape == (MAX_SLOTS,)

    def test_observation_space_contains_sample(self):
        obs_space = make_observation_space()
        assert obs_space.contains(obs_space.sample())

    def test_action_space_is_slot_based(self):
        act_space = make_action_space()
        assert set(act_space.spaces) == {
            "action_type",
            "slot_index",
            "ammo_amount",
            "anchor_index",
        }
        assert act_space["action_type"].n == NUM_ACTION_TYPES
        assert act_space["slot_index"].n == MAX_SLOTS
        assert act_space["anchor_index"].n == MAX_ANCHORS
        # No raw coordinate fields anymore.
        assert "target_x" not in act_space.spaces
        assert "source_x" not in act_space.spaces

    def test_action_space_sample(self):
        act_space = make_action_space()
        sample = act_space.sample()
        assert 0 <= sample["action_type"] < NUM_ACTION_TYPES
        assert 0 <= sample["slot_index"] < MAX_SLOTS
        assert 0 <= sample["anchor_index"] < MAX_ANCHORS

    def test_flatten_action_space_layout(self):
        flat = flatten_action_space()
        assert list(flat.nvec) == [
            NUM_ACTION_TYPES,
            MAX_SLOTS,
            AMMO_AMOUNT_LEVELS,
            MAX_ANCHORS,
        ]

    def test_tracked_items_not_empty(self):
        assert "firearm-magazine" in TRACKED_ITEMS
        assert "gun-turret" in TRACKED_ITEMS

    def test_constants(self):
        assert NUM_CHANNELS == 8
        assert DEFAULT_GRID_SIZE == 64
        assert MAX_SLOTS == 64
        assert SLOT_FEATURES == 5
