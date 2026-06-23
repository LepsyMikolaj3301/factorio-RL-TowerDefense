"""E2E smoke tests for TowerDefenseEnv against a live Factorio server.

All tests are skipped automatically if no Factorio containers are running
(handled by the session-scoped `td_env` fixture in conftest.py).

Start containers first:
    fle cluster start --num-instances 1 --scenario tower_defense
"""

import math

import numpy as np
import pytest


def _char_pos(env):
    raw = env.instance.rcon_client.send_command(
        "/sc local c=storage.agent_characters and storage.agent_characters[1]; "
        "if c and c.valid then rcon.print(c.position.x..','..c.position.y) "
        "else rcon.print('nil') end"
    ).strip()
    if not raw or raw == "nil":
        pytest.skip("No agent character available to inspect.")
    x, y = raw.split(",")
    return float(x), float(y)


class TestTowerDefenseSmoke:
    """Smoke tests that require a live Factorio server."""

    def test_reset_returns_valid_obs(self, td_env):
        """reset() returns an obs dict with correct keys, shapes, and dtypes."""
        obs, info = td_env.reset()

        expected_keys = set(td_env.observation_space.spaces)
        assert set(obs.keys()) == expected_keys

        space = td_env.observation_space
        for key, arr in obs.items():
            assert space[key].contains(arr), (
                f"obs['{key}'] shape {arr.shape} dtype {arr.dtype} "
                f"not in space {space[key]}"
            )

        assert obs["map"].dtype == np.uint8
        assert obs["inventory"].dtype == np.int32
        assert obs["game"].shape == (2,)

    def test_step_returns_five_tuple(self, td_env):
        """step() returns (obs, reward, terminated, truncated, info) with correct types."""
        td_env.reset()
        action = td_env.action_space.sample()
        result = td_env.step(action)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

        assert isinstance(reward, float)
        assert math.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "elapsed_ticks" in info

    def test_noop_does_not_crash(self, td_env):
        """Stepping with NOOP action returns valid obs and finite reward."""
        from fle.env.gym_env.td_spaces import ACTION_NOOP

        td_env.reset()
        action = {
            "action_type": ACTION_NOOP,
            "slot_index": 0,
            "ammo_amount": 0,
        }
        obs, reward, terminated, truncated, info = td_env.step(action)

        assert math.isfinite(reward)
        assert not info.get("invalid_action", False)

    def test_elapsed_ticks_increment(self, td_env):
        """Elapsed ticks in game obs increase over consecutive steps."""
        from fle.env.gym_env.td_spaces import ACTION_NOOP

        obs, _ = td_env.reset()
        initial_ticks = obs["game"][0]

        noop = {
            "action_type": ACTION_NOOP,
            "slot_index": 0,
            "ammo_amount": 0,
        }
        for _ in range(3):
            obs, _, terminated, truncated, _ = td_env.step(noop)
            if terminated or truncated:
                break

        assert obs["game"][0] > initial_ticks, "Elapsed ticks did not increase after steps"

    def test_move_anchor_moves_character_and_finishes(self, td_env):
        """MOVE_ANCHOR advances the live character and eventually clears is_moving."""
        from fle.env.gym_env.td_spaces import ACTION_MOVE_ANCHOR, ACTION_NOOP

        td_env.reset()
        anchors = td_env.unwrapped._anchor_slots
        if anchors is None or len(anchors) == 0:
            pytest.skip("Map has no anchor slots.")

        sx, sy = _char_pos(td_env)
        distances = np.hypot(anchors[:, 0] - sx, anchors[:, 1] - sy)
        target = int(np.argmax(distances))
        if float(distances[target]) <= 0.5:
            pytest.skip("No anchor far enough from the current character position.")

        move = {
            "action_type": ACTION_MOVE_ANCHOR,
            "slot_index": 0,
            "ammo_amount": 0,
            "anchor_index": target,
        }
        _, _, terminated, truncated, info = td_env.step(move)
        if terminated or truncated:
            pytest.skip("Episode ended during MOVE_ANCHOR step.")
        assert not info.get("invalid_action", False)

        mx, my = _char_pos(td_env)
        assert math.hypot(mx - sx, my - sy) > 0.5, "character did not move after MOVE_ANCHOR"

        noop = {
            "action_type": ACTION_NOOP,
            "slot_index": 0,
            "ammo_amount": 0,
            "anchor_index": 0,
        }
        last_info = info
        for _ in range(20):
            if not last_info.get("is_moving", False):
                break
            _, _, terminated, truncated, last_info = td_env.step(noop)
            if terminated or truncated:
                pytest.skip("Episode ended while waiting for anchor movement to finish.")

        assert not last_info.get("is_moving", False), "MOVE_ANCHOR stayed active"
        fx, fy = _char_pos(td_env)
        tx, ty = float(anchors[target, 0]), float(anchors[target, 1])
        assert math.hypot(fx - tx, fy - ty) <= 1.0

    def test_reset_is_idempotent(self, td_env):
        """Calling reset() twice returns obs with the same shapes both times."""
        obs1, _ = td_env.reset()
        obs2, _ = td_env.reset()

        assert set(obs1.keys()) == set(obs2.keys())
        for key in obs1:
            assert obs1[key].shape == obs2[key].shape, (
                f"obs['{key}'] shape changed between resets: "
                f"{obs1[key].shape} vs {obs2[key].shape}"
            )

    def test_check_env(self, td_env):
        """gymnasium check_env passes without errors."""
        from gymnasium.utils.env_checker import check_env

        # check_env calls reset() and step() internally
        check_env(td_env, warn=True)

    def test_slot_masks_partition_real_slots(self, td_env):
        """place + refill masks are disjoint and together cover the real slots."""
        obs, _ = td_env.reset()

        valid = np.asarray(obs["slot_valid_mask"], dtype=bool)
        place = np.asarray(obs["place_slot_mask"], dtype=bool)
        refill = np.asarray(obs["refill_slot_mask"], dtype=bool)

        # Every place/refill slot is a real slot.
        assert np.all(place <= valid)
        assert np.all(refill <= valid)
        # A slot is either empty (place) or occupied (refill), never both.
        assert not np.any(place & refill)
        # Real slots are exactly the union of empty and occupied.
        assert np.array_equal(valid, place | refill)

    def test_deletion_leaves_empty_slots(self, td_env):
        """With deletion_percentage > 0 and slots present, some start empty."""
        obs, _ = td_env.reset(seed=0)
        n_slots = int(np.asarray(obs["slot_valid_mask"]).sum())
        if n_slots == 0:
            pytest.skip("Map has no turret slots; nothing to test.")
        if td_env.unwrapped.config.turret_deletion_percentage <= 0:
            pytest.skip("Deletion disabled for this config.")
        n_empty = int(np.asarray(obs["place_slot_mask"]).sum())
        assert n_empty > 0, "Expected at least one deleted (empty) slot"

    def test_place_into_empty_slot_occupies_it(self, td_env):
        """PLACE_TURRET on an empty slot makes that slot occupied next obs."""
        from fle.env.gym_env.td_spaces import ACTION_PLACE_TURRET

        obs, _ = td_env.reset(seed=0)
        place = np.asarray(obs["place_slot_mask"], dtype=bool)
        empty_slots = np.flatnonzero(place)
        if empty_slots.size == 0:
            pytest.skip("No empty slot available to place into.")

        slot = int(empty_slots[0])
        action = {
            "action_type": ACTION_PLACE_TURRET,
            "slot_index": slot,
            "ammo_amount": 0,
        }
        obs2, _, terminated, truncated, info = td_env.step(action)
        if terminated or truncated:
            pytest.skip("Episode ended during the step.")
        assert not info.get("invalid_action", False)
        assert bool(np.asarray(obs2["refill_slot_mask"], dtype=bool)[slot])
