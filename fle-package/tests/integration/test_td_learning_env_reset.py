"""Focused integration test for TowerDefenseEnv episode reset.

This is the regression test for the training-loop failure mode where
``env.reset()`` restored from a GameState snapshot but the authored TD entities
did not come back: no radar, no turrets, zero coverage, and immediate terminal
episodes.

Run:
    fle cluster start -n 1
    cd fle-package
    pytest -m integration tests/integration/test_td_learning_env_reset.py -v -s
"""

import math

import pytest

from fle.env.gym_env.td_config import TDScenarioConfig
from fle.env.gym_env.td_environment import TowerDefenseEnv

pytestmark = pytest.mark.integration


def _count(inst, lua_filter: str) -> int:
    raw = inst.rcon_client.send_command(
        "/sc local n=0 for _,e in pairs(game.surfaces[1]"
        f".find_entities_filtered{{{lua_filter}}}) do n=n+1 end rcon.print(n)"
    ).strip()
    return int(float(raw or 0))


def _destroy(inst, lua_filter: str) -> None:
    inst.rcon_client.send_command(
        "/sc for _,e in pairs(game.surfaces[1]"
        f".find_entities_filtered{{{lua_filter}}}) do e.destroy() end"
    )


def _infinity_filter_count(inst) -> int:
    raw = inst.rcon_client.send_command(
        "/sc local n=0 "
        "for _,e in pairs(game.surfaces[1].find_entities_filtered{name='infinity-chest'}) do "
        "local ok,filters=pcall(function() return e.infinity_container_filters end) "
        "if (not ok) or (not filters) then "
        "ok,filters=pcall(function() return e.infinity_container_filter end) "
        "end "
        "if ok and filters then "
        "if filters.name then filters={filters} end "
        "for _,f in pairs(filters) do if f and f.name then n=n+1 end end "
        "end "
        "end "
        "rcon.print(n)"
    ).strip()
    return int(float(raw or 0))


def test_learning_env_reset_restores_snapshot_core_entities(td_save_instance):
    """A second reset must reload the saved TD map, not an empty shell.

    The test intentionally damages exactly the entities that made the observed
    failure obvious. The next ``env.reset()`` should restore them from the
    snapshot captured on the first reset, then apply the usual per-episode
    turret deletion.
    """
    inst = td_save_instance
    deletion_pct = 0.5
    cfg = TDScenarioConfig(
        spawn_waves_at_runtime=False,
        clear_biters_on_reset=True,
        restore_starting_nests_on_reset=True,
        disable_enemy_expansion=True,
        turret_deletion_percentage=deletion_pct,
    )
    env = TowerDefenseEnv(instance=inst, config=cfg)

    obs1, _ = env.reset(seed=123)
    slots = int(obs1["slot_valid_mask"].sum())
    boilers = int(obs1["boiler_valid_mask"].sum())
    expected_empty = int(math.floor(slots * deletion_pct))
    expected_live_turrets = slots - expected_empty

    assert float(obs1["radar"][2]) > 0.0, "first reset did not find a live radar"
    assert slots > 0, "first reset did not read turret slots from the TD save"
    assert boilers > 0, "first reset did not read boilers from the TD save"
    assert _count(inst, "name='infinity-chest'") > 0, "TD save has no infinity chests"
    assert _infinity_filter_count(inst) > 0, (
        "TD save infinity chests have no filters to restore"
    )

    # Vandalize the live episode state. This simulates a bad episode ending and
    # specifically covers the prior broken signature: after reset, radar/turrets
    # were absent and turret deletion destroyed 0.
    _destroy(inst, "name='radar'")
    _destroy(inst, "name='gun-turret'")
    _destroy(inst, "name='boiler'")
    _destroy(inst, "name='infinity-chest'")
    assert _count(inst, "name='radar'") == 0
    assert _count(inst, "name='gun-turret'") == 0
    assert _count(inst, "name='boiler'") == 0
    assert _count(inst, "name='infinity-chest'") == 0

    obs2, _ = env.reset(seed=123)

    assert float(obs2["radar"][2]) > 0.0, "radar HP is zero after snapshot reset"
    assert int(obs2["slot_valid_mask"].sum()) == slots, (
        "turret slot geometry changed after snapshot reset"
    )
    assert int(obs2["boiler_valid_mask"].sum()) == boilers, (
        "boiler geometry changed after snapshot reset"
    )
    assert int(obs2["place_slot_mask"].sum()) == expected_empty, (
        "turret deletion did not run after snapshot restore"
    )
    assert _count(inst, "name='gun-turret'") == expected_live_turrets, (
        "live turret count after reset does not match the expected deleted subset"
    )
    assert _count(inst, "name='boiler'") == boilers, "boilers were not restored"
    assert _count(inst, "name='infinity-chest'") > 0, (
        "infinity chests were not restored"
    )
    assert _infinity_filter_count(inst) > 0, (
        "infinity-chest filters were not restored"
    )

    # One no-op should not immediately reproduce the empty-map terminal loop.
    action = {
        "action_type": 0,
        "slot_index": 0,
        "ammo_amount": 0,
        "anchor_index": 0,
    }
    obs3, _reward, terminated, _truncated, info = env.step(action)
    assert float(obs3["radar"][2]) > 0.0
    assert not terminated, f"episode terminated immediately after reset: {info}"
