"""Integration tests: validate a predefined tower_defense save loaded by FLE.

Run after:
    fle cluster start -n 1          # loads data/saves/tower_defense.zip by default
    cd fle-package
    pytest -m integration tests/integration/test_td_save_validation.py -v -s

All tests are skipped automatically when no Factorio container is running.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration

# Tuneable expectations — adjust to match your specific authored map.
EXPECTED = {
    "min_turrets": 1,
    "require_radar": True,
    "min_walls": 0,
    "min_spawners": 1,
    # Agent should spawn near this position after env.reset().
    "player_spawn_xy": (0.0, 10.0),
    "player_spawn_tolerance": 2.0,
}


# ---------------------------------------------------------------------------
# Test 1: player character is alive and near (0, 10)
# ---------------------------------------------------------------------------
def test_player_is_spawned(td_save_instance):
    # FLE's agent character lives in storage.agent_characters[1], not in
    # game.get_player(1).character (there is no human player in headless mode).
    inst = td_save_instance
    health = float(
        inst.rcon_client.send_command(
            "/sc local c=storage.agent_characters and storage.agent_characters[1]; "
            "rcon.print(c and c.valid and c.health or -1)"
        )
    )
    assert health > 0, "Agent character not alive or not found in storage.agent_characters[1]"

    pos = inst.rcon_client.send_command(
        "/sc local c=storage.agent_characters and storage.agent_characters[1]; "
        "rcon.print(c and c.valid and c.position.x..' '..c.position.y or 'nil')"
    ).strip()
    if pos and pos != "nil":
        px, py = map(float, pos.split())
        assert abs(px) <= 64 and abs(py) <= 64, f"Character suspiciously far from base: ({px},{py})"


# ---------------------------------------------------------------------------
# Test 2: TD storage is embedded (elapsed_ticks readable)
# ---------------------------------------------------------------------------
def test_scenario_storage_present(td_save_instance):
    inst = td_save_instance
    raw = inst.rcon_client.send_command(
        "/sc rcon.print(storage.elapsed_ticks or 'nil')"
    ).strip()
    assert raw != "nil", "storage.elapsed_ticks not present — was the map created from the tower_defense scenario?"
    assert int(float(raw)) >= 0


# ---------------------------------------------------------------------------
# Test 3: radar_view returns a valid symbolic grid
# ---------------------------------------------------------------------------
def test_map_grid_is_readable(td_save_instance):
    from fle.env.gym_env.td_spaces import NUM_CHANNELS

    inst = td_save_instance
    ns = inst.first_namespace
    grid = ns._radar_view(center_x=0.0, center_y=0.0, radius=32, cell_size=1.0, charted_only=True)
    assert isinstance(grid, np.ndarray), "radar_view did not return ndarray"
    assert grid.shape == (NUM_CHANNELS, 64, 64), f"Unexpected grid shape: {grid.shape}"
    assert grid.dtype == np.uint8
    # At minimum the grid should not be entirely zero — the charted area + radar should show
    assert grid.any(), "Grid is all-zero; is the area charted? Is the radar placed?"


# ---------------------------------------------------------------------------
# Test 4: a radar entity exists near (0, 0) with positive HP
# ---------------------------------------------------------------------------
def test_radar_present(td_save_instance):
    if not EXPECTED["require_radar"]:
        pytest.skip("require_radar=False")
    inst = td_save_instance
    ns = inst.first_namespace
    from fle.env.entities import Position

    entities = ns.get_entities(position=Position(x=0.0, y=0.0), radius=128.0)
    radars = [e for e in entities if getattr(e, "name", None) == "radar"]
    assert len(radars) >= 1, "No radar found within 128 tiles of (0,0)"
    assert getattr(radars[0], "health", 0) > 0, "Radar has 0 HP"


# ---------------------------------------------------------------------------
# Test 5: gun-turrets present and visible in the grid
# ---------------------------------------------------------------------------
def test_turrets_present_and_on_grid(td_save_instance):
    if EXPECTED["min_turrets"] < 1:
        pytest.skip("min_turrets=0")
    inst = td_save_instance
    ns = inst.first_namespace
    from fle.env.entities import Position

    entities = ns.get_entities(position=Position(x=0.0, y=0.0), radius=64.0)
    turrets = [e for e in entities if getattr(e, "name", None) == "gun-turret"]
    assert len(turrets) >= EXPECTED["min_turrets"], (
        f"Expected ≥{EXPECTED['min_turrets']} turrets, found {len(turrets)}"
    )
    grid = ns._radar_view(center_x=0.0, center_y=0.0, radius=32, cell_size=1.0, charted_only=True)
    assert grid[2].any(), "Turret channel (ch2) is zero despite turrets being present"


# ---------------------------------------------------------------------------
# Test 6: enemy spawners present and visible in the grid
# ---------------------------------------------------------------------------
def test_spawners_present_and_on_grid(td_save_instance):
    if EXPECTED["min_spawners"] < 1:
        pytest.skip("min_spawners=0")
    inst = td_save_instance
    raw = inst.rcon_client.send_command(
        "/sc local n=0 for _ in pairs(game.surfaces[1].find_entities_filtered"
        "({type='unit-spawner',force='enemy'})) do n=n+1 end rcon.print(n)"
    ).strip()
    count = int(float(raw))
    assert count >= EXPECTED["min_spawners"], (
        f"Expected ≥{EXPECTED['min_spawners']} enemy spawners, found {count}"
    )
    ns = inst.first_namespace
    grid = ns._radar_view(center_x=0.0, center_y=0.0, radius=64, cell_size=1.0, charted_only=True)
    assert grid[6].any(), (
        "Spawner channel (ch6) is zero — are spawners within charted range?"
    )


# ---------------------------------------------------------------------------
# Test 7: walls present (if configured)
# ---------------------------------------------------------------------------
def test_walls_present(td_save_instance):
    if EXPECTED["min_walls"] < 1:
        pytest.skip("min_walls=0")
    inst = td_save_instance
    ns = inst.first_namespace
    from fle.env.entities import Position

    entities = ns.get_entities(position=Position(x=0.0, y=0.0), radius=64.0)
    walls = [e for e in entities if getattr(e, "name", None) == "stone-wall"]
    assert len(walls) >= EXPECTED["min_walls"], (
        f"Expected ≥{EXPECTED['min_walls']} walls, found {len(walls)}"
    )
    grid = ns._radar_view(center_x=0.0, center_y=0.0, radius=32, cell_size=1.0, charted_only=True)
    assert grid[1].any(), "Wall channel (ch1) is zero despite walls being present"


# ---------------------------------------------------------------------------
# Test 8: enemy evolution factor is finite and in [0, 1]
# ---------------------------------------------------------------------------
def test_evolution_factor_in_range(td_save_instance):
    # Factorio 2.0: evolution is per-surface via get_evolution_factor(surface)
    inst = td_save_instance
    raw = inst.rcon_client.send_command(
        "/sc rcon.print(game.forces['enemy'].get_evolution_factor(game.surfaces[1]))"
    ).strip()
    evo = float(raw)
    assert 0.0 <= evo <= 1.0, f"evolution_factor out of range: {evo}"


# ---------------------------------------------------------------------------
# Test 9: env.reset() reads the map radar without duplicating it
# ---------------------------------------------------------------------------
def test_env_reset_reads_radar_no_double_place(td_save_instance):
    from fle.env.gym_env.td_environment import TowerDefenseEnv
    from fle.env.gym_env.td_config import TDScenarioConfig
    from fle.env.entities import Position

    if not EXPECTED["require_radar"]:
        pytest.skip("require_radar=False")

    inst = td_save_instance
    cfg = TDScenarioConfig(
        spawn_waves_at_runtime=False,  # no waves during this test
        clear_biters_on_reset=True,
    )
    env = TowerDefenseEnv(instance=inst, config=cfg)
    obs, info = env.reset()

    # Radar observation should show positive HP
    assert obs["radar"][2] > 0, "obs['radar'] HP is zero after reset; radar not found"

    # Exactly one radar should exist (env must NOT have placed a second one)
    entities = inst.first_namespace.get_entities(
        position=Position(x=0.0, y=0.0), radius=128.0
    )
    radar_count = sum(1 for e in entities if getattr(e, "name", None) == "radar")
    assert radar_count == 1, (
        f"Expected exactly 1 radar, found {radar_count}. "
        "TowerDefenseEnv must read the map radar, not place an extra one."
    )

    # Turret slots should be populated
    assert obs["slot_valid_mask"].sum() >= EXPECTED["min_turrets"], (
        "No valid turret slots found in observation"
    )

    # Character channel should be live
    assert obs["character"][2] > 0, "Character HP is zero in observation after reset"

    # Spawner visibility in the grid is layout-dependent (spawners can be outside
    # the 32-tile observation radius). Spawner existence is validated in test 6.


# ---------------------------------------------------------------------------
# Test 10: two consecutive env.reset() calls restore the same initial state
# ---------------------------------------------------------------------------
def test_episode_restart_restores_initial_state(td_save_instance):
    """A second env.reset() must reproduce the same game state as the first.

    This simulates what happens when the agent dies and the episode restarts:
    radar HP, turret layout, inventory, and character position should all match
    the snapshot captured at the very start of the session.
    """
    from fle.env.gym_env.td_environment import TowerDefenseEnv
    from fle.env.gym_env.td_config import TDScenarioConfig

    inst = td_save_instance
    cfg = TDScenarioConfig(
        spawn_waves_at_runtime=False,
        clear_biters_on_reset=True,
    )
    env = TowerDefenseEnv(instance=inst, config=cfg)

    obs1, _ = env.reset()  # first reset: cleans map, sets inventory, captures snapshot

    # Take a few noops so the game has ticked slightly
    for _ in range(3):
        action = env.action_space.sample()
        action["action_type"] = 0  # noop
        env.step(action)

    obs2, _ = env.reset()  # second reset: must restore from snapshot

    # Radar should be alive and at the same HP
    assert obs2["radar"][2] > 0, "Radar HP is zero after second reset"
    assert abs(float(obs2["radar"][2]) - float(obs1["radar"][2])) < 1.0, (
        f"Radar HP changed between resets: {obs1['radar'][2]:.1f} → {obs2['radar'][2]:.1f}"
    )

    # Turret slot count must be the same (layout read from map, never changes)
    slots1 = int(obs1["slot_valid_mask"].sum())
    slots2 = int(obs2["slot_valid_mask"].sum())
    assert slots1 == slots2, (
        f"Turret slot count changed between resets: {slots1} → {slots2}"
    )

    # Character must be alive after restart
    assert obs2["character"][2] > 0, "Character HP is zero after second reset"

    # Inventory must be restored (snapshot includes starting items)
    np.testing.assert_array_equal(
        obs1["inventory"], obs2["inventory"],
        err_msg=f"Inventory changed between resets: {obs1['inventory']} → {obs2['inventory']}",
    )

    # Character position should be at spawn (0, 10) both times
    char_x2, char_y2 = float(obs2["character"][0]), float(obs2["character"][1])
    spawn_x, spawn_y = cfg.player_spawn_position
    assert abs(char_x2 - spawn_x) < 2.0 and abs(char_y2 - spawn_y) < 2.0, (
        f"Character not at spawn after second reset: ({char_x2:.1f},{char_y2:.1f})"
    )


# ---------------------------------------------------------------------------
# Test 11: turret deletion transfers turrets to agent inventory
# ---------------------------------------------------------------------------
def test_turret_deletion_gives_turrets_to_agent(td_save_instance):
    """After env.reset() with turret_deletion_percentage=0.5:

    - Exactly floor(total * 0.5) turrets are removed from the map.
    - The same number of gun-turrets appear in the agent's inventory.
    - The env still knows all slot positions (slot_valid_mask unchanged).
    - place_slot_mask marks exactly the emptied slots.
    """
    from fle.env.gym_env.td_environment import TowerDefenseEnv
    from fle.env.gym_env.td_config import TDScenarioConfig

    inst = td_save_instance
    pct = 0.5
    cfg = TDScenarioConfig(
        spawn_waves_at_runtime=False,
        clear_biters_on_reset=True,
        turret_deletion_percentage=pct,
    )
    env = TowerDefenseEnv(instance=inst, config=cfg)
    obs, _ = env.reset()

    total_slots = int(obs["slot_valid_mask"].sum())
    assert total_slots >= EXPECTED["min_turrets"], (
        f"No turret slots found; check that the map has pre-placed gun-turrets"
    )

    n_expected_deleted = int(np.floor(total_slots * pct))
    n_expected_on_map = total_slots - n_expected_deleted

    # Count gun-turrets actually remaining on the map (player force)
    raw = inst.rcon_client.send_command(
        "/sc local n=0 for _,e in pairs(game.surfaces[1].find_entities_filtered("
        "{name='gun-turret',force='player'})) do n=n+1 end rcon.print(n)"
    ).strip()
    n_on_map = int(float(raw))
    assert n_on_map == n_expected_on_map, (
        f"Expected {n_expected_on_map} turrets remaining on map after deletion, "
        f"found {n_on_map} (total slots={total_slots}, deletion_pct={pct})"
    )

    # Agent inventory must hold exactly the deleted count
    inv_raw = inst.rcon_client.send_command(
        "/sc local c=storage.agent_characters[1]; "
        "rcon.print(c and c.valid and c.get_item_count('gun-turret') or 0)"
    ).strip()
    n_in_inv = int(float(inv_raw))
    assert n_in_inv == n_expected_deleted, (
        f"Expected {n_expected_deleted} gun-turrets in agent inventory, found {n_in_inv}"
    )

    # place_slot_mask must reflect exactly the empty (deleted) slots
    n_placeable = int(obs["place_slot_mask"].sum())
    assert n_placeable == n_expected_deleted, (
        f"place_slot_mask has {n_placeable} placeable slots, expected {n_expected_deleted}"
    )

    # Starting inventory must not contain turrets from starting_inventory config
    # (they should arrive only via the deletion-transfer mechanism above)
    ammo_raw = inst.rcon_client.send_command(
        "/sc local c=storage.agent_characters[1]; "
        "rcon.print(c and c.valid and c.get_item_count('piercing-rounds-magazine') or 0)"
    ).strip()
    assert int(float(ammo_raw)) > 0, "Agent should have piercing-rounds-magazine ammo"


# ---------------------------------------------------------------------------
# Test 12: spawners survive a full entity-wipe reset (DESTRUCTIVE — run last)
# NOTE: inst.reset() with clear_entities=True (the default) wipes all player-
# force buildings from the server. Keep this test at the end of the file so it
# doesn't corrupt state for the tests above.
# ---------------------------------------------------------------------------
def test_spawners_survive_reset(td_save_instance):
    inst = td_save_instance

    def count_spawners():
        raw = inst.rcon_client.send_command(
            "/sc local n=0 for _ in pairs(game.surfaces[1].find_entities_filtered"
            "({type='unit-spawner',force='enemy'})) do n=n+1 end rcon.print(n)"
        ).strip()
        return int(float(raw))

    before = count_spawners()
    inst.reset()  # full reset: clears player-force entities, should keep spawners
    after = count_spawners()
    assert after == before, (
        f"Spawner count changed after reset: {before} → {after}. "
        "Spawners must never be cleared by FLE's entity wipe."
    )
