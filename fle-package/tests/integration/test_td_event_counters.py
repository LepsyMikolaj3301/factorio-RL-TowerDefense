"""Integration tests for runtime TD loss/kill event counters."""

import pytest

pytestmark = pytest.mark.integration


def _radar_pos(inst):
    raw = inst.rcon_client.send_command(
        "/sc local e=game.surfaces[1].find_entities_filtered{name='radar'}[1]; "
        "if e then rcon.print(e.position.x..','..e.position.y) "
        "else rcon.print('0,0') end"
    ).strip()
    x, y = raw.split(",")
    return float(x), float(y)


def _create_and_die(inst, name, x, y):
    raw = inst.rcon_client.send_command(
        f"/sc local s=game.surfaces[1] "
        f"local p=s.find_non_colliding_position('{name}',{{{x},{y}}},24,1) "
        f"if not p then rcon.print('nil') return end "
        f"local e=s.create_entity{{name='{name}',position=p,force='player'}} "
        f"if not e then rcon.print('nil') return end "
        f"e.die(game.forces.enemy) "
        f"rcon.print('ok')"
    ).strip()
    assert raw == "ok", f"Could not create and kill {name} near ({x},{y}): {raw!r}"


def test_td_event_counters_include_directional_losses(td_save_instance):
    inst = td_save_instance
    ns = inst.first_namespace
    cx, cy = _radar_pos(inst)

    # Flush any setup/noise from the current decision window.
    ns._read_td_events()

    offsets = {
        "n": (0, -96),
        "e": (96, 0),
        "s": (0, 96),
        "w": (-96, 0),
    }
    for _dir, (dx, dy) in offsets.items():
        _create_and_die(inst, "stone-wall", cx + dx, cy + dy)
    for _dir, (dx, dy) in offsets.items():
        _create_and_die(inst, "gun-turret", cx + dx, cy + dy)

    ev = ns._read_td_events()

    assert ev["walls_lost"] == 4
    assert ev["turrets_lost"] == 4
    for direction in ("n", "e", "s", "w"):
        assert ev[f"wall_{direction}"] == 1
        assert ev[f"turret_{direction}"] == 1
