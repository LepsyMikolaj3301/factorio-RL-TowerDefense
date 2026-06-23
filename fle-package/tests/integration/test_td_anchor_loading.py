"""Integration test: anchor tiles load from the predefined tower_defense save.

The map author paints the positions the agent may stand on with the
``hazard-concrete-left`` tile texture. The env scans for these tiles once on
first reset and records their centers into ``TowerDefenseEnv._anchor_slots``
(mirroring how turret slots are read). This test verifies that loading path
end to end against a live game.

Prerequisite: the loaded save (data/saves/tower_defense.zip by default) must
contain at least one ``hazard-concrete-left`` tile.

Run after:
    fle cluster start -n 1          # loads data/saves/tower_defense.zip by default
    cd fle-package
    pytest -m integration tests/integration/test_td_anchor_loading.py -v -s

Skipped automatically when no Factorio container is running.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration

ANCHOR_TILE = "hazard-concrete-left"
SCAN_RADIUS = 128


def _rcon_anchor_tiles(inst):
    """Return the set of (x, y) tile-corner coords of anchor tiles via RCON."""
    raw = inst.rcon_client.send_command(
        "/sc local out={} "
        "for _,t in pairs(game.surfaces[1].find_tiles_filtered{"
        f"name='{ANCHOR_TILE}', area={{{{{-SCAN_RADIUS},{-SCAN_RADIUS}}},"
        f"{{{SCAN_RADIUS},{SCAN_RADIUS}}}}}"
        "}) do out[#out+1]=t.position.x..','..t.position.y end "
        "rcon.print(table.concat(out, ';'))"
    ).strip()
    tiles = set()
    if raw:
        for pair in raw.split(";"):
            if not pair:
                continue
            tx, ty = pair.split(",")
            tiles.add((float(tx), float(ty)))
    return tiles


def test_anchor_tiles_present_on_map(td_save_instance):
    """The authored save must paint at least one hazard-concrete-left tile."""
    tiles = _rcon_anchor_tiles(td_save_instance)
    print(f"\n[anchor tiles] RCON found {len(tiles)} '{ANCHOR_TILE}' tile(s):")
    for (tx, ty) in sorted(tiles):
        print(f"    corner=({tx}, {ty})  center=({tx + 0.5}, {ty + 0.5})")
    assert len(tiles) > 0, (
        f"No '{ANCHOR_TILE}' tiles found on the map. The agent has no anchor "
        "positions to move to — paint anchor tiles in the authored save."
    )


def test_env_reads_anchor_slots(td_save_instance):
    """TowerDefenseEnv._read_anchor_slots loads tile centers matching the map."""
    from fle.env.gym_env.td_environment import TowerDefenseEnv

    inst = td_save_instance
    rcon_tiles = _rcon_anchor_tiles(inst)
    if not rcon_tiles:
        pytest.skip(f"No '{ANCHOR_TILE}' tiles on this map — nothing to load.")

    # Build the env and run only the (non-destructive) anchor-loading path:
    # _read_radar sets the scan center; _read_anchor_slots populates memory.
    env = TowerDefenseEnv(inst)
    env._read_radar()
    env._read_anchor_slots()

    anchors = env._anchor_slots
    assert anchors is not None, "_anchor_slots was never populated"
    assert anchors.ndim == 2 and anchors.shape[1] == 2, (
        f"Expected (N, 2) anchor array, got shape {anchors.shape}"
    )
    assert anchors.dtype == np.float32

    n = len(anchors)
    print(f"\n[anchor slots] env._anchor_slots loaded {n} anchor center(s):")
    for i, (ax, ay) in enumerate(anchors):
        print(f"    [{i}] ({ax}, {ay})")
    assert n > 0, "No anchors loaded despite tiles existing on the map"
    assert n <= env.max_anchors, f"Loaded {n} anchors, exceeds cap {env.max_anchors}"

    # Every loaded anchor is a tile center (+0.5) of a real anchor tile. When the
    # map has more tiles than the cap, only the count comparison is meaningful.
    if n == len(rcon_tiles):
        loaded = {(round(x - 0.5, 3), round(y - 0.5, 3)) for x, y in anchors}
        assert loaded == rcon_tiles, (
            "Loaded anchor centers do not match the map's anchor tiles."
        )
