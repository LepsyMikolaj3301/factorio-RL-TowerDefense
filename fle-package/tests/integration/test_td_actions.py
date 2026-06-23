"""Action-level integration tests for the Tower Defense primitives.

These tests exercise the underlying namespace tools directly (NOT the gym
``TowerDefenseEnv.step()`` path) so we can verify each primitive the RL model
needs, in isolation, against a live Factorio server:

1. A* move to a position           -> ``move_to``
2. Take a turret from A, place at B -> ``place_entity`` + ``pickup_entity``
3. Supply / resupply a turret       -> ``insert_item``
4. Read the current map             -> ``_radar_view`` + ``_spawn_enemy``

Each test runs against its own clean instance (the ``clean_namespace`` fixture
in conftest.py) and logs every step so the run reads as a clear narrative.

Start a container first, then run with live logs::

    fle cluster start --num-instances 1 --scenario tower_defense
    pytest tests/integration/test_td_actions.py -m integration --log-cli-level=INFO -s -v

All tests skip automatically when no Factorio container is running.
"""

import logging

import pytest

from fle.env.entities import Direction, Position
from fle.env.game_types import Prototype
from fle.env.tools.admin.radar_view.client import (
    CHANNEL_BITER,
    CHANNEL_TURRET,
    CHANNEL_WALL,
)

log = logging.getLogger("td_actions")

pytestmark = pytest.mark.integration


def _dist(a: Position, b: Position) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _gun_turret_at(ns, position: Position, radius: float = 0.6):
    """Return the first gun-turret entity within ``radius`` of ``position``, or None."""
    entities = ns.get_entities(position=position, radius=radius)
    for e in entities:
        if e.name == "gun-turret":
            return e
    return None


def _turret_ammo(turret) -> int:
    """Total ammo magazines loaded in a gun-turret (read from its turret_ammo inventory)."""
    inv = getattr(turret, "turret_ammo", None)
    if inv is None:
        return 0
    return sum(int(v) for _, v in inv.items())


class TestTowerDefenseActions:
    """Verify each Tower Defense primitive action against a live server."""

    def test_move_to_reaches_target(self, clean_namespace):
        """move_to (A*) walks the character to the requested position."""
        ns = clean_namespace
        start = Position(x=ns.player_location.x, y=ns.player_location.y)
        target = Position(x=10.0, y=0.0)
        log.info("move_to: start=%s target=%s", start, target)

        result = ns.move_to(target)
        log.info("move_to: result=%s player_location=%s", result, ns.player_location)

        # Reached the target (move_to quantises to 1/4 tile, allow ~1 tile slack).
        assert abs(result.x - target.x) <= 1.0, f"x off target: {result}"
        assert abs(result.y - target.y) <= 1.0, f"y off target: {result}"
        # Actually moved from the spawn.
        assert _dist(start, result) > 1.0, "character did not move from spawn"
        # Namespace tracking agrees with the returned position.
        assert _dist(ns.player_location, result) <= 0.5

    def test_take_turret_and_place_neighbour(self, clean_namespace):
        """Place a turret at A, pick it up, then place it at a neighbouring B."""
        ns = clean_namespace
        a = Position(x=2.0, y=0.0)

        n_start = ns.inspect_inventory().get(Prototype.GunTurret)
        log.info("take/place: gun-turret inventory at start=%s", n_start)

        # Place at A.
        ns.place_entity(Prototype.GunTurret, Direction.UP, a)
        assert _gun_turret_at(ns, a) is not None, "turret not present at A after place"
        n_after_place = ns.inspect_inventory().get(Prototype.GunTurret)
        log.info("take/place: placed at A=%s, inventory=%s", a, n_after_place)
        assert n_after_place == n_start - 1, "inventory did not decrease on place"

        # Pick it back up.
        picked = ns.pickup_entity(Prototype.GunTurret, position=a)
        assert picked is True, "pickup_entity did not report success"
        assert _gun_turret_at(ns, a) is None, "turret still present at A after pickup"
        n_after_pickup = ns.inspect_inventory().get(Prototype.GunTurret)
        log.info("take/place: picked up from A, inventory=%s", n_after_pickup)
        assert n_after_pickup == n_start, "inventory not restored after pickup"

        # Find a valid neighbouring cell (gun-turret is 2x2 -> centres >= 2 apart).
        candidates = [
            Position(x=4.0, y=0.0),
            Position(x=2.0, y=2.0),
            Position(x=0.0, y=0.0),
            Position(x=2.0, y=-2.0),
            Position(x=5.0, y=0.0),
        ]
        b = next(
            (
                c
                for c in candidates
                if ns.can_place_entity(Prototype.GunTurret, Direction.UP, c)
            ),
            candidates[0],
        )
        log.info("take/place: chosen neighbour B=%s", b)

        # Place at B.
        ns.place_entity(Prototype.GunTurret, Direction.UP, b)
        assert _gun_turret_at(ns, b) is not None, "turret not present at B after place"
        assert _gun_turret_at(ns, a) is None, "turret reappeared at A"
        log.info("take/place: turret successfully relocated A=%s -> B=%s", a, b)

    def test_resupply_turret(self, clean_namespace):
        """insert_item loads ammo into an existing turret, repeatedly."""
        ns = clean_namespace
        a = Position(x=2.0, y=0.0)
        ns.place_entity(Prototype.GunTurret, Direction.UP, a)

        turret = _gun_turret_at(ns, a)
        assert turret is not None
        ammo_before = _turret_ammo(turret)
        mags_before = ns.inspect_inventory().get(Prototype.FirearmMagazine)
        log.info("resupply: ammo_before=%s magazines_in_inv=%s", ammo_before, mags_before)

        # First resupply.
        ns.insert_item(Prototype.FirearmMagazine, turret, quantity=10)
        turret = _gun_turret_at(ns, a)
        ammo_after = _turret_ammo(turret)
        mags_after = ns.inspect_inventory().get(Prototype.FirearmMagazine)
        log.info("resupply: ammo_after=%s magazines_in_inv=%s", ammo_after, mags_after)

        assert ammo_after > ammo_before, "turret ammo did not increase after insert"
        assert mags_after < mags_before, "magazines not consumed from inventory"

        # Second resupply increases further (until capacity).
        ns.insert_item(Prototype.FirearmMagazine, turret, quantity=10)
        turret = _gun_turret_at(ns, a)
        ammo_after_2 = _turret_ammo(turret)
        log.info("resupply: ammo_after_second_insert=%s", ammo_after_2)
        assert ammo_after_2 >= ammo_after, "second resupply lowered ammo"

    def test_read_map_sees_turret_and_biters(self, clean_namespace):
        """_radar_view renders placed turrets and spawned biters on the grid."""
        ns = clean_namespace
        # Place off-origin so the turret does not collide with the spawn character.
        turret_pos = Position(x=2.0, y=0.0)
        ns.place_entity(Prototype.GunTurret, Direction.UP, turret_pos)
        ns.place_entity(Prototype.StoneWall, Direction.UP, Position(x=6.0, y=0.0))

        spawned = ns._spawn_enemy("small-biter", x=20.0, y=0.0, count=5)
        log.info("read_map: spawned biters=%s", spawned)
        assert int(spawned) >= 1, f"_spawn_enemy reported no spawns: {spawned!r}"

        radius, cell_size = 32, 1.0
        grid = ns._radar_view(
            center_x=0.0,
            center_y=0.0,
            radius=radius,
            cell_size=cell_size,
            charted_only=False,
        )

        assert grid.shape == (8, 64, 64), f"unexpected grid shape {grid.shape}"
        assert grid.dtype.name == "uint8"

        n_turret_cells = int(grid[CHANNEL_TURRET].sum() > 0)
        n_biter_cells = int((grid[CHANNEL_BITER] > 0).sum())
        n_wall_cells = int((grid[CHANNEL_WALL] > 0).sum())
        log.info(
            "read_map: nonzero cells turret=%s biter=%s wall=%s",
            int((grid[CHANNEL_TURRET] > 0).sum()),
            n_biter_cells,
            n_wall_cells,
        )

        assert (grid[CHANNEL_TURRET] > 0).any(), "turret not visible on radar grid"
        assert (grid[CHANNEL_BITER] > 0).any(), "biters not visible on radar grid"

        # The turret should render near its world cell.
        def to_idx(world: float) -> int:
            return int(round((world - (0.0 - radius)) / cell_size))

        ty, tx = to_idx(turret_pos.y), to_idx(turret_pos.x)
        window = grid[CHANNEL_TURRET][max(0, ty - 1):ty + 2, max(0, tx - 1):tx + 2]
        assert window.any(), "turret not rendered near its expected grid cell"

        # Cross-check the turret against get_entities so a disagreement fails loudly.
        # (Note: get_entities only returns player/neutral entities, not enemy units,
        # so biter presence is verified via the radar grid and the spawn count above.)
        names = [e.name for e in ns.get_entities(position=turret_pos, radius=40.0)]
        log.info("read_map: get_entities names=%s", names)
        assert "gun-turret" in names, "get_entities does not see the turret"
