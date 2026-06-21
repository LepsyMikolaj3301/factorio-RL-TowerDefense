"""Indefinite anchor-walk loop: visually verify A* pathfinding between anchor tiles.

Picks random anchor pairs, requests an A* path via Factorio's surface.request_path(),
prints every waypoint, then steps the character through them one teleport at a time
(with a short sleep between each) so movement is visible in-game.

How A* movement reaches Factorio
---------------------------------
Only 3 RCON calls move the character across N waypoints:

  1. _request_path(start, goal)
         Calls surface.request_path() → Factorio A* runs async.
         Returns path_handle (int).

  2. _get_path(path_handle)
         Polls storage.paths[handle] with exponential backoff until "success".
         Returns List[Position] — the N waypoints (~1 tile apart).

  3. [in move_to] storage.actions.move_to(player_index, path_handle, ...)
         Loads all N waypoints into storage.walking_queues[player_index] at once.
         Server on_tick fires every 5 ticks, sets
             player.walking_state = {walking=true, direction=<to_next_wp>}
         advancing waypoints automatically until the queue is empty.

This test skips step 3 (move_to) and instead teleports the character along the
path itself, so the A* route is transparent and fully observable.

By default the character GLIDES smoothly along the path (sub-tile teleport
interpolation) and the console stays quiet — only the move header and the arrival
line are printed. Two flags change that:

    --walk-verbose : print every A* waypoint coordinate
    --walk-step    : jump tile-by-tile instead of the smooth glide

The loop is RCON-reconnect-resilient: when a human player joins the server the
connection is briefly dropped during map sync; the test waits and reconnects
automatically rather than failing.

Tip: connect to the server BEFORE launching the test. The agent character is
(re)created at startup, which wipes all other characters — so joining first leaves
you character-less (observing) without stealing the agent character.

Run:
    fle cluster start -n 1     # loads data/saves/tower_defense.zip
    cd fle-package
    pytest -m integration tests/integration/test_td_anchor_walk.py -v -s -p no:warnings
    # add --walk-verbose to print waypoints, --walk-step for tile-by-tile motion.
    # Press Ctrl+C to stop.
"""

import math
import random
import time

import pytest
from factorio_rcon.factorio_rcon import RCONClosed

from fle.env.entities import Position

pytestmark = pytest.mark.integration

STEP_SLEEP = 0.10    # --walk-step: seconds between whole-tile teleports
SUB_TILE = 0.25      # smooth glide: teleport every quarter-tile
GLIDE_SLEEP = 0.03   # smooth glide: seconds between sub-tile teleports
ARRIVE_SLEEP = 0.5   # pause at destination before picking the next anchor


def _get_char_pos(inst):
    """Read the agent character's current position from the game."""
    raw = inst.rcon_client.send_command(
        "/sc local c=storage.agent_characters and storage.agent_characters[1]; "
        "if c and c.valid then rcon.print(c.position.x..','..c.position.y) "
        "else rcon.print('0,0') end"
    ).strip()
    x, y = raw.split(",")
    return float(x), float(y)


def _teleport(inst, x, y):
    """Teleport the agent character to (x, y)."""
    inst.rcon_client.send_command(
        f"/sc storage.agent_characters[1].teleport({{{x},{y}}})"
    )


def _glide(inst, frm: Position, to: Position):
    """Smoothly teleport the character from `frm` to `to` in sub-tile increments.

    Interpolating at SUB_TILE-tile resolution makes the motion read as a continuous
    glide in-game rather than a one-tile-per-frame hop.
    """
    dist = math.hypot(to.x - frm.x, to.y - frm.y)
    n = max(1, int(math.ceil(dist / SUB_TILE)))
    for k in range(1, n + 1):
        t = k / n
        _teleport(inst, frm.x + (to.x - frm.x) * t, frm.y + (to.y - frm.y) * t)
        time.sleep(GLIDE_SLEEP)


def test_walk_anchors_loop(td_save_instance, request):
    """Indefinitely walk between random anchor pairs along their A* paths."""
    inst = td_save_instance
    ns = inst.first_namespace

    verbose = request.config.getoption("--walk-verbose")
    step_mode = request.config.getoption("--walk-step")

    # Read the scenario's canonical slot data the same way the env does at
    # startup — one read per session, reused every episode.
    from fle.env.gym_env.td_environment import TowerDefenseEnv
    env = TowerDefenseEnv(inst)
    env._read_radar()
    env._read_turret_slots()
    env._read_anchor_slots()
    env._build_anchor_reach_matrix()

    anchors = env._anchor_slots  # numpy (N, 2), each row is [x, y]
    if anchors is None or len(anchors) < 2:
        pytest.skip(
            "Need at least 2 'hazard-concrete-left' anchor tiles on the map; "
            f"found {0 if anchors is None else len(anchors)}."
        )

    print(f"\nLoaded {len(anchors)} anchors:")
    for i, (ax, ay) in enumerate(anchors):
        print(f"  [{i:2d}]  ({ax:7.2f}, {ay:7.2f})")

    slots = env._turret_slots   # numpy (M, 2) or None
    n_slots = 0 if slots is None else len(slots)
    print(f"\nLoaded {n_slots} turret slot(s) from TD scenario startup:")
    if slots is not None:
        for i, (sx, sy) in enumerate(slots):
            print(f"  [{i:2d}]  ({sx:7.2f}, {sy:7.2f})")

    # Teleport character to a random starting anchor
    start_idx = random.randrange(len(anchors))
    sx, sy = float(anchors[start_idx, 0]), float(anchors[start_idx, 1])
    _teleport(inst, sx, sy)
    ns.player_location = Position(x=sx, y=sy)
    print(f"\nStarting at anchor[{start_idx}] ({sx}, {sy})")
    mode = "tile-step" if step_mode else "smooth glide"
    print(f"Movement: {mode}. {'Verbose waypoints ON.' if verbose else 'Quiet (use --walk-verbose for waypoints).'}")
    print("Connect to Factorio BEFORE the test to observe. Press Ctrl+C to stop.\n")

    move_count = 0
    visited = set()
    visited.add(start_idx)
    prev_idx = start_idx
    curr_pos = Position(x=sx, y=sy)

    try:
        while True:
            # Pick a different random anchor as target
            choices = [i for i in range(len(anchors)) if i != prev_idx]
            tgt_idx = random.choice(choices)
            tx, ty = float(anchors[tgt_idx, 0]), float(anchors[tgt_idx, 1])

            print(f"→ Move {move_count + 1}: "
                  f"anchor[{prev_idx}] ({curr_pos.x:.1f},{curr_pos.y:.1f})  →  "
                  f"anchor[{tgt_idx}] ({tx:.1f},{ty:.1f})")

            try:
                # Use Python-tracked position (we control every teleport, so
                # re-reading from RCON is unreliable if the character got moved
                # by an unexpected event on the server side).
                ns.player_location = curr_pos

                # Request A* path
                waypoints = []
                try:
                    path_handle = ns._request_path(
                        curr_pos,
                        Position(x=tx, y=ty),
                        allow_paths_through_own_entities=True,
                        resolution=-1,
                    )
                    waypoints = ns._get_path(path_handle)
                    if verbose:
                        print(f"   {len(waypoints)} waypoints:")
                        for i, wp in enumerate(waypoints):
                            marker = (
                                " ← start" if i == 0
                                else (" ← target" if i == len(waypoints) - 1 else "")
                            )
                            print(f"   [{i:3d}]  ({wp.x:7.2f}, {wp.y:7.2f}){marker}")
                except Exception as e:
                    print(f"   [A* failed: {e}] — direct glide fallback")
                    waypoints = [Position(x=tx, y=ty)]

                # Move along the path. Default: smooth sub-tile glide between
                # consecutive waypoints. --walk-step: discrete tile-by-tile hops.
                t0 = time.time()
                pos = curr_pos
                for wp in waypoints:
                    if step_mode:
                        _teleport(inst, wp.x, wp.y)
                        time.sleep(STEP_SLEEP)
                    else:
                        _glide(inst, pos, wp)
                    pos = wp
                elapsed = time.time() - t0

                # Track state
                curr_pos = Position(x=tx, y=ty)
                ns.player_location = curr_pos
                move_count += 1
                visited.add(tgt_idx)
                prev_idx = tgt_idx

                # Arrival report — show which turret slots are reachable from here,
                # using the reach matrix precomputed at TD scenario startup.
                line = f"   arrived at anchor[{tgt_idx}] ({tx:.1f},{ty:.1f}) in {elapsed:.1f}s"
                if env._anchor_reach_matrix is not None and n_slots > 0:
                    row = env._anchor_reach_matrix[tgt_idx]
                    reachable_slots = [j for j in range(n_slots) if row[j]]
                    line += (f"  | reach: {len(reachable_slots)}/{n_slots} slot(s) "
                             f"{reachable_slots if len(reachable_slots) <= 4 else reachable_slots[:4] + ['...']}")
                print(line)

                time.sleep(ARRIVE_SLEEP)

            except RCONClosed:
                print("\n   [RCON dropped — player joined? reconnecting...]")
                try:
                    inst.reconnect_rcon(pause_after=False)
                    print("   [reconnected — resuming]")
                except RuntimeError as re:
                    print(f"   [reconnect failed: {re}]")
                    time.sleep(5.0)
                # Don't advance prev_idx — retry the same move
                continue

    except KeyboardInterrupt:
        print(f"\n--- Stopped after {move_count} moves, "
              f"{len(visited)}/{len(anchors)} anchors visited ---")
