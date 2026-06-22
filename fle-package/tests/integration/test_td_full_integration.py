"""Full end-to-end TD pre-training integration test.

This is the capstone integration test: it stitches together every piece the RL
model depends on and asserts they agree, against a live Factorio container loaded
from the predefined tower_defense save. Run it before training a real model.

What it exercises, in order:

  1. STARTUP — TowerDefenseEnv.reset() reads the radar, turret slots and anchor
     tiles from the authored map, builds the (Euclidean) anchor→turret reach
     matrix, and deletes a subset of turrets (moving them into the agent's
     inventory) so there are empty slots to (re)fill.
  2. REACH MASK — the per-anchor turret reachability matrix the policy is masked
     with. Reachability is *proximity* (slot within REACH_DISTANCE of an anchor),
     NOT pathfinding. Built once at startup. We assert every turret slot is
     serviceable from at least one anchor (no orphan turrets).
  3. NESTS — read & report enemy unit-spawner (biter nest) positions. Never spawn
     a wave (runtime wave spawning is disabled in the env).
  4. INVENTORY — confirm the agent received ammo and the deleted turrets.
  5. WALK + SERVICE — A* is used ONLY here, to move the character between anchors
     (request_path -> get_path -> glide). At each anchor, for every turret
     reachable FROM THAT ANCHOR, act at the TURRET's coordinates (never the
     anchor): empty slot + turret in inventory -> place then ammo; occupied slot
     -> ammo. Ammo is given on every visit.
  6. FINAL — assert full coverage (all anchors visited, every reachable empty slot
     filled, every reachable turret got ammo) and that inventory accounting holds.

Run:
    fle cluster start -n 1     # loads data/saves/tower_defense.zip
    cd fle-package
    pytest -m integration tests/integration/test_td_full_integration.py -v -s -p no:warnings

Connect a Factorio client BEFORE launching to watch the character glide between
every anchor and service its reachable turrets. Skipped automatically when no
container is running (via the td_save_instance fixture).
"""

import math
import time

import pytest

from fle.env.entities import Direction, Position
from fle.env.game_types import Prototype
from fle.env.gym_env.td_config import TDScenarioConfig
from fle.env.gym_env.td_environment import TowerDefenseEnv
from fle.env.gym_env.td_spaces import REACH_DISTANCE, TRACKED_ITEMS

pytestmark = pytest.mark.integration

# Movement tuning (mirrors test_td_anchor_walk.py).
SUB_TILE = 0.25       # smooth glide: teleport every quarter-tile
GLIDE_SLEEP = 0.02    # seconds between sub-tile teleports
ARRIVE_SLEEP = 0.25   # pause on each anchor before servicing

# Ammo inserted into a turret per service action.
AMMO_PER = 5
# Turret deletion fraction for this run (leaves empty slots to (re)fill).
DELETION_PCT = 0.5


# ---------------------------------------------------------------------------
# Low-level movement helpers (copied from test_td_anchor_walk.py — small and
# self-contained so this test stands alone).
# ---------------------------------------------------------------------------
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
    """Smoothly teleport the character from `frm` to `to` in sub-tile steps."""
    dist = math.hypot(to.x - frm.x, to.y - frm.y)
    n = max(1, int(math.ceil(dist / SUB_TILE)))
    for k in range(1, n + 1):
        t = k / n
        _teleport(inst, frm.x + (to.x - frm.x) * t, frm.y + (to.y - frm.y) * t)
        time.sleep(GLIDE_SLEEP)


def _scan_nests(inst):
    """Return (x, y) positions of enemy unit-spawners (biter nests) via RCON."""
    raw = inst.rcon_client.send_command(
        "/sc local out={} "
        "for _,e in pairs(game.surfaces[1].find_entities_filtered"
        "{type='unit-spawner', force='enemy'}) do "
        "out[#out+1]=e.position.x..','..e.position.y end "
        "rcon.print(table.concat(out, ';'))"
    ).strip()
    nests = []
    if raw:
        for pair in raw.split(";"):
            if not pair:
                continue
            nx, ny = pair.split(",")
            nests.append((float(nx), float(ny)))
    return nests


def _scan_turrets(inst):
    """Return the set of (x, y) positions of live gun-turrets via RCON (ground truth)."""
    raw = inst.rcon_client.send_command(
        "/sc local out={} "
        "for _,e in pairs(game.surfaces[1].find_entities_filtered{name='gun-turret'}) do "
        "out[#out+1]=string.format('%.3f,%.3f', e.position.x, e.position.y) end "
        "rcon.print(table.concat(out, ';'))"
    ).strip()
    pos = set()
    if raw:
        for pair in raw.split(";"):
            if not pair:
                continue
            tx, ty = pair.split(",")
            pos.add((round(float(tx), 1), round(float(ty), 1)))
    return pos


def _set_nests_active(inst, active: bool):
    """Toggle every enemy unit-spawner's `active` flag.

    Disabling the nests stops them spawning biters so a long, unpaused walk can
    run without biters destroying the turrets it is validating. Live biters are
    not affected — clear them separately.
    """
    flag = "true" if active else "false"
    inst.rcon_client.send_command(
        "/sc for _,e in pairs(game.surfaces[1].find_entities_filtered"
        f"{{type='unit-spawner', force='enemy'}}) do e.active={flag} end"
    )


def _spawn_enemy(inst, name, x, y):
    """Create an enemy-force entity at a non-colliding spot near (x, y)."""
    inst.rcon_client.send_command(
        f"/sc local s=game.surfaces[1] "
        f"local p=s.find_non_colliding_position('{name}',{{{x},{y}}},32,1) "
        f"if p then s.create_entity{{name='{name}',position=p,force='enemy'}} end"
    )


def _count_live_biters(inst):
    """Return the number of live enemy units (biters/spitters) on the map."""
    raw = inst.rcon_client.send_command(
        "/sc local n=0 for _ in pairs(game.surfaces[1].find_entities_filtered"
        "{type='unit', force='enemy'}) do n=n+1 end rcon.print(n)"
    ).strip()
    return int(float(raw))


def _inv_count(ns, item: str) -> int:
    """Current count of `item` in the agent's inventory."""
    try:
        return int(ns.inspect_inventory().get(item, 0))
    except Exception:
        return 0


def _astar_walk(ns, inst, frm: Position, to: Position):
    """Move the character from `frm` to `to` along an A* path; return arrival pos.

    A* is used ONLY for movement between anchors. On any failure, fall back to a
    direct glide so the test still makes progress.
    """
    ns.player_location = frm
    waypoints = []
    try:
        handle = ns._request_path(
            frm, to, allow_paths_through_own_entities=True, resolution=-1
        )
        waypoints = ns._get_path(handle)
    except Exception as e:
        print(f"    [A* failed: {e}] — direct glide fallback")
        waypoints = [to]
    if not waypoints:
        waypoints = [to]

    pos = frm
    for wp in waypoints:
        _glide(inst, pos, wp)
        pos = wp
    arrived = Position(x=to.x, y=to.y)
    _teleport(inst, arrived.x, arrived.y)  # snap exactly onto the anchor
    ns.player_location = arrived
    return arrived


# ---------------------------------------------------------------------------
# The full integration test
# ---------------------------------------------------------------------------
def test_td_full_integration(td_save_instance):
    inst = td_save_instance
    ns = inst.first_namespace

    # -- 1. STARTUP -----------------------------------------------------------
    # Give the agent firearm-magazine ammo so the env's REFILL path
    # (Prototype.FirearmMagazine) has something to insert. Turrets arrive only
    # via the deletion-transfer mechanism, never from starting_inventory.
    cfg = TDScenarioConfig(
        spawn_waves_at_runtime=False,
        clear_biters_on_reset=True,
        turret_deletion_percentage=DELETION_PCT,
        starting_inventory={"firearm-magazine": 1000},
    )
    env = TowerDefenseEnv(instance=inst, config=cfg)
    obs, info = env.reset()

    # Radar must be found and alive (it is the grid center + a terminal condition).
    assert obs["radar"][2] > 0, "Radar not found / has 0 HP after reset"

    anchors = env._anchor_slots
    slots = env._turret_slots
    assert anchors is not None and len(anchors) >= 2, (
        f"Need >=2 anchor tiles, found {0 if anchors is None else len(anchors)}"
    )
    assert slots is not None and len(slots) >= 1, (
        f"Need >=1 turret slot, found {0 if slots is None else len(slots)}"
    )
    n_anchors = len(anchors)
    n_slots = len(slots)

    # Deletion moved turrets into inventory; empty slots == floor(n_slots * pct).
    n_deleted = int(math.floor(n_slots * DELETION_PCT))
    n_empty = int(obs["place_slot_mask"][:n_slots].sum())
    assert n_empty == n_deleted, (
        f"place_slot_mask has {n_empty} empty slots, expected {n_deleted}"
    )
    start_turrets = _inv_count(ns, "gun-turret")
    assert start_turrets == n_deleted, (
        f"Agent inventory has {start_turrets} gun-turrets, expected {n_deleted} "
        "(deleted turrets should transfer to inventory)"
    )

    print(f"\n[startup] {n_anchors} anchors, {n_slots} turret slots, "
          f"{n_deleted} deleted -> {start_turrets} turrets in inventory")

    # -- 2. REACH MASK (Euclidean, proximity-based, built once at startup) -----
    reach = env._anchor_reach_matrix  # (MAX_ANCHORS, MAX_SLOTS) int8
    assert reach is not None, "env._anchor_reach_matrix was never built"
    assert reach[:n_anchors, :n_slots].sum() > 0, (
        "No (anchor, slot) pair is reachable — anchors are not near any turret"
    )

    print("[reach mask] per-anchor reachable turret slots "
          f"(proximity <= {REACH_DISTANCE} tiles):")
    for i in range(n_anchors):
        ax, ay = float(anchors[i, 0]), float(anchors[i, 1])
        reachable_i = [j for j in range(n_slots) if reach[i, j]]
        print(f"    anchor[{i:2d}] ({ax:7.2f},{ay:7.2f}) -> "
              f"{len(reachable_i)} slot(s): {reachable_i}")

    # Every turret slot must be serviceable from at least one anchor; a turret no
    # anchor can reach is dead weight (a map-authoring bug) — surface it now.
    orphan_slots = [
        j for j in range(n_slots)
        if not any(reach[i, j] for i in range(n_anchors))
    ]
    assert not orphan_slots, (
        f"Turret slots unreachable from EVERY anchor: {orphan_slots}. "
        "Add an anchor near them or move them within reach."
    )

    # -- 3. NESTS (read & report only; never spawn) ---------------------------
    nests = _scan_nests(inst)
    print(f"[nests] {len(nests)} biter nest(s): {nests}")
    assert len(nests) >= 1, "No enemy unit-spawners (biter nests) found on the map"
    print(f"[biters] {_count_live_biters(inst)} live enemy unit(s) at startup")

    # -- 4. INVENTORY ---------------------------------------------------------
    ammo_idx = TRACKED_ITEMS.index("firearm-magazine")
    start_ammo = int(obs["inventory"][ammo_idx])
    assert start_ammo > 0, "Agent has no firearm-magazine ammo to dispense"
    print(f"[inventory] ammo(firearm-magazine)={start_ammo}, "
          f"gun-turret={start_turrets}")

    # -- 4b. STARTUP MEMORY VALIDATION (paused, vs RCON ground truth) ----------
    # Cross-check every value the env derived once at reset against what RCON
    # actually sees on the surface, while the game is still PAUSED so nothing
    # (biters, decay) can perturb it. This is the authoritative check that the
    # turret slots, anchor positions, occupancy masks and reach matrix are
    # internally consistent with the live game.
    live_turrets = _scan_turrets(inst)  # ground-truth gun-turret positions
    slot_pos = [(round(float(slots[j, 0]), 1), round(float(slots[j, 1]), 1))
                for j in range(n_slots)]

    # (a) Occupancy mask must agree with RCON exactly: slots the mask marks
    #     occupied are precisely the live turrets; empty slots have no turret.
    occ_from_mask = {slot_pos[j] for j in range(n_slots)
                     if not obs["place_slot_mask"][j]}
    empty_from_mask = {slot_pos[j] for j in range(n_slots)
                       if obs["place_slot_mask"][j]}
    assert occ_from_mask == live_turrets, (
        "place_slot_mask occupancy disagrees with RCON.\n"
        f"  mask-occupied not live: {sorted(occ_from_mask - live_turrets)}\n"
        f"  live not mask-occupied: {sorted(live_turrets - occ_from_mask)}"
    )
    assert empty_from_mask.isdisjoint(live_turrets), (
        f"Slots marked empty still have a live turret: "
        f"{sorted(empty_from_mask & live_turrets)}"
    )
    assert len(live_turrets) == n_slots - n_deleted, (
        f"Live turret count {len(live_turrets)} != slots-deleted "
        f"{n_slots - n_deleted}"
    )

    # (b) Anchor positions must equal the map's hazard-concrete-left tile centers.
    raw_tiles = inst.rcon_client.send_command(
        "/sc local o={} for _,t in pairs(game.surfaces[1].find_tiles_filtered"
        "{name='hazard-concrete-left'}) do "
        "o[#o+1]=(t.position.x+0.5)..','..(t.position.y+0.5) end "
        "rcon.print(table.concat(o,';'))"
    ).strip()
    rcon_anchors = set()
    for pair in (raw_tiles.split(";") if raw_tiles else []):
        if pair:
            ax_, ay_ = pair.split(",")
            rcon_anchors.add((round(float(ax_), 1), round(float(ay_), 1)))
    env_anchors = {(round(float(anchors[i, 0]), 1), round(float(anchors[i, 1]), 1))
                   for i in range(n_anchors)}
    # Every env anchor must be a real tile (no phantoms); no centers lost to dedup.
    assert env_anchors.issubset(rcon_anchors), (
        "Env anchors not present on the map as hazard-concrete-left tiles: "
        f"{sorted(env_anchors - rcon_anchors)}"
    )
    assert len(env_anchors) == n_anchors, (
        f"Anchor positions collapsed by rounding/dedup: {n_anchors} -> "
        f"{len(env_anchors)} unique"
    )
    # If the map has no more tiles than the anchor cap, the env must have read them all.
    if len(rcon_anchors) <= env.max_anchors:
        assert env_anchors == rcon_anchors, (
            f"Env missed anchor tiles: {sorted(rcon_anchors - env_anchors)}"
        )

    # (c) Reach matrix must match Euclidean proximity for every (anchor, slot).
    reach_mismatch = []
    for i in range(n_anchors):
        ax, ay = float(anchors[i, 0]), float(anchors[i, 1])
        for j in range(n_slots):
            sx, sy = float(slots[j, 0]), float(slots[j, 1])
            expected = 1 if math.hypot(ax - sx, ay - sy) <= REACH_DISTANCE else 0
            if int(reach[i, j]) != expected:
                reach_mismatch.append((i, j))
    assert not reach_mismatch, (
        f"reach matrix disagrees with Euclidean distance at {reach_mismatch[:8]}"
    )

    # (d) Inventory accounting: deleted turrets transferred to the agent.
    assert start_turrets == n_deleted == n_slots - len(live_turrets), (
        f"Inventory turret accounting off: inv={start_turrets}, "
        f"deleted={n_deleted}, missing_from_map={n_slots - len(live_turrets)}"
    )
    print(f"[startup-validate] OK — masks/turrets/anchors/reach all agree with RCON "
          f"({len(live_turrets)} live turrets, {len(rcon_anchors)} anchors)")

    # Isolate the walk from biter interference: live biters destroy turrets mid-
    # walk and desync the (once-computed) place_slot_mask from the board. Disable
    # the nests and clear any biters so the walk validates the place/service
    # plumbing deterministically. (Reset-time biter/nest handling is covered in
    # section 6b/7.) Re-enabled implicitly by env.reset() recreating the nests.
    _set_nests_active(inst, False)
    env._clear_live_biters()

    # Unpause so Factorio can process A* path requests (reset() leaves it paused)
    # and the glide is observable. Movement teleports work either way.
    inst.set_speed(10.0)
    inst.unpause()

    # -- 5. WALK + SERVICE (single pass over every anchor) --------------------
    placed: set = set()       # slot indices we placed a turret into
    serviced: set = set()     # slot indices that received ammo at least once
    visited: list = []        # anchor indices visited, in order
    turrets_left = start_turrets
    refill_events = 0
    ammo_inserted = 0

    cx, cy = _get_char_pos(inst)
    curr = Position(x=cx, y=cy)

    for i in range(n_anchors):
        ax, ay = float(anchors[i, 0]), float(anchors[i, 1])
        target = Position(x=ax, y=ay)
        print(f"-> anchor[{i}] ({ax:.1f},{ay:.1f})")
        curr = _astar_walk(ns, inst, curr, target)
        env._current_anchor_index = i  # keep env state coherent for reach_slot_mask
        visited.append(i)
        time.sleep(ARRIVE_SLEEP)

        reachable_i = [j for j in range(n_slots) if reach[i, j]]
        for j in reachable_i:
            sx, sy = float(slots[j, 0]), float(slots[j, 1])  # TURRET coord, not anchor
            turret = env._find_turret_at(sx, sy)

            # Empty slot: place a turret here if we still have one in inventory.
            if turret is None and turrets_left > 0 and j not in placed:
                try:
                    ns.place_entity(
                        Prototype.GunTurret, Direction.UP, Position(x=sx, y=sy)
                    )
                    placed.add(j)
                    turrets_left -= 1
                    turret = env._find_turret_at(sx, sy)
                    print(f"    placed turret in slot[{j}] ({sx:.1f},{sy:.1f})")
                except Exception as e:
                    print(f"    [place slot[{j}] failed: {e}]")

            # Give ammo to whatever turret now occupies the slot.
            if turret is not None:
                try:
                    ns.insert_item(Prototype.FirearmMagazine, turret, AMMO_PER)
                    serviced.add(j)
                    refill_events += 1
                    ammo_inserted += AMMO_PER
                except Exception as e:
                    print(f"    [refill slot[{j}] failed: {e}]")

    # -- 6. FINAL ASSERTIONS + SUMMARY ----------------------------------------
    # Every anchor visited exactly once.
    assert len(visited) == n_anchors, (
        f"Visited {len(visited)} anchors, expected {n_anchors}"
    )

    # Every initially-empty slot was filled (all are reachable: orphans asserted
    # out above, and inventory held exactly n_deleted turrets for n_deleted slots).
    empty_slots = {j for j in range(n_slots) if obs["place_slot_mask"][j]}
    assert placed == empty_slots, (
        f"Empty slots not fully (re)placed. placed={sorted(placed)}, "
        f"expected={sorted(empty_slots)}"
    )

    # Every turret reachable from some anchor received ammo at least once.
    all_reachable = {
        j for j in range(n_slots) if any(reach[i, j] for i in range(n_anchors))
    }
    assert serviced == all_reachable, (
        f"Some reachable turrets never got ammo. serviced={sorted(serviced)}, "
        f"reachable={sorted(all_reachable)}"
    )

    # Inventory accounting: turret count dropped by exactly the number placed.
    end_turrets = _inv_count(ns, "gun-turret")
    assert end_turrets == start_turrets - len(placed), (
        f"Turret inventory off: start={start_turrets}, end={end_turrets}, "
        f"placed={len(placed)}"
    )

    # -- 6b. POLLUTE ENEMY STATE (simulate mid-episode expansion) -------------
    # Add a nest and biters that did NOT exist at startup, mimicking biter
    # expansion + spawning during an episode. reset() must rebuild the nest
    # layout to exactly the starting set and clear all live biters.
    n_nests_start = len(nests)
    _spawn_enemy(inst, "biter-spawner", nests[0][0] + 16.0, nests[0][1] + 16.0)
    _spawn_enemy(inst, "small-biter", 8.0, 8.0)
    _spawn_enemy(inst, "small-biter", -8.0, 8.0)
    time.sleep(0.5)
    nests_polluted = _scan_nests(inst)
    biters_polluted = _count_live_biters(inst)
    assert len(nests_polluted) == n_nests_start + 1, (
        f"Failed to add expansion nest: started {n_nests_start}, "
        f"after pollute {len(nests_polluted)}"
    )
    assert biters_polluted >= 1, (
        f"Failed to spawn test biters (got {biters_polluted}) — "
        "cannot verify biter clearing on reset"
    )
    print(f"[pollute] nests {n_nests_start}->{len(nests_polluted)}, "
          f"live biters {biters_polluted}")

    # -- 7. RESET VERIFICATION ------------------------------------------------
    # Simulate what the RL loop does when the episode ends (character dies or
    # radar destroyed): wait a few seconds so the game advances, then call
    # env.reset() and verify the state is fully restored from the snapshot:
    # - Radar alive at original HP.
    # - Turret slot count unchanged (layout read once from the map, never changes).
    # - Deletion re-applied: the same fraction of slots emptied, turrets back in
    #   the agent's inventory (the exact *which* slots changes because re-roll is
    #   random, but the *counts* must be identical).
    # - Character alive and near spawn.
    # - Inventory ammo restored from starting_inventory config.
    print("\n[reset] waiting 3s then calling env.reset() ...")
    time.sleep(3.0)

    obs2, info2 = env.reset()

    assert obs2["radar"][2] > 0, (
        f"Radar HP is zero after reset: {obs2['radar'][2]:.1f}"
    )
    assert abs(float(obs2["radar"][2]) - float(obs["radar"][2])) < 5.0, (
        f"Radar HP diverged between episodes: "
        f"ep1={obs['radar'][2]:.1f}, ep2={obs2['radar'][2]:.1f}"
    )

    n_slots_after = int(obs2["slot_valid_mask"].sum())
    assert n_slots_after == n_slots, (
        f"Turret slot count changed after reset: {n_slots} -> {n_slots_after}"
    )

    n_empty_after = int(obs2["place_slot_mask"].sum())
    assert n_empty_after == n_deleted, (
        f"After reset, expected {n_deleted} empty slots (deletion re-rolled), "
        f"found {n_empty_after}"
    )

    turrets_after_reset = _inv_count(ns, "gun-turret")
    assert turrets_after_reset == n_deleted, (
        f"After reset, expected {n_deleted} gun-turrets in inventory, "
        f"found {turrets_after_reset}"
    )

    assert obs2["character"][2] > 0, "Character HP is zero after reset"
    spawn_x, spawn_y = cfg.player_spawn_position
    # obs character coords are radar-relative + normalized; de-normalize to world.
    cx2 = float(obs2["character"][0]) * env._norm + env._center_x
    cy2 = float(obs2["character"][1]) * env._norm + env._center_y
    assert abs(cx2 - spawn_x) < 2.0 and abs(cy2 - spawn_y) < 2.0, (
        f"Character not at spawn after reset: ({cx2:.1f},{cy2:.1f}), "
        f"expected near ({spawn_x},{spawn_y})"
    )

    ammo_after_reset = int(obs2["inventory"][ammo_idx])
    assert ammo_after_reset > 0, "Agent has no ammo after reset"

    # Enemy nest layout rebuilt to the starting set (expansion nest removed),
    # and all live biters cleared.
    nests_after_reset = _scan_nests(inst)
    assert len(nests_after_reset) == n_nests_start, (
        f"Nest layout not restored after reset: started {n_nests_start}, "
        f"polluted to {len(nests_polluted)}, after reset {len(nests_after_reset)}"
    )
    biters_after_reset = _count_live_biters(inst)
    assert biters_after_reset == 0, (
        f"Live biters remain after reset: {biters_after_reset} "
        "(expected 0 — stale biters should be cleared)"
    )
    print(f"[reset] nests {len(nests_polluted)}->{len(nests_after_reset)} "
          f"(start {n_nests_start}), biters {biters_polluted}->{biters_after_reset}")

    print(
        f"[reset] PASS — radar={obs2['radar'][2]:.0f}HP, "
        f"slots={n_slots_after} ({n_empty_after} empty), "
        f"turrets_in_inv={turrets_after_reset}, "
        f"ammo={ammo_after_reset}, "
        f"char_HP={obs2['character'][2]:.0f}"
    )

    # -- 8. SECOND WALK (post-reset) ------------------------------------------
    # Verify the reset→walk loop works end-to-end and measure how long a full
    # pass takes. The reach matrix is the same (static), but deletion re-rolled
    # so the set of empty slots may differ from episode 1. Character starts at
    # spawn (where reset() left it).
    print("\n[walk2] starting second anchor walk after reset ...")
    t_walk2_start = time.time()

    placed2: set = set()
    serviced2: set = set()
    visited2: list = []
    turrets_left2 = turrets_after_reset
    refill_events2 = 0
    ammo_inserted2 = 0
    empty_slots2 = {j for j in range(n_slots) if obs2["place_slot_mask"][j]}

    # Same biter isolation as walk 1 (reset() recreated the nests active).
    _set_nests_active(inst, False)
    env._clear_live_biters()
    inst.unpause()  # reset() pauses; unpause so A* and glide work
    spawn_x2, spawn_y2 = cfg.player_spawn_position
    curr2 = Position(x=spawn_x2, y=spawn_y2)

    for i in range(n_anchors):
        ax, ay = float(anchors[i, 0]), float(anchors[i, 1])
        target2 = Position(x=ax, y=ay)
        print(f"-> [walk2] anchor[{i}] ({ax:.1f},{ay:.1f})")
        curr2 = _astar_walk(ns, inst, curr2, target2)
        env._current_anchor_index = i
        visited2.append(i)
        time.sleep(ARRIVE_SLEEP)

        reachable_i = [j for j in range(n_slots) if reach[i, j]]
        for j in reachable_i:
            sx, sy = float(slots[j, 0]), float(slots[j, 1])
            turret2 = env._find_turret_at(sx, sy)

            if turret2 is None and turrets_left2 > 0 and j not in placed2:
                try:
                    ns.place_entity(
                        Prototype.GunTurret, Direction.UP, Position(x=sx, y=sy)
                    )
                    placed2.add(j)
                    turrets_left2 -= 1
                    turret2 = env._find_turret_at(sx, sy)
                    print(f"    [walk2] placed turret in slot[{j}] ({sx:.1f},{sy:.1f})")
                except Exception as e:
                    print(f"    [walk2] place slot[{j}] failed: {e}")

            if turret2 is not None:
                try:
                    ns.insert_item(Prototype.FirearmMagazine, turret2, AMMO_PER)
                    serviced2.add(j)
                    refill_events2 += 1
                    ammo_inserted2 += AMMO_PER
                except Exception as e:
                    print(f"    [walk2] refill slot[{j}] failed: {e}")

    t_walk2_elapsed = time.time() - t_walk2_start

    # Walk 2 must cover all anchors and all reachable empty slots.
    assert len(visited2) == n_anchors, (
        f"[walk2] visited {len(visited2)} anchors, expected {n_anchors}"
    )
    assert placed2 == empty_slots2, (
        f"[walk2] empty slots not fully refilled. placed={sorted(placed2)}, "
        f"expected={sorted(empty_slots2)}"
    )
    assert serviced2 == all_reachable, (
        f"[walk2] not all reachable turrets got ammo. serviced={sorted(serviced2)}, "
        f"reachable={sorted(all_reachable)}"
    )

    end_turrets2 = _inv_count(ns, "gun-turret")
    assert end_turrets2 == turrets_after_reset - len(placed2), (
        f"[walk2] turret inventory off: start={turrets_after_reset}, "
        f"end={end_turrets2}, placed={len(placed2)}"
    )

    print(f"[walk2] done in {t_walk2_elapsed:.1f}s")

    print(
        "\n=== FULL INTEGRATION SUMMARY ===\n"
        f"  anchors visited     : {len(visited)}/{n_anchors} (ep1) | "
        f"{len(visited2)}/{n_anchors} (ep2)\n"
        f"  turret slots        : {n_slots} ({n_deleted} started empty)\n"
        f"  biter nests         : {len(nests)}\n"
        f"  ep1 placed/serviced : {len(placed)}/{len(all_reachable)} | "
        f"refills={refill_events} ({ammo_inserted} mags)\n"
        f"  ep2 placed/serviced : {len(placed2)}/{len(all_reachable)} | "
        f"refills={refill_events2} ({ammo_inserted2} mags)\n"
        f"  ep2 walk time       : {t_walk2_elapsed:.1f}s\n"
        f"  gun-turret inventory: {start_turrets} (ep1 start) -> "
        f"{end_turrets} (ep1 end) -> {turrets_after_reset} (reset) -> "
        f"{end_turrets2} (ep2 end)\n"
        "================================"
    )
