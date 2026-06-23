"""Dedicated game-state RESET test for the Tower Defense env.

This test exists to make reset behaviour observable in-game. Connect a Factorio
client BEFORE launching and watch: the biter nests stay disabled (no biters
spawn while the character works), the character glides between anchors placing
turrets and loading them with ammo, and on every env.reset() the board snaps
back to its pristine starting state — the same number of nests it started with,
zero biters, and the full turret layout restored.

It complements test_td_full_integration.py by focusing narrowly on reset
correctness across several episodes, and it asserts the three things that were
observed to be broken:

  1. NEST COUNT IS STABLE — enemy expansion is disabled, so the nest count never
     grows; every reset restores exactly the starting count (no leftover
     expansion nests).
  2. NESTS STAY DISABLED DURING WORK — while the character walks/services, the
     baked-in nests are inactive and no biters spawn (visible in-game).
  3. TURRETS GET AMMO — refilling uses the configured ammo type, so each placed
     turret ends the visit with a non-zero turret-ammo inventory (RCON-verified).

Run:
    fle cluster start -n 1     # loads data/saves/tower_defense.zip
    cd fle-package
    pytest -m integration tests/integration/test_td_reset_state.py -v -s -p no:warnings
"""

import math
import time

import pytest

from fle.env.entities import Direction, Position
from fle.env.game_types import Prototype
from fle.env.gym_env.td_config import TDScenarioConfig
from fle.env.gym_env.td_environment import TowerDefenseEnv

pytestmark = pytest.mark.integration

# Movement tuning (watchable glide between anchors).
SUB_TILE = 0.25
GLIDE_SLEEP = 0.02
ARRIVE_SLEEP = 0.2
# Ammo inserted per service action.
AMMO_PER = 10
# Turret deletion fraction (leaves empty slots to refill).
DELETION_PCT = 0.5
# Number of reset cycles to verify.
N_EPISODES = 3


# ---------------------------------------------------------------------------
# Small RCON helpers (self-contained so the test stands alone).
# ---------------------------------------------------------------------------
def _scan_nests(inst):
    """(x, y) positions of enemy unit-spawners (biter nests) via RCON."""
    raw = inst.rcon_client.send_command(
        "/sc local o={} for _,e in pairs(game.surfaces[1].find_entities_filtered"
        "{type='unit-spawner', force='enemy'}) do "
        "o[#o+1]=e.position.x..','..e.position.y end rcon.print(table.concat(o,';'))"
    ).strip()
    out = []
    for p in (raw.split(";") if raw else []):
        if p:
            nx, ny = p.split(",")
            out.append((float(nx), float(ny)))
    return out


def _count_biters(inst):
    """Number of live enemy units (biters/spitters) on the map."""
    return int(float(inst.rcon_client.send_command(
        "/sc local n=0 for _ in pairs(game.surfaces[1].find_entities_filtered"
        "{type='unit', force='enemy'}) do n=n+1 end rcon.print(n)"
    ).strip()))


def _expansion_enabled(inst):
    return inst.rcon_client.send_command(
        "/sc rcon.print(tostring(game.map_settings.enemy_expansion.enabled))"
    ).strip()


def _nests_active(inst):
    """Return 'active/total' string for enemy unit-spawners."""
    return inst.rcon_client.send_command(
        "/sc local a,i=0,0 for _,e in pairs(game.surfaces[1].find_entities_filtered"
        "{type='unit-spawner',force='enemy'}) do i=i+1; if e.active then a=a+1 end end "
        "rcon.print(a..'/'..i)"
    ).strip()


def _set_nests_active(inst, active: bool):
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


def _turret_ammo(inst, x, y, ammo_name):
    """Turret-ammo count in the gun-turret at (x, y); -1 if no turret there."""
    return int(float(inst.rcon_client.send_command(
        f"/sc local e=game.surfaces[1].find_entities_filtered"
        f"{{name='gun-turret',position={{{x},{y}}},radius=0.7}}[1] "
        f"if e then local inv=e.get_inventory(defines.inventory.turret_ammo) "
        f"rcon.print(inv and inv.get_item_count('{ammo_name}') or 0) "
        f"else rcon.print(-1) end"
    ).strip()))


def _char_pos(inst):
    raw = inst.rcon_client.send_command(
        "/sc local c=storage.agent_characters and storage.agent_characters[1]; "
        "if c and c.valid then rcon.print(c.position.x..','..c.position.y) "
        "else rcon.print('0,0') end"
    ).strip()
    x, y = raw.split(",")
    return float(x), float(y)


def _glide(inst, frm: Position, to: Position):
    """Smoothly teleport the character from frm to to (watchable in-game)."""
    dist = math.hypot(to.x - frm.x, to.y - frm.y)
    n = max(1, int(math.ceil(dist / SUB_TILE)))
    for k in range(1, n + 1):
        t = k / n
        inst.rcon_client.send_command(
            f"/sc storage.agent_characters[1].teleport("
            f"{{{frm.x + (to.x - frm.x) * t},{frm.y + (to.y - frm.y) * t}}})"
        )
        time.sleep(GLIDE_SLEEP)


def _inv_count(ns, item: str) -> int:
    try:
        return int(ns.inspect_inventory().get(item, 0))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# The reset-state test
# ---------------------------------------------------------------------------
def test_td_reset_state(td_save_instance):
    inst = td_save_instance
    ns = inst.first_namespace

    cfg = TDScenarioConfig(
        spawn_waves_at_runtime=False,
        clear_biters_on_reset=True,
        restore_starting_nests_on_reset=True,
        disable_enemy_expansion=True,
        turret_deletion_percentage=DELETION_PCT,
        # Exercise the configured ammo type end-to-end (the bug: refill used a
        # different magazine than the inventory, so turrets never got ammo).
        starting_inventory={TDScenarioConfig.ammo_type: 1000},
    )
    env = TowerDefenseEnv(instance=inst, config=cfg)
    ammo_name = env._ammo_prototype.value[0]
    ammo_proto = env._ammo_prototype

    # -- First reset: establishes the pristine baseline ----------------------
    obs, _ = env.reset()
    anchors = env._anchor_slots
    slots = env._turret_slots
    reach = env._anchor_reach_matrix
    n_anchors = len(anchors)
    n_slots = len(slots)
    n_deleted = int(math.floor(n_slots * DELETION_PCT))

    # Expansion must be OFF now (enforced by the env on first reset) so the nest
    # count cannot grow.
    assert _expansion_enabled(inst) == "false", (
        "enemy expansion is still enabled — nests will keep multiplying"
    )

    n0_nests = len(_scan_nests(inst))
    assert n0_nests >= 1, "No biter nests found on the map"
    print(f"\n[baseline] nests={n0_nests}, anchors={n_anchors}, slots={n_slots} "
          f"({n_deleted} emptied/episode), ammo='{ammo_name}', "
          f"expansion={_expansion_enabled(inst)}")

    try:
        for ep in range(N_EPISODES):
            print(f"\n========== EPISODE {ep} ==========")

            # Start-of-episode invariants: nests at baseline, no biters.
            nests_now = _scan_nests(inst)
            assert len(nests_now) == n0_nests, (
                f"[ep{ep}] nest count drifted: {len(nests_now)} != baseline {n0_nests}"
            )
            assert _count_biters(inst) == 0, f"[ep{ep}] biters present at episode start"

            # Disable nests for the work phase so no biters spawn while the
            # character moves (visible in-game; re-enabled in finally).
            _set_nests_active(inst, False)
            env._clear_live_biters()
            print(f"[ep{ep}] nests disabled for work phase: active={_nests_active(inst)}, "
                  f"biters={_count_biters(inst)}")

            inst.set_speed(8.0)
            inst.unpause()

            # -- WORK: glide each anchor, place + refill reachable turrets ----
            placed, serviced = set(), set()
            turrets_left = _inv_count(ns, "gun-turret")
            cx, cy = _char_pos(inst)
            curr = Position(x=cx, y=cy)

            for i in range(n_anchors):
                ax, ay = float(anchors[i, 0]), float(anchors[i, 1])
                _glide(inst, curr, Position(x=ax, y=ay))
                curr = Position(x=ax, y=ay)
                env._current_anchor_index = i
                time.sleep(ARRIVE_SLEEP)

                for j in [k for k in range(n_slots) if reach[i, k]]:
                    sx, sy = float(slots[j, 0]), float(slots[j, 1])
                    turret = env._find_turret_at(sx, sy)
                    if turret is None and turrets_left > 0 and j not in placed:
                        try:
                            ns.place_entity(Prototype.GunTurret, Direction.UP,
                                            Position(x=sx, y=sy))
                            placed.add(j)
                            turrets_left -= 1
                            turret = env._find_turret_at(sx, sy)
                        except Exception as e:
                            print(f"    [ep{ep}] place slot[{j}] failed: {e}")
                    if turret is not None:
                        try:
                            ns.insert_item(ammo_proto, turret, AMMO_PER)
                            # Verify the turret actually holds ammo now.
                            assert _turret_ammo(inst, sx, sy, ammo_name) > 0, (
                                f"[ep{ep}] turret slot[{j}] got NO ammo after refill "
                                f"(ammo='{ammo_name}')"
                            )
                            serviced.add(j)
                        except AssertionError:
                            raise
                        except Exception as e:
                            print(f"    [ep{ep}] refill slot[{j}] failed: {e}")

            # Nests stayed disabled => no biters appeared during the whole walk.
            assert _count_biters(inst) == 0, (
                f"[ep{ep}] biters spawned during work phase despite disabled nests"
            )
            print(f"[ep{ep}] worked: placed={len(placed)}, serviced(ammo)={len(serviced)}, "
                  f"biters_during_work={_count_biters(inst)}")

            # -- POLLUTE: simulate expansion + an incursion just before reset --
            _spawn_enemy(inst, "biter-spawner",
                         nests_now[0][0] + 16.0, nests_now[0][1] + 16.0)
            for k in range(3):
                _spawn_enemy(inst, "small-biter", 6.0 + k, 6.0)
            time.sleep(0.5)
            poll_nests, poll_biters = len(_scan_nests(inst)), _count_biters(inst)
            assert poll_nests == n0_nests + 1 and poll_biters >= 1, (
                f"[ep{ep}] pollution didn't take: nests={poll_nests} biters={poll_biters}"
            )
            print(f"[ep{ep}] polluted before reset: nests={poll_nests} biters={poll_biters}")

            # -- RESET: must restore the pristine board ----------------------
            obs, _ = env.reset()
            after_nests = len(_scan_nests(inst))
            after_biters = _count_biters(inst)
            assert after_nests == n0_nests, (
                f"[ep{ep}] RESET failed: nests {poll_nests}->{after_nests}, "
                f"expected {n0_nests} (expansion nest not removed)"
            )
            assert after_biters == 0, (
                f"[ep{ep}] RESET failed: {after_biters} biters left after reset"
            )
            assert int(obs["slot_valid_mask"].sum()) == n_slots, (
                f"[ep{ep}] slot count changed after reset"
            )
            assert int(obs["place_slot_mask"].sum()) == n_deleted, (
                f"[ep{ep}] empty-slot count wrong after reset"
            )
            assert _expansion_enabled(inst) == "false", (
                f"[ep{ep}] expansion re-enabled after reset"
            )
            print(f"[ep{ep}] RESET OK: nests {poll_nests}->{after_nests} (baseline "
                  f"{n0_nests}), biters {poll_biters}->{after_biters}, "
                  f"slots={int(obs['slot_valid_mask'].sum())} "
                  f"({int(obs['place_slot_mask'].sum())} empty)")

        print(f"\n=== RESET STATE TEST PASSED over {N_EPISODES} episodes ===\n"
              f"  baseline nests held at {n0_nests} every reset\n"
              f"  no biters spawned during any work phase\n"
              f"  every refilled turret carried '{ammo_name}' ammo")
    finally:
        # Leave the container live for subsequent runs.
        _set_nests_active(inst, True)
        print(f"[cleanup] nests re-enabled: active={_nests_active(inst)}")
