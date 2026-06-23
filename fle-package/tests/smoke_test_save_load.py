#!/usr/bin/env python3
"""
Grand startup smoke test for the TowerDefense scenario.

Starts a single container with the prebuilt save, runs the full
TowerDefenseEnv.reset() startup sequence (spawns the agent character,
charts the map, sets inventory, reads turret slots), then unpauses the
game and prints a summary so you can join in-game and eyeball everything.

Run from fle-package/:
    python tests/smoke_test_save_load.py [--save PATH] [--keep-alive] [--speed N]

Flags:
    --save PATH     path to .zip save (default: data/saves/tower_defense.zip)
    --keep-alive    leave the container running after the test for in-game inspection
    --speed N       game speed multiplier while running (default: 1.0 — real-time)
"""

import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from factorio_rcon import RCONClient

from fle.cluster.run_envs import (
    RCON_PASSWORD,
    START_GAME_PORT,
    START_RCON_PORT,
    ClusterManager,
)

RCON_HOST = "localhost"
RCON_TIMEOUT = 90


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rcon_wait(host: str, port: int, password: str, timeout: int) -> RCONClient:
    """Block until RCON is genuinely ready, not just until the TCP port opens.

    On a cold start Factorio opens the RCON listener *before* the save has
    finished loading. A plain TCP accept + a no-timeout RCONClient.connect()
    will then block forever on the auth read ("server doesn't respond"). So we
    use a finite per-attempt timeout AND require a real command to return a
    sane value before declaring the server up.
    """
    deadline = time.time() + timeout
    print(f"  Polling RCON on {host}:{port} (up to {timeout}s) ...", end="", flush=True)
    while time.time() < deadline:
        client = None
        try:
            # Finite timeout: a mid-load server that accepts TCP but never
            # answers the auth/probe will raise instead of hanging forever.
            client = RCONClient(host, port, password, timeout=5)
            # Probe with a real command — proves the game (not just the
            # listener) is responding.
            tick = client.send_command("/sc rcon.print(game.tick)")
            if tick is not None and str(tick).strip().isdigit():
                print(" up.")
                return client
            raise RuntimeError(f"unexpected probe response: {tick!r}")
        except Exception:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            print(".", end="", flush=True)
            time.sleep(2)
    print()
    raise TimeoutError(f"RCON on {host}:{port} did not become available within {timeout}s")


def _q(rcon: RCONClient, cmd: str) -> str:
    return (rcon.send_command(cmd) or "").strip()


# ---------------------------------------------------------------------------
# Phase 1: container + raw RCON
# ---------------------------------------------------------------------------

def start_container(save_path: Path) -> RCONClient:
    print(f"\n{'='*60}")
    print(f"[1/4] Starting container  save={save_path.name}")
    print(f"{'='*60}")

    manager = ClusterManager()
    try:
        manager.start(
            num_instances=1,
            scenario="tower_defense",
            save_file=str(save_path),
        )
    except SystemExit as e:
        sys.exit(f"Cluster start failed: {e}")

    rcon = _rcon_wait(RCON_HOST, START_RCON_PORT, RCON_PASSWORD, RCON_TIMEOUT)
    version = _q(rcon, "/version")
    tick = _q(rcon, "/sc rcon.print(game.tick)")
    print(f"  server version : {version}")
    print(f"  current tick   : {tick}")
    return rcon


# ---------------------------------------------------------------------------
# Phase 2: full TowerDefenseEnv.reset() startup
# ---------------------------------------------------------------------------

def run_scenario_startup(rcon: RCONClient) -> dict:
    """Create FactorioInstance + TowerDefenseEnv, call reset(), return summary."""
    print(f"\n{'='*60}")
    print("[2/4] Running TowerDefenseEnv startup (reset)")
    print(f"{'='*60}")

    from fle.env import FactorioInstance
    from fle.env.gym_env.td_environment import TowerDefenseEnv
    from fle.env.gym_env.td_config import TDScenarioConfig

    print("  Creating FactorioInstance ...")
    inst = FactorioInstance(
        address=RCON_HOST,
        tcp_port=START_RCON_PORT,
        inventory={
            "gun-turret": 5,
            "firearm-magazine": 100,
            "stone-wall": 20,
            "pistol": 1,
        },
        clear_entities=False,   # preserve the save's radar/turrets/walls
        peaceful=False,
        cache_scripts=True,
        fast=True,
        all_technologies_researched=True,
    )

    cfg = TDScenarioConfig(
        game_speed=1.0,         # real-time so you can spectate comfortably
        spawn_waves_at_runtime=True,
        clear_biters_on_reset=True,
        turret_deletion_percentage=0.0,  # keep all turrets for the visual check
    )

    print("  Creating TowerDefenseEnv ...")
    env = TowerDefenseEnv(instance=inst, config=cfg)

    print("  Calling env.reset() — spawning player, charting map, reading slots ...")
    obs, info = env.reset()

    return {"env": env, "inst": inst, "obs": obs, "info": info}


# ---------------------------------------------------------------------------
# Phase 3: summarise what the env sees
# ---------------------------------------------------------------------------

def print_startup_summary(ctx: dict):
    obs = ctx["obs"]
    env = ctx["env"]
    inst = ctx["inst"]
    rcon = inst.rcon_client

    print(f"\n{'='*60}")
    print("[3/4] Startup summary")
    print(f"{'='*60}")

    # --- Character ---
    char = obs["character"]
    print(f"\n  CHARACTER")
    print(f"    position : ({char[0]:.1f}, {char[1]:.1f})")
    print(f"    health   : {char[2]:.0f}")
    raw_health = _q(rcon,
        "/sc local c=storage.agent_characters and storage.agent_characters[1];"
        " rcon.print(c and c.valid and c.health or -1)"
    )
    print(f"    health (live RCON check) : {raw_health}")

    # --- Radar ---
    radar = obs["radar"]
    print(f"\n  RADAR")
    print(f"    position : ({radar[0]:.1f}, {radar[1]:.1f})")
    print(f"    health   : {radar[2]:.0f}")
    raw_radar = _q(rcon,
        "/sc local e=game.surfaces[1].find_entities_filtered{name='radar'}[1];"
        " rcon.print(e and e.health or 'not found')"
    )
    print(f"    health (live RCON check) : {raw_radar}")

    # --- Inventory ---
    inv = obs["inventory"]
    print(f"\n  INVENTORY (agent)")
    from fle.env.gym_env.td_spaces import TRACKED_ITEMS
    for i, item in enumerate(TRACKED_ITEMS):
        if i < len(inv):
            print(f"    {item:25s}: {inv[i]}")

    # --- Turret slots ---
    n_valid = int(obs["slot_valid_mask"].sum())
    n_occupied = int(obs["refill_slot_mask"].sum())
    n_empty = int(obs["place_slot_mask"].sum())
    print(f"\n  TURRET SLOTS")
    print(f"    total slots : {n_valid}")
    print(f"    occupied    : {n_occupied}")
    print(f"    empty       : {n_empty}")

    # --- Enemy spawners ---
    raw_spawners = _q(rcon,
        "/sc local n=0 for _,_ in pairs(game.surfaces[1].find_entities_filtered("
        "{type='unit-spawner',force='enemy'})) do n=n+1 end rcon.print(n)"
    )
    print(f"\n  ENEMIES")
    print(f"    spawner count : {raw_spawners}")
    raw_units = _q(rcon,
        "/sc local n=0 for _,_ in pairs(game.surfaces[1].find_entities_filtered("
        "{type='unit',force='enemy'})) do n=n+1 end rcon.print(n)"
    )
    print(f"    live units    : {raw_units}")

    # --- Map grid quick-check ---
    import numpy as np
    grid = obs["map"]
    ch_names = ["empty","wall","turret","ammo%","biter","spitter","spawner","character"]
    print(f"\n  MAP GRID  shape={grid.shape}")
    for i, name in enumerate(ch_names):
        nonzero = int(np.count_nonzero(grid[i]))
        print(f"    ch{i} {name:12s}: {nonzero} non-zero cells")

    # --- Game tick ---
    raw_tick = _q(rcon, "/sc rcon.print(game.tick)")
    raw_elapsed = _q(rcon, "/sc rcon.print(storage.elapsed_ticks or 0)")
    print(f"\n  GAME")
    print(f"    game.tick      : {raw_tick}")
    print(f"    elapsed_ticks  : {raw_elapsed}")


# ---------------------------------------------------------------------------
# Phase 4: unpause and hand off for visual inspection
# ---------------------------------------------------------------------------

def unpause_for_inspection(ctx: dict, speed: float):
    env = ctx["env"]
    inst = ctx["inst"]

    print(f"\n{'='*60}")
    print("[4/4] Unpausing game for in-game inspection")
    print(f"{'='*60}")

    inst.set_speed(speed)
    inst.unpause()

    raw_paused = _q(inst.rcon_client, "/sc rcon.print(game.tick_paused)")
    print(f"  game.tick_paused : {raw_paused}")
    print(f"  game speed       : {speed}x")


def print_connection_info():
    game_port = START_GAME_PORT
    print(f"\n{'='*60}")
    print("  HOW TO CONNECT IN-GAME (SPECTATOR)")
    print(f"{'='*60}")
    print(f"  1. Open Factorio 2.0.73")
    print(f"  2. Multiplayer → Connect to server")
    print(f"  3. Address : localhost:{game_port}  (UDP)")
    print(f"  4. No game password.")
    print()
    print(f"  ACCESS (whitelist is active):")
    print(f"    Option A — set your Factorio username to 'client_master'")
    print(f"               (that's the admin; admins bypass the whitelist)")
    print(f"               No Factorio.com account needed (require_user_verification=false).")
    print(f"    Option B — add your username to:")
    print(f"               fle-package/fle/cluster/config/server-whitelist.json")
    print(f"               e.g.:  [\"your_name\"]  then restart the container.")
    print()
    print(f"  The agent character is at position (0, 10) — right behind the radar.")
    print(f"  Stop the container later: fle cluster stop")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TowerDefense grand startup smoke test")
    parser.add_argument(
        "--save",
        default=str(Path(__file__).parent.parent / "data" / "saves" / "tower_defense.zip"),
        help="Path to the .zip save file",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Leave the container running so you can join in-game to inspect",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Game speed while running for visual inspection (default: 1.0 = real-time)",
    )
    args = parser.parse_args()

    save_path = Path(args.save).expanduser().resolve()
    if not save_path.exists():
        sys.exit(f"ERROR: save file not found: {save_path}")

    manager = ClusterManager()

    try:
        _rcon_raw = start_container(save_path)
        ctx = run_scenario_startup(_rcon_raw)
        print_startup_summary(ctx)
        unpause_for_inspection(ctx, args.speed)
        print_connection_info()

        if args.keep_alive:
            print("Container is still running.  Press Ctrl+C when done, then run:")
            print("  fle cluster stop")
            try:
                while True:
                    time.sleep(5)
            except KeyboardInterrupt:
                print("\nCtrl+C received.")
        else:
            print("Stopping container (pass --keep-alive to leave it running) ...")
            ctx["env"].close()
            manager.stop()
            print("Done.")

    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback; traceback.print_exc()
        print("\nCheck container logs with:  fle cluster logs")
        sys.exit(1)


if __name__ == "__main__":
    main()
