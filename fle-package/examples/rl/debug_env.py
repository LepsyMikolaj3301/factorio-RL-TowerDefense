"""Single-environment debug runner for TowerDefenseEnv.

Connects to a live Factorio container, runs one episode step-by-step, and
prints a detailed report every step: pause/unpause timing, every observation
array (shape + key stats), reward breakdown, and info dict.

Usage:
    # Start a container first
    fle cluster start -n 1

    # Run with default settings (10 steps, noop policy, game_speed=1 for visibility)
    python examples/rl/debug_env.py

    # More steps, random policy
    python examples/rl/debug_env.py --steps 20 --policy random

    # Keep game speed at 10x (faster, harder to watch)
    python examples/rl/debug_env.py --game-speed 10
"""

import argparse
import logging
import time

import numpy as np

from fle.env.gym_env.td_config import TDScenarioConfig
from fle.env.gym_env.td_spaces import (
    ACTION_MOVE_ANCHOR,
    ACTION_NOOP,
    ACTION_PICK_TURRET,
    ACTION_PLACE_TURRET,
    ACTION_REFILL_TURRET,
    MAX_ANCHORS,
    MAX_SLOTS,
)

ACTION_NAMES = {
    ACTION_NOOP: "NOOP",
    ACTION_PICK_TURRET: "PICK_TURRET",
    ACTION_PLACE_TURRET: "PLACE_TURRET",
    ACTION_REFILL_TURRET: "REFILL_TURRET",
    ACTION_MOVE_ANCHOR: "MOVE_ANCHOR",
}

SEP = "─" * 72


def _obs_summary(obs: dict) -> list:
    """Return a list of human-readable lines summarising the observation dict."""
    lines = []

    # --- map grid channel occupancy ---
    grid = obs["map"]  # (C, H, W) uint8
    ch_names = ["empty", "wall", "turret", "ammo%", "biter", "spitter", "spawner", "char"]
    total_cells = grid.shape[1] * grid.shape[2]
    lines.append("  map grid channels:")
    for ch, name in enumerate(ch_names):
        nz = int(np.count_nonzero(grid[ch]))
        max_v = int(grid[ch].max())
        bar_len = min(30, nz * 30 // max(1, total_cells))
        bar = "█" * bar_len + "░" * (30 - bar_len)
        lines.append(f"    ch{ch} {name:<8}  {bar}  {nz:>4}/{total_cells} cells  max={max_v}")

    # --- scalar fields ---
    char = obs["character"]
    lines.append(
        f"  character : pos=({char[0]:.3f},{char[1]:.3f}) "
        f"hp={char[2]:.2f}  ammo={char[3]:.2f}"
    )
    radar = obs["radar"]
    lines.append(
        f"  radar     : pos=({radar[0]:.3f},{radar[1]:.3f}) hp={radar[2]:.3f}"
    )
    game = obs["game"]
    lines.append(f"  game      : elapsed_norm={game[0]:.4f}  wave={int(game[1])}")

    inv = obs["inventory"]
    lines.append(f"  inventory : {list(inv)}")

    # --- turret slots ---
    slot_valid = obs["slot_valid_mask"]
    place_mask = obs["place_slot_mask"]
    refill_mask = obs["refill_slot_mask"]
    pick_mask = obs["pick_slot_mask"]
    n_valid = int(slot_valid.sum())
    n_empty = int(place_mask.sum())
    n_occupied = n_valid - n_empty
    n_refillable = int(refill_mask.sum())
    lines.append(
        f"  slots     : {n_valid} valid  {n_occupied} occupied  "
        f"{n_empty} empty  {n_refillable} refillable"
    )

    # --- anchors ---
    anch_valid = obs["anchor_valid_mask"]
    n_anch = int(anch_valid.sum())
    lines.append(f"  anchors   : {n_anch} valid")

    # --- movement ---
    mov = obs["movement"]
    if mov[0] > 0.5:
        lines.append(
            f"  movement  : IN TRANSIT → anchor {mov[1] * MAX_ANCHORS:.0f}  "
            f"dist_norm={mov[2]:.3f}  heading=({mov[3]:.2f},{mov[4]:.2f})"
        )
    else:
        lines.append("  movement  : STATIONARY")

    # --- threat ---
    gvm = obs["group_valid_mask"]
    nvm = obs["nest_valid_mask"]
    n_groups = int(gvm.sum())
    n_nests = int(nvm.sum())
    if n_groups > 0:
        groups = obs["biter_groups"]
        lines.append(f"  threats   : {n_groups} biter group(s), {n_nests} nest(s)")
        for i in range(n_groups):
            g = groups[i]
            swarm = " [SWARM]" if g[9] > 0.5 else ""
            lines.append(
                f"    group[{i}]  cx=({g[0]:.2f},{g[1]:.2f})  "
                f"count={g[2]:.1f}  dist={g[6]:.2f}  eta={g[8]:.2f}{swarm}"
            )
    else:
        lines.append(f"  threats   : 0 biter groups, {n_nests} nest(s)")

    # --- reach masks ---
    pr = obs["place_reach_mask"]
    rr = obs["refill_reach_mask"]
    pk = obs["pick_reach_mask"]
    lines.append(
        f"  reach     : place={int(pr.sum())}  refill={int(rr.sum())}  pick={int(pk.sum())} reachable slots"
    )

    return lines


def _fmt_action(action: dict) -> str:
    atype = int(action.get("action_type", 0))
    name = ACTION_NAMES.get(atype, f"?({atype})")
    slot = int(action.get("slot_index", 0))
    anchor = int(action.get("anchor_index", 0))
    ammo = int(action.get("ammo_amount", 0))
    if atype == ACTION_MOVE_ANCHOR:
        return f"{name}(anchor={anchor})"
    elif atype in (ACTION_PLACE_TURRET, ACTION_PICK_TURRET):
        return f"{name}(slot={slot})"
    elif atype == ACTION_REFILL_TURRET:
        return f"{name}(slot={slot}, ammo={ammo})"
    return name


def run_debug(n_steps: int = 10, policy: str = "noop", game_speed: float = 1.0):
    # Enable DEBUG logging for the env so the pause/unpause timing shows up.
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Only show DEBUG from our env; keep other libraries quiet.
    logging.getLogger("fle.env.gym_env.td_environment").setLevel(logging.DEBUG)
    logging.getLogger("fle").setLevel(logging.INFO)
    logging.getLogger("fle.env.gym_env.td_environment").setLevel(logging.DEBUG)
    for noisy in ("urllib3", "requests", "docker", "factorio_rcon"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    import dataclasses
    from fle.env.gym_env.registry import make_td_env

    cfg = dataclasses.replace(
        TDScenarioConfig.MEDIUM,
        game_speed=game_speed,
        spawn_waves_at_runtime=False,
        clear_biters_on_reset=True,
        turret_deletion_percentage=0.5,
    )

    print(f"\n{SEP}")
    print("  TD env debug runner")
    print(f"  steps={n_steps}  policy={policy}  game_speed={game_speed}x")
    print(f"  cadence={cfg.decision_cadence} ticks / step  "
          f"(wall ≈ {cfg.decision_cadence/60/game_speed:.2f}s/step)")
    print(SEP)

    print("\n[init] Creating environment and connecting to Factorio container...")
    t_init = time.perf_counter()
    env = make_td_env(run_idx=0, config=cfg)
    print(f"[init] Connected in {time.perf_counter() - t_init:.2f}s\n")

    # reset() prints the save validation report itself.
    print("[reset] Calling env.reset() ...")
    t_reset = time.perf_counter()
    obs, info = env.reset()
    print(f"[reset] Done in {time.perf_counter() - t_reset:.2f}s")

    print(f"\n{SEP}  INITIAL OBSERVATION  {SEP}")
    for line in _obs_summary(obs):
        print(line)
    print(SEP)

    # --- step loop ---
    total_reward = 0.0
    total_invalid = 0
    t_episode = time.perf_counter()

    for step in range(1, n_steps + 1):
        # Pick action
        if policy == "noop":
            action = {"action_type": ACTION_NOOP, "slot_index": 0,
                      "anchor_index": 0, "ammo_amount": 0}
        elif policy == "random":
            action = env.action_space.sample()
        else:
            action = {"action_type": ACTION_NOOP, "slot_index": 0,
                      "anchor_index": 0, "ammo_amount": 0}

        action_str = _fmt_action(action)
        print(f"\n{'━'*72}")
        print(f"  STEP {step:>3} / {n_steps}   action={action_str}")
        print(f"{'━'*72}")

        t_step = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        wall = time.perf_counter() - t_step
        total_reward += reward
        if info.get("invalid_action"):
            total_invalid += 1

        # Reward breakdown
        print(f"  reward        : {reward:+.4f}  (cumulative {total_reward:+.4f})")
        print(f"  terminated    : {terminated}  truncated={truncated}")
        print(f"  wall time     : {wall:.3f}s")

        # Info
        ticks = info.get("elapsed_ticks", "?")
        kills = info.get("kills", 0)
        t_lost = info.get("turrets_lost", 0)
        w_lost = info.get("walls_lost", 0)
        cov = info.get("coverage", 0.0)
        moving = info.get("is_moving", False)
        invalid = info.get("invalid_action", False)
        print(
            f"  ticks         : {ticks}\n"
            f"  kills         : {kills}  turrets_lost={t_lost}  walls_lost={w_lost}\n"
            f"  coverage      : {cov:.2%}  is_moving={moving}  invalid={invalid}"
        )

        # Observation
        print(f"\n  — observation —")
        for line in _obs_summary(obs):
            print(line)

        if terminated or truncated:
            reason = "TERMINATED (radar/char died)" if terminated else "TRUNCATED (time limit)"
            print(f"\n  *** {reason} ***")
            break

    elapsed = time.perf_counter() - t_episode
    print(f"\n{SEP}")
    print(f"  Episode summary")
    print(f"    steps run     : {step}")
    print(f"    total reward  : {total_reward:+.4f}")
    print(f"    invalid acts  : {total_invalid}")
    print(f"    wall time     : {elapsed:.2f}s  ({elapsed/step:.2f}s/step)")
    print(SEP)

    env.close()


def main():
    p = argparse.ArgumentParser(description="Debug one TowerDefenseEnv episode")
    p.add_argument("--steps", type=int, default=10,
                   help="Number of steps to run (default: 10)")
    p.add_argument("--policy", choices=["noop", "random"], default="noop",
                   help="Action policy: noop (safe, shows timing) or random")
    p.add_argument("--game-speed", type=float, default=1.0,
                   help="Factorio game speed multiplier (default: 1.0 for visibility, "
                        "10.0 for normal training speed)")
    args = p.parse_args()
    run_debug(n_steps=args.steps, policy=args.policy, game_speed=args.game_speed)


if __name__ == "__main__":
    main()
