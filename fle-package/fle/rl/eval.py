"""Tower Defense RL evaluation pipeline (inference, no training).

Take an already-trained model and run it on the scenario exactly as during
training, ending when the episode terminates (radar destroyed, boiler loss, or
character death) or truncates (max ticks reached).

Canonical entry point:

    # Start a container first (loads data/saves/tower_defense.zip):
    fle cluster start -n 1

    # Evaluate a saved model on one episode:
    python -m fle.rl.eval --model td_runs/<run>/final_model.zip --evolution-factor 0.5

    # Slow the game down to watch it play:
    python -m fle.rl.eval --model td_runs/<run>/final_model.zip --game-speed 1.0

Requires the RL extra (``pip install 'fle[rl]'``) for ``sb3-contrib``
(MaskablePPO). The model is loaded with the same wrapper stack used in
training (``FlatTDActionWrapper`` -> ``ActionMaskWrapper``), so the policy sees
the observation/action shapes it was trained on.

Difficulty is set by ``--evolution-factor`` and ``--group-size`` (the biter
attack-group cap), applied to the world on the first reset and re-applied after
every reset. Both default to the save's own values when omitted. The
observation and action spaces are independent of these knobs, so any saved
model loads regardless.
"""

import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import fields
from typing import Optional

import numpy as np

from fle.rl.config import EvalConfig


# ---------------------------------------------------------------------------
# Env construction (mirrors fle.rl.train so the model sees the same wrappers)
# ---------------------------------------------------------------------------
def _scenario_config(cfg: EvalConfig):
    """Build the TD scenario config from the difficulty knobs (evolution factor
    + biter group size). Starts from the base config; None leaves save defaults."""
    from fle.env.gym_env.td_config import TDScenarioConfig

    scenario = dataclasses.replace(
        TDScenarioConfig(),
        evolution_factor=cfg.evolution_factor,
        max_unit_group_size=cfg.max_unit_group_size,
    )
    if cfg.game_speed is not None:
        scenario = dataclasses.replace(scenario, game_speed=cfg.game_speed)
    return scenario


def _make_eval_env(cfg: EvalConfig):
    """Build one fully-wrapped, maskable TD env (single, not vectorized)."""
    from fle.env.gym_env.registry import make_td_env
    from fle.env.gym_env.td_spaces import FlatTDActionWrapper
    from fle.env.gym_env.action_mask import ActionMaskWrapper

    scenario = _scenario_config(cfg)

    env = make_td_env(
        run_idx=cfg.run_idx,
        save_path=cfg.save_path,
        config=scenario,
    )
    env = FlatTDActionWrapper(env)      # Dict -> MultiDiscrete
    env = ActionMaskWrapper(env)        # adds masks + action_masks()
    return env


# ---------------------------------------------------------------------------
# End-reason inference
# ---------------------------------------------------------------------------
def _end_reason(env, obs: dict, terminated: bool, truncated: bool, hit_cap: bool) -> str:
    """Classify why the episode ended, from the final observation."""
    if truncated:
        return "time_limit"
    if hit_cap and not terminated:
        return "max_steps"
    if not terminated:
        return "incomplete"

    radar = np.asarray(obs.get("radar", [0, 0, 1.0]))
    if radar.shape[0] >= 3 and float(radar[2]) <= 0.0:
        return "radar_destroyed"

    # Boiler loss threshold (mirror _check_terminated in td_environment.py).
    boiler_valid = obs.get("boiler_valid_mask")
    boiler_rows = obs.get("boilers")
    if boiler_valid is not None and boiler_rows is not None:
        n_boilers = int(np.asarray(boiler_valid).sum())
        if n_boilers > 0:
            rows = np.asarray(boiler_rows)[:n_boilers]
            alive = int(np.count_nonzero(rows[:, 2] > 0.5))
            lost = n_boilers - alive
            frac = getattr(env.unwrapped.config, "boiler_loss_fraction", 1.0 / 3.0)
            threshold = max(1, int(math.ceil(n_boilers * frac)))
            if lost >= threshold:
                return "boilers_lost"

    char = np.asarray(obs.get("character", [0, 0, 1.0, 0]))
    if char.shape[0] >= 3 and float(char[2]) <= 0.0:
        return "character_died"

    return "terminated"


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def evaluate(cfg: EvalConfig) -> dict:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(cfg.seed)

    model_stem = os.path.splitext(os.path.basename(cfg.model_path))[0]
    run_name = cfg.run_name or f"eval_{model_stem}"
    run_dir = os.path.join(cfg.log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    output_json = cfg.output_json or os.path.join(run_dir, "eval_results.json")

    print(f"[fle.rl.eval] model={cfg.model_path} | "
          f"evolution={cfg.evolution_factor} | group_size={cfg.max_unit_group_size} | "
          f"device={cfg.device} | run_dir={run_dir}")

    env = _make_eval_env(cfg)
    model = MaskablePPO.load(cfg.model_path, device=cfg.device)
    print(f"[fle.rl.eval] loaded model on device={model.device}")

    obs, info = env.reset(seed=cfg.seed)

    # --- accumulators ---
    total_reward = 0.0
    total_kills = 0
    total_turrets_lost = 0
    total_walls_lost = 0
    total_boilers_lost = 0
    invalid_count = 0
    action_counts = {"noop": 0, "pick": 0, "place": 0, "refill": 0, "move": 0}
    last_ticks = 0
    last_wave = 0
    last_coverage = 0.0

    terminated = truncated = hit_cap = False
    step = 0
    t_episode = time.perf_counter()

    while True:
        masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=cfg.deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        total_reward += float(reward)
        total_kills += int(info.get("kills", 0))
        total_turrets_lost += int(info.get("turrets_lost", 0))
        total_walls_lost += int(info.get("walls_lost", 0))
        total_boilers_lost += int(info.get("boilers_lost", 0))
        if info.get("invalid_action"):
            invalid_count += 1
        name = info.get("action_name", "unknown")
        if name in action_counts:
            action_counts[name] += 1
        last_ticks = int(info.get("elapsed_ticks", last_ticks))
        last_wave = int(info.get("wave", last_wave))
        last_coverage = float(info.get("coverage", last_coverage))

        if terminated or truncated:
            break
        if step >= cfg.max_steps:
            hit_cap = True
            break

    wall_time = time.perf_counter() - t_episode
    reason = _end_reason(env, obs, terminated, truncated, hit_cap)

    results = {
        "model_path": cfg.model_path,
        "evolution_factor": cfg.evolution_factor,
        "max_unit_group_size": cfg.max_unit_group_size,
        "deterministic": cfg.deterministic,
        "seed": cfg.seed,
        "steps": step,
        "survival_ticks": last_ticks,
        "wave_reached": last_wave,
        "total_reward": total_reward,
        "kills": total_kills,
        "turrets_lost": total_turrets_lost,
        "walls_lost": total_walls_lost,
        "boilers_lost": total_boilers_lost,
        "invalid_actions": invalid_count,
        "invalid_rate": (invalid_count / step) if step else 0.0,
        "final_coverage": last_coverage,
        "action_counts": action_counts,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "end_reason": reason,
        "wall_time_s": wall_time,
        "config": {f.name: getattr(cfg, f.name) for f in fields(cfg)},
    }

    _print_summary(results)

    with open(output_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[fle.rl.eval] wrote results to {output_json}")

    env.close()
    return results


def _print_summary(r: dict) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  EVALUATION SUMMARY")
    print(sep)
    print(f"  end reason     : {r['end_reason']}  "
          f"(terminated={r['terminated']} truncated={r['truncated']})")
    print(f"  survival ticks : {r['survival_ticks']:,}  ({r['survival_ticks'] / 60.0:.1f}s game time)")
    print(f"  steps          : {r['steps']}")
    print(f"  wave reached   : {r['wave_reached']}")
    print(f"  total reward   : {r['total_reward']:+.3f}")
    print(f"  kills          : {r['kills']}")
    print(f"  losses         : turrets={r['turrets_lost']}  walls={r['walls_lost']}  "
          f"boilers={r['boilers_lost']}")
    print(f"  final coverage : {r['final_coverage']:.1%}")
    print(f"  invalid acts   : {r['invalid_actions']}  ({r['invalid_rate']:.1%} of steps)")
    ac = r["action_counts"]
    print(f"  actions        : noop={ac['noop']}  pick={ac['pick']}  place={ac['place']}  "
          f"refill={ac['refill']}  move={ac['move']}")
    print(f"  wall time      : {r['wall_time_s']:.1f}s")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv=None) -> EvalConfig:
    p = argparse.ArgumentParser(
        description="Evaluate a trained Tower Defense RL model on one episode (no training)"
    )
    p.add_argument("--model", required=True, help="Path to a saved model .zip "
                   "(final_model / best / checkpoint)")
    p.add_argument("--evolution-factor", type=float, default=None,
                   help="Enemy evolution factor 0..1 (default: keep the save's value)")
    p.add_argument("--group-size", type=int, default=None,
                   help="Max biters per attack group (default: engine default)")
    p.add_argument("--save-path", default=None)
    p.add_argument("--run-idx", type=int, default=0, help="Which container to connect to")
    p.add_argument("--game-speed", type=float, default=None,
                   help="Override scenario game speed (e.g. 1.0 to watch, 10.0 = training speed)")
    p.add_argument("--no-deterministic", action="store_true",
                   help="Sample from the policy instead of taking the argmax action")
    p.add_argument("--max-steps", type=int, default=100_000,
                   help="Safety cap on steps before forcing the episode to end")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default="./td_runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-json", default=None,
                   help="Explicit path for the results JSON (default <run_dir>/eval_results.json)")
    a = p.parse_args(argv)
    return EvalConfig(
        model_path=a.model,
        evolution_factor=a.evolution_factor,
        max_unit_group_size=a.group_size,
        save_path=a.save_path,
        run_idx=a.run_idx,
        game_speed=a.game_speed,
        deterministic=not a.no_deterministic,
        max_steps=a.max_steps,
        device=a.device,
        seed=a.seed,
        log_dir=a.log_dir,
        run_name=a.run_name,
        output_json=a.output_json,
    )


def main(argv=None):
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("fle.env.gym_env.td_environment").setLevel(logging.INFO)
    for noisy in ("urllib3", "requests", "docker", "stable_baselines3", "factorio_rcon"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    evaluate(_parse_args(argv))


if __name__ == "__main__":
    main()
