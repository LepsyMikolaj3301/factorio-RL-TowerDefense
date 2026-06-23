"""Masked PPO-LSTM evaluation pipeline (inference, no training) — v2 model.

Recurrent sibling of :mod:`fle.rl.eval`. Loads a ``.pt`` checkpoint saved by
``fle.rl.train_lstm``, runs one episode on a single env carrying the LSTM hidden
state across steps and resetting it on episode boundaries.

    # Start a container first:
    fle cluster start -n 1

    # Evaluate:
    python -m fle.rl.eval_lstm --model td_runs/<run>_lstm/final_model.pt --evolution-factor 0.5

    # Slow the game down to watch it play:
    python -m fle.rl.eval_lstm --model td_runs/<run>_lstm/final_model.pt --game-speed 1.0
"""

import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import fields

import numpy as np
import torch

from fle.rl.lstm_config import RecurrentEvalConfig
from fle.rl.lstm_policy import ACTION_FACTORS, ACTION_MASK_DIM, build_agent


def _scenario_config(cfg: RecurrentEvalConfig):
    from fle.env.gym_env.td_config import TDScenarioConfig

    scenario = dataclasses.replace(
        TDScenarioConfig(),
        evolution_factor=cfg.evolution_factor,
        max_unit_group_size=cfg.max_unit_group_size,
    )
    if cfg.game_speed is not None:
        scenario = dataclasses.replace(scenario, game_speed=cfg.game_speed)
    return scenario


def _make_eval_env(cfg: RecurrentEvalConfig):
    from fle.env.gym_env.registry import make_td_env
    from fle.env.gym_env.td_spaces import FlatTDActionWrapper
    from fle.env.gym_env.action_mask import ActionMaskWrapper

    env = make_td_env(run_idx=cfg.run_idx, save_path=cfg.save_path, config=_scenario_config(cfg))
    env = FlatTDActionWrapper(env)
    env = ActionMaskWrapper(env)
    return env


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _obs_to_tensors(obs: dict, device) -> dict:
    """Single-env obs dict -> batched (1, *) tensors on device."""
    return {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}


def _assert_checkpoint_action_space(ckpt: dict, path: str) -> None:
    saved_factors = ckpt.get("action_factors")
    saved_dim = ckpt.get("action_mask_dim")
    if saved_factors is None:
        first_head = ckpt.get("model_state", {}).get("actor_heads.0.weight")
        if first_head is not None:
            saved_factors = [int(first_head.shape[0]), *ACTION_FACTORS[1:]]
    if saved_factors is not None and list(saved_factors) != list(ACTION_FACTORS):
        raise ValueError(
            f"Incompatible LSTM checkpoint action factors in {path}: checkpoint "
            f"has {list(saved_factors)}, environment expects {list(ACTION_FACTORS)}. "
            "Start a new 6-action run or select a compatible checkpoint."
        )
    if saved_dim is not None and int(saved_dim) != ACTION_MASK_DIM:
        raise ValueError(
            f"Incompatible LSTM checkpoint mask dim in {path}: checkpoint has "
            f"{int(saved_dim)}, environment expects {ACTION_MASK_DIM}."
        )


def _end_reason(env, obs: dict, terminated: bool, truncated: bool, hit_cap: bool) -> str:
    if truncated:
        return "time_limit"
    if hit_cap and not terminated:
        return "max_steps"
    if not terminated:
        return "incomplete"

    radar = np.asarray(obs.get("radar", [0, 0, 1.0]))
    if radar.shape[0] >= 3 and float(radar[2]) <= 0.0:
        return "radar_destroyed"

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


def evaluate(cfg: RecurrentEvalConfig) -> dict:
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model_stem = os.path.splitext(os.path.basename(cfg.model_path))[0]
    run_name = cfg.run_name or f"eval_{model_stem}_lstm"
    run_dir = os.path.join(cfg.log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    output_json = cfg.output_json or os.path.join(run_dir, "eval_results.json")

    print(f"[fle.rl.eval_lstm] model={cfg.model_path} | evolution={cfg.evolution_factor} | "
          f"group_size={cfg.max_unit_group_size} | device={device} | run_dir={run_dir}")

    env = _make_eval_env(cfg)

    ckpt = torch.load(cfg.model_path, map_location=device)
    _assert_checkpoint_action_space(ckpt, cfg.model_path)
    agent = build_agent(env.observation_space, ckpt["model_kwargs"]).to(device)
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()
    print(f"[fle.rl.eval_lstm] loaded model (step {ckpt.get('global_step', '?')}) on {device}")

    obs, info = env.reset(seed=cfg.seed)
    next_obs = _obs_to_tensors(obs, device)
    lstm_state = agent.initial_state(1, device)
    done_t = torch.zeros(1, device=device)

    total_reward = 0.0
    total_kills = total_turrets_lost = total_walls_lost = total_boilers_lost = 0
    invalid_count = 0
    action_counts = {
        "noop": 0,
        "pick": 0,
        "place": 0,
        "refill": 0,
        "move": 0,
        "take_ammo": 0,
    }
    last_ticks = last_wave = 0
    last_coverage = 0.0

    terminated = truncated = hit_cap = False
    step = 0
    t_episode = time.perf_counter()

    while True:
        mask = torch.as_tensor(
            env.action_masks(), dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _, lstm_state = agent.get_action_and_value(
                next_obs, lstm_state, done_t, mask, deterministic=cfg.deterministic
            )
        action_np = action.squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action_np)
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

        next_obs = _obs_to_tensors(obs, device)
        done_t = torch.as_tensor(
            [1.0 if (terminated or truncated) else 0.0], device=device
        )  # resets hidden on the next forward at an episode boundary

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
    print(f"[fle.rl.eval_lstm] wrote results to {output_json}")

    env.close()
    return results


def _print_summary(r: dict) -> None:
    sep = "─" * 60
    print(f"\n{sep}\n  EVALUATION SUMMARY (LSTM)\n{sep}")
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
          f"refill={ac['refill']}  move={ac['move']}  take_ammo={ac['take_ammo']}")
    print(f"  wall time      : {r['wall_time_s']:.1f}s\n{sep}")


def _parse_args(argv=None) -> RecurrentEvalConfig:
    p = argparse.ArgumentParser(
        description="Evaluate a trained Tower Defense masked PPO-LSTM model (no training)"
    )
    p.add_argument("--model", required=True, help="Path to a saved .pt checkpoint")
    p.add_argument("--evolution-factor", type=float, default=None)
    p.add_argument("--group-size", type=int, default=None)
    p.add_argument("--save-path", default=None)
    p.add_argument("--run-idx", type=int, default=0)
    p.add_argument("--game-speed", type=float, default=None)
    p.add_argument("--no-deterministic", action="store_true",
                   help="Sample from the policy instead of taking the argmax action")
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default="./td_runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-json", default=None)
    a = p.parse_args(argv)
    return RecurrentEvalConfig(
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
