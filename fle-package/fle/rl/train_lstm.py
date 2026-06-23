"""Masked PPO-LSTM training pipeline for Tower Defense.

This module implements a small PPO loop for a recurrent policy with action
masking. It uses the same environment wrappers and feature extractor as the
feedforward trainer, and writes to a ``*_lstm`` run directory.

    # Start containers first (loads data/saves/tower_defense.zip):
    fle cluster start -n 4

    # Train the recurrent model:
    python -m fle.rl.train_lstm --num-envs 4 --total-timesteps 1000000 --evolution-factor 0.5

Design notes:
  * Rollout buffer stores obs (dict), the 4-factor action, the **flat action
    mask** (so the update reconstructs the identical masked distribution), value,
    logprob, reward, and the per-step ``done`` (episode_start) for hidden reset.
  * Minibatching is **by env** — each env's full length-``n_steps`` rollout is one
    sequence, kept whole, so there is no sequence padding (and thus no
    padding-mask/action-mask conflation).
"""

import argparse
import dataclasses
import os
import time
from dataclasses import asdict, fields
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from fle.rl.lstm_config import RecurrentTrainConfig
from fle.rl.lstm_policy import (
    ACTION_FACTORS,
    ACTION_MASK_DIM,
    RecurrentMaskableActorCritic,
)


def _scenario_config(cfg: RecurrentTrainConfig):
    from fle.env.gym_env.td_config import TDScenarioConfig

    return dataclasses.replace(
        TDScenarioConfig(),
        evolution_factor=cfg.evolution_factor,
        max_unit_group_size=cfg.max_unit_group_size,
        game_speed=cfg.game_speed
        if cfg.game_speed is not None
        else TDScenarioConfig().game_speed,
    )


def _make_env_thunk(run_idx: int, cfg: RecurrentTrainConfig):
    def _init():
        from fle.env.gym_env.registry import make_td_env
        from fle.env.gym_env.td_spaces import FlatTDActionWrapper
        from fle.env.gym_env.action_mask import ActionMaskWrapper

        env = make_td_env(run_idx=run_idx, save_path=cfg.save_path, config=_scenario_config(cfg))
        env = FlatTDActionWrapper(env)      # Dict -> MultiDiscrete
        env = ActionMaskWrapper(env)        # adds masks + action_masks()
        return env

    return _init


def _build_vec_env(cfg: RecurrentTrainConfig, thunks: List):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    if len(thunks) > 1 and cfg.use_subproc:
        venv = SubprocVecEnv(thunks)
    else:
        venv = DummyVecEnv(thunks)
    return VecMonitor(venv)


def _obs_to_tensors(obs: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    """VecEnv dict-of-(N,*) numpy -> dict-of-(N,*) torch on device (dtype kept)."""
    return {k: torch.as_tensor(v, device=device) for k, v in obs.items()}


def _gather_masks(venv, device) -> torch.Tensor:
    """Per-env flat action masks -> (N, ACTION_MASK_DIM) float tensor."""
    masks = np.stack(venv.env_method("action_masks"))
    return torch.as_tensor(masks, dtype=torch.float32, device=device)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _model_kwargs(cfg: RecurrentTrainConfig) -> dict:
    return dict(
        features_dim=cfg.features_dim,
        cnn_dim=cfg.cnn_dim,
        set_dim=cfg.set_dim,
        lstm_hidden=cfg.lstm_hidden,
    )


def _save_checkpoint(
    agent,
    cfg: RecurrentTrainConfig,
    path: str,
    global_step: int,
    optimizer=None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "checkpoint_version": 2,
        "action_factors": list(ACTION_FACTORS),
        "action_mask_dim": ACTION_MASK_DIM,
        "model_state": agent.state_dict(),
        "model_kwargs": _model_kwargs(cfg),
        "global_step": global_step,
        "config": asdict(cfg),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def _assert_checkpoint_action_space(ckpt: dict, path: str) -> None:
    saved_factors = ckpt.get("action_factors")
    saved_dim = ckpt.get("action_mask_dim")
    if saved_factors is None:
        first_head = ckpt.get("model_state", {}).get("actor_heads.0.weight")
        if first_head is not None:
            saved_factors = [int(first_head.shape[0]), *ACTION_FACTORS[1:]]
    if saved_factors is not None and list(saved_factors) != list(ACTION_FACTORS):
        raise ValueError(
            f"Incompatible LSTM checkpoint action factors in {path}: "
            f"checkpoint has {list(saved_factors)}, environment expects "
            f"{list(ACTION_FACTORS)}. Start a new 6-action run or select a "
            "compatible checkpoint."
        )
    if saved_dim is not None and int(saved_dim) != ACTION_MASK_DIM:
        raise ValueError(
            f"Incompatible LSTM checkpoint mask dim in {path}: checkpoint has "
            f"{int(saved_dim)}, environment expects {ACTION_MASK_DIM}."
        )


def train(cfg: RecurrentTrainConfig):
    from torch.utils.tensorboard import SummaryWriter

    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_name = cfg.resolved_run_name()
    run_dir = os.path.join(cfg.log_dir, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    if cfg.checkpoint_freq > 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    wandb_run = None
    if cfg.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project, name=run_name,
            config={f.name: getattr(cfg, f.name) for f in fields(cfg)},
            sync_tensorboard=True,
        )

    N = cfg.num_envs
    T = cfg.n_steps
    venv = _build_vec_env(cfg, [_make_env_thunk(i, cfg) for i in range(N)])
    obs_space = venv.observation_space  # gym Dict (single env)

    agent = RecurrentMaskableActorCritic(obs_space, **_model_kwargs(cfg)).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    start_step = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        _assert_checkpoint_action_space(ckpt, cfg.resume)
        agent.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            print(
                "[fle.rl.train_lstm] resume checkpoint has no optimizer_state; "
                "continuing with a fresh optimizer"
            )
        start_step = int(ckpt.get("global_step", 0))
        print(f"[fle.rl.train_lstm] resumed from {cfg.resume} @ step {start_step}")

    print(
        f"[fle.rl.train_lstm] device={device} | run_dir={run_dir} | "
        f"num_envs={N} | n_steps={T} | total_timesteps={cfg.total_timesteps}"
    )

    obs_buf: Dict[str, torch.Tensor] = {}
    for k, space in obs_space.spaces.items():
        dt = torch.uint8 if space.dtype == np.uint8 else torch.float32
        obs_buf[k] = torch.zeros((T, N, *space.shape), dtype=dt, device=device)
    actions_buf = torch.zeros((T, N, len(agent.FACTORS)), dtype=torch.long, device=device)
    masks_buf = torch.zeros((T, N, ACTION_MASK_DIM), dtype=torch.float32, device=device)
    logprobs_buf = torch.zeros((T, N), device=device)
    rewards_buf = torch.zeros((T, N), device=device)
    dones_buf = torch.zeros((T, N), device=device)
    values_buf = torch.zeros((T, N), device=device)

    next_obs = _obs_to_tensors(venv.reset(), device)
    next_done = torch.zeros(N, device=device)
    next_mask = _gather_masks(venv, device)
    next_lstm_state = agent.initial_state(N, device)

    batch_size = N * T
    envs_per_batch = max(1, N // cfg.num_minibatches)
    num_updates = max(1, (cfg.total_timesteps) // batch_size)
    global_step = start_step
    last_ckpt = global_step
    t_start = time.perf_counter()

    for update in range(1, num_updates + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        # lstm state at the start of this rollout (replayed during the update).
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        ep_returns, ep_lengths, ep_ticks = [], [], []
        threatened_readiness = []
        early_setup_rewards = []
        ammo_taken = []
        action_counts = {
            "noop": 0,
            "pick": 0,
            "place": 0,
            "refill": 0,
            "move": 0,
            "take_ammo": 0,
        }
        invalid_steps = 0

        for t in range(T):
            global_step += N
            for k in obs_buf:
                obs_buf[k][t] = next_obs[k]
            dones_buf[t] = next_done
            masks_buf[t] = next_mask

            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                    next_obs, next_lstm_state, next_done, next_mask
                )
            values_buf[t] = value
            actions_buf[t] = action
            logprobs_buf[t] = logprob

            obs_np, reward, done, infos = venv.step(action.cpu().numpy())
            rewards_buf[t] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_obs = _obs_to_tensors(obs_np, device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)
            next_mask = _gather_masks(venv, device)

            for info in infos:
                if info.get("invalid_action"):
                    invalid_steps += 1
                if "threatened_turret_readiness" in info:
                    threatened_readiness.append(float(info["threatened_turret_readiness"]))
                if "early_setup_reward" in info:
                    early_setup_rewards.append(float(info["early_setup_reward"]))
                if "ammo_taken" in info:
                    ammo_taken.append(float(info["ammo_taken"]))
                action_name = info.get("action_name")
                if action_name in action_counts:
                    action_counts[action_name] += 1
                ep = info.get("episode")
                if ep is not None:
                    ep_returns.append(float(ep["r"]))
                    ep_lengths.append(int(ep["l"]))
                    ep_ticks.append(int(info.get("elapsed_ticks", 0)))

        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(N)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + cfg.gamma * nextvalues * nextnonterminal - values_buf[t]
                lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values_buf

        # Minibatch BY ENV: each env's whole (T,) sequence is replayed intact.
        env_inds = np.arange(N)
        clipfracs = []
        approx_kl = pg_loss = v_loss = entropy_loss = torch.tensor(0.0)
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(env_inds)
            for start in range(0, N, envs_per_batch):
                mb_envs = env_inds[start:start + envs_per_batch]
                B = len(mb_envs)
                mb_env_t = torch.as_tensor(mb_envs, device=device)

                # (T, B, ...) -> T-major flatten (T*B, ...)
                mb_obs = {
                    k: obs_buf[k][:, mb_env_t].reshape((T * B, *v.shape[2:]))
                    for k, v in obs_buf.items()
                }
                mb_dones = dones_buf[:, mb_env_t].reshape(T * B)
                mb_masks = masks_buf[:, mb_env_t].reshape(T * B, ACTION_MASK_DIM)
                mb_actions = actions_buf[:, mb_env_t].reshape(T * B, len(agent.FACTORS))
                mb_logp_old = logprobs_buf[:, mb_env_t].reshape(T * B)
                mb_adv = advantages[:, mb_env_t].reshape(T * B)
                mb_returns = returns[:, mb_env_t].reshape(T * B)
                mb_values_old = values_buf[:, mb_env_t].reshape(T * B)
                mb_init_state = (
                    initial_lstm_state[0][:, mb_env_t],
                    initial_lstm_state[1][:, mb_env_t],
                )

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    mb_obs, mb_init_state, mb_dones, mb_masks, action=mb_actions
                )

                logratio = newlogprob - mb_logp_old
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                adv = mb_adv
                if cfg.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if cfg.clip_vloss:
                    v_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values_old + torch.clamp(
                        newvalue - mb_values_old, -cfg.clip_coef, cfg.clip_coef
                    )
                    v_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        pg_v = float(pg_loss.detach())
        v_v = float(v_loss.detach())
        ent_v = float(entropy_loss.detach())
        kl_v = float(approx_kl.detach())
        sps = int((global_step - start_step) / (time.perf_counter() - t_start))
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/policy_loss", pg_v, global_step)
        writer.add_scalar("losses/value_loss", v_v, global_step)
        writer.add_scalar("losses/entropy", ent_v, global_step)
        writer.add_scalar("losses/approx_kl", kl_v, global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)) if clipfracs else 0.0, global_step)
        writer.add_scalar("td/invalid_rate", invalid_steps / (T * N), global_step)
        if threatened_readiness:
            writer.add_scalar(
                "td/threatened_turret_readiness",
                float(np.mean(threatened_readiness)),
                global_step,
            )
        if early_setup_rewards:
            writer.add_scalar(
                "td/early_setup_reward",
                float(np.mean(early_setup_rewards)),
                global_step,
            )
        if ammo_taken:
            writer.add_scalar("td/ammo_taken", float(np.mean(ammo_taken)), global_step)
        for action_name, count in action_counts.items():
            writer.add_scalar(
                f"td/action_{action_name}",
                count / float(T * N),
                global_step,
            )
        if ep_returns:
            writer.add_scalar("charts/episodic_return", float(np.mean(ep_returns)), global_step)
            writer.add_scalar("charts/episodic_length", float(np.mean(ep_lengths)), global_step)
            writer.add_scalar("td/survival_ticks", float(np.mean(ep_ticks)), global_step)
        if device.type == "cuda":
            writer.add_scalar("gpu/allocated_mb", torch.cuda.memory_allocated(device) / 1e6, global_step)

        ret_str = f" | ret {np.mean(ep_returns):+.2f}" if ep_returns else ""
        print(
            f"[fle.rl.train_lstm] update {update}/{num_updates} | step {global_step} | "
            f"SPS {sps} | pg {pg_v:+.3f} | v {v_v:.3f} | "
            f"ent {ent_v:.3f} | kl {kl_v:.4f}{ret_str}"
        )

        if cfg.checkpoint_freq > 0 and (global_step - last_ckpt) >= cfg.checkpoint_freq:
            ckpt_path = os.path.join(checkpoint_dir, f"td_lstm_{global_step}_steps.pt")
            _save_checkpoint(agent, cfg, ckpt_path, global_step, optimizer=optimizer)
            last_ckpt = global_step
            print(f"[fle.rl.train_lstm] saved checkpoint to {ckpt_path}")

    final_path = os.path.join(run_dir, "final_model.pt")
    _save_checkpoint(agent, cfg, final_path, global_step, optimizer=optimizer)
    print(f"[fle.rl.train_lstm] saved final model to {final_path}")

    venv.close()
    writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    return agent


def _parse_args(argv=None) -> RecurrentTrainConfig:
    p = argparse.ArgumentParser(description="Train the Tower Defense masked PPO-LSTM agent (v2 model)")
    p.add_argument("--evolution-factor", type=float, default=None)
    p.add_argument("--group-size", type=int, default=None)
    p.add_argument("--game-speed", type=float, default=None,
                   help="Factorio game speed during training; lower values are easier to watch")
    p.add_argument("--save-path", default=None)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--no-subproc", action="store_true", help="Force DummyVecEnv")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--n-steps", type=int, default=128, help="rollout / BPTT sequence length")
    p.add_argument("--num-minibatches", type=int, default=2)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--lstm-hidden", type=int, default=256)
    p.add_argument("--features-dim", type=int, default=256)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default="./td_runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--debug-env", choices=["", "info", "verbose"], default="")
    a = p.parse_args(argv)
    return RecurrentTrainConfig(
        evolution_factor=a.evolution_factor,
        max_unit_group_size=a.group_size,
        game_speed=a.game_speed,
        save_path=a.save_path,
        num_envs=a.num_envs,
        use_subproc=not a.no_subproc,
        total_timesteps=a.total_timesteps,
        learning_rate=a.lr,
        anneal_lr=not a.no_anneal_lr,
        n_steps=a.n_steps,
        num_minibatches=a.num_minibatches,
        update_epochs=a.update_epochs,
        gamma=a.gamma,
        gae_lambda=a.gae_lambda,
        ent_coef=a.ent_coef,
        target_kl=a.target_kl,
        lstm_hidden=a.lstm_hidden,
        features_dim=a.features_dim,
        device=a.device,
        seed=a.seed,
        log_dir=a.log_dir,
        run_name=a.run_name,
        checkpoint_freq=a.checkpoint_freq,
        use_wandb=a.wandb,
        resume=a.resume,
        debug_env=a.debug_env,
    )


def main(argv=None):
    cfg = _parse_args(argv)
    if cfg.debug_env:
        import logging

        level = logging.DEBUG if cfg.debug_env == "verbose" else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s.%(msecs)03d  %(levelname)-7s  %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger("fle.env.gym_env.td_environment").setLevel(level)
        for noisy in ("urllib3", "requests", "docker", "stable_baselines3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    train(cfg)


if __name__ == "__main__":
    main()
