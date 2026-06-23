"""Tower Defense RL training pipeline.

Canonical entry point:

    # Start containers first (loads data/saves/tower_defense.zip):
    fle cluster start -n 4

    # Train:
    python -m fle.rl.train --num-envs 4 --total-timesteps 1000000 --evolution-factor 0.5

Requires the RL extra (``pip install 'fle[rl]'``) which provides
``stable-baselines3`` and ``sb3-contrib`` (MaskablePPO). Uses the v1 model
(`fle.rl.policy.TDExtractor`) with reach/transit-aware action masking.
"""

import argparse
import os
from dataclasses import fields
from typing import List

from fle.rl.config import TrainConfig


def _scenario_config(cfg):
    """Build the TD scenario config from the difficulty knobs (evolution factor
    + biter group size). Starts from the base config; None leaves save defaults."""
    import dataclasses

    from fle.env.gym_env.td_config import TDScenarioConfig

    return dataclasses.replace(
        TDScenarioConfig(),
        evolution_factor=cfg.evolution_factor,
        max_unit_group_size=cfg.max_unit_group_size,
        game_speed=cfg.game_speed
        if cfg.game_speed is not None
        else TDScenarioConfig().game_speed,
    )


def _make_env_thunk(run_idx: int, cfg: TrainConfig):
    """Return a thunk that builds one fully-wrapped, maskable TD env."""

    def _init():
        from fle.env.gym_env.registry import make_td_env
        from fle.env.gym_env.td_spaces import FlatTDActionWrapper
        from fle.env.gym_env.action_mask import ActionMaskWrapper

        env = make_td_env(
            run_idx=run_idx,
            save_path=cfg.save_path,
            config=_scenario_config(cfg),
        )
        env = FlatTDActionWrapper(env)      # Dict -> MultiDiscrete
        env = ActionMaskWrapper(env)        # adds masks + action_masks()
        return env

    return _init


def _build_vec_env(cfg: TrainConfig, thunks: List):
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecMonitor,
    )

    if len(thunks) > 1 and cfg.use_subproc:
        venv = SubprocVecEnv(thunks)
    else:
        venv = DummyVecEnv(thunks)
    venv = VecMonitor(venv)
    if cfg.frame_stack and cfg.frame_stack > 1:
        from stable_baselines3.common.vec_env import VecFrameStack

        venv = VecFrameStack(venv, n_stack=cfg.frame_stack)
    return venv


def _make_metrics_callback():
    from stable_baselines3.common.callbacks import BaseCallback

    class TDMetricsCallback(BaseCallback):
        """Log TD-specific signals (kills, losses, actions, GPU memory)."""

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for key in (
                "kills",
                "turrets_lost",
                "walls_lost",
                "buildings_lost",
                "walls_lost_step",
                "turrets_lost_step",
                "walls_lost_episode",
                "turrets_lost_episode",
                "coverage",
                "threatened_turret_readiness",
                "early_setup_reward",
                "ammo_taken",
                "invalid_action",
                "wall_n",
                "wall_e",
                "wall_s",
                "wall_w",
                "turret_n",
                "turret_e",
                "turret_s",
                "turret_w",
            ):
                vals = [i[key] for i in infos if key in i]
                if vals:
                    self.logger.record_mean(f"td/{key}", float(sum(vals) / len(vals)))
            for action in ("noop", "pick", "place", "refill", "move", "take_ammo"):
                vals = [1.0 if i.get("action_name") == action else 0.0 for i in infos]
                if vals:
                    self.logger.record_mean(f"td/action_{action}", float(sum(vals) / len(vals)))
            self._record_gpu_memory()
            return True

        def _record_gpu_memory(self) -> None:
            try:
                import torch

                device = getattr(self.model, "device", None)
                if device is None or getattr(device, "type", None) != "cuda":
                    return
                idx = device.index if device.index is not None else torch.cuda.current_device()
                mb = 1024.0 * 1024.0
                self.logger.record("gpu/allocated_mb", torch.cuda.memory_allocated(idx) / mb)
                self.logger.record("gpu/reserved_mb", torch.cuda.memory_reserved(idx) / mb)
                self.logger.record(
                    "gpu/max_allocated_mb", torch.cuda.max_memory_allocated(idx) / mb
                )
            except Exception:
                return

    return TDMetricsCallback()


def _print_device_summary(model, requested_device: str) -> None:
    import torch

    resolved = model.device
    cuda_available = torch.cuda.is_available()
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else "n/a"
    try:
        first_param_device = next(model.policy.parameters()).device
    except StopIteration:
        first_param_device = "n/a"
    print(
        "[fle.rl] requested_device="
        f"{requested_device} | resolved_device={resolved} | "
        f"torch_cuda_available={cuda_available} | cuda_device={cuda_name} | "
        f"policy_param_device={first_param_device}"
    )


def train(cfg: TrainConfig):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.utils import set_random_seed

    from fle.env.gym_env.td_spaces import NUM_ACTION_TYPES
    from fle.rl.policy import make_td_policy_kwargs

    set_random_seed(cfg.seed)
    run_name = cfg.run_name or f"td_{cfg.num_envs}env"
    run_dir = os.path.join(cfg.log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    venv = _build_vec_env(cfg, [_make_env_thunk(i, cfg) for i in range(cfg.num_envs)])

    callbacks = [_make_metrics_callback()]
    if cfg.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, cfg.checkpoint_freq // cfg.num_envs),
                save_path=os.path.join(run_dir, "checkpoints"),
                name_prefix="td",
            )
        )
    if cfg.eval_freq > 0:
        # Eval needs its own free container at run_idx == num_envs.
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

        eval_env = _build_vec_env(cfg, [_make_env_thunk(cfg.num_envs, cfg)])
        callbacks.append(
            MaskableEvalCallback(
                eval_env,
                best_model_save_path=os.path.join(run_dir, "best"),
                eval_freq=max(1, cfg.eval_freq // cfg.num_envs),
                n_eval_episodes=cfg.n_eval_episodes,
                deterministic=True,
            )
        )

    wandb_run = None
    if cfg.use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb_run = wandb.init(
            project=cfg.wandb_project, name=run_name,
            config={f.name: getattr(cfg, f.name) for f in fields(cfg)},
            sync_tensorboard=True,
        )
        callbacks.append(WandbCallback())

    try:
        import tensorboard  # noqa: F401
        tb_log = run_dir
    except ImportError:
        print("[fle.rl] tensorboard not installed — TensorBoard logging disabled "
              "(pip install tensorboard to enable)")
        tb_log = None

    if cfg.resume:
        model = MaskablePPO.load(cfg.resume, env=venv, device=cfg.device)
        saved_nvec = getattr(getattr(model, "action_space", None), "nvec", None)
        if saved_nvec is not None and int(saved_nvec[0]) != NUM_ACTION_TYPES:
            raise ValueError(
                f"Incompatible TD checkpoint action space: model has "
                f"{int(saved_nvec[0])} action types, environment expects "
                f"{NUM_ACTION_TYPES}. Start a new 6-action run or select a "
                "compatible checkpoint."
            )
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            venv,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            policy_kwargs=make_td_policy_kwargs(
                features_dim=cfg.features_dim, cnn_dim=cfg.cnn_dim, set_dim=cfg.set_dim
            ),
            tensorboard_log=tb_log,
            device=cfg.device,
            seed=cfg.seed,
            verbose=1,
        )

    _print_device_summary(model, cfg.device)
    print(f"[fle.rl] device={model.device} | run_dir={run_dir} | "
          f"num_envs={cfg.num_envs} | total_timesteps={cfg.total_timesteps}")
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList(callbacks),
        reset_num_timesteps=not bool(cfg.resume),
    )

    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)
    print(f"[fle.rl] saved final model to {final_path}.zip")
    venv.close()
    if wandb_run is not None:
        wandb_run.finish()
    return model


def _parse_args(argv=None) -> TrainConfig:
    p = argparse.ArgumentParser(description="Train the Tower Defense RL agent (v1 model)")
    p.add_argument("--evolution-factor", type=float, default=None,
                   help="Enemy evolution factor 0..1 (default: keep the save's value)")
    p.add_argument("--group-size", type=int, default=None,
                   help="Max biters per attack group (default: engine default)")
    p.add_argument("--game-speed", type=float, default=None,
                   help="Factorio game speed during training; lower values are easier to watch")
    p.add_argument("--save-path", default=None)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--no-subproc", action="store_true", help="Force DummyVecEnv")
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda (use cuda to force GPU)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default="./td_runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--debug-env", choices=["", "info", "verbose"], default="",
                   help="info=one-line per step, verbose=per-call timing")
    a = p.parse_args(argv)
    return TrainConfig(
        evolution_factor=a.evolution_factor,
        max_unit_group_size=a.group_size,
        game_speed=a.game_speed,
        save_path=a.save_path,
        num_envs=a.num_envs,
        use_subproc=not a.no_subproc,
        frame_stack=a.frame_stack,
        total_timesteps=a.total_timesteps,
        learning_rate=a.lr,
        n_steps=a.n_steps,
        batch_size=a.batch_size,
        device=a.device,
        seed=a.seed,
        log_dir=a.log_dir,
        run_name=a.run_name,
        checkpoint_freq=a.checkpoint_freq,
        eval_freq=a.eval_freq,
        use_wandb=a.wandb,
        resume=a.resume,
        debug_env=a.debug_env,
    )


def main(argv=None):
    cfg = _parse_args(argv)
    if cfg.debug_env:
        import logging
        # Show INFO from the env by default; DEBUG shows per-call timing.
        level = logging.DEBUG if cfg.debug_env == "verbose" else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s.%(msecs)03d  %(levelname)-7s  %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger("fle.env.gym_env.td_environment").setLevel(level)
        # Suppress noise from other libs
        for noisy in ("urllib3", "requests", "docker", "stable_baselines3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    train(cfg)


if __name__ == "__main__":
    main()
