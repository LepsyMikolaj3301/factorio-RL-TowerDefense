"""Training configuration for the Tower Defense RL pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """All knobs for `fle.rl.train`. Sensible defaults for a first real run."""

    # --- Environment / scenario ---
    difficulty: str = "medium"          # easy | medium | hard
    save_path: Optional[str] = None     # prebuilt Factorio save (else container default)
    num_envs: int = 1                   # parallel containers
    use_subproc: bool = True            # SubprocVecEnv when num_envs > 1
    frame_stack: int = 1                # >1 stacks recent obs (Box keys only)

    # --- PPO hyperparameters ---
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 1024                 # rollout length per env
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.999               # long horizon (survival)
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # --- Model (TDExtractor) ---
    features_dim: int = 256
    cnn_dim: int = 128
    set_dim: int = 64

    # --- Infra ---
    device: str = "auto"               # "auto" -> CUDA if available, else CPU
    seed: int = 0

    # --- Logging / checkpoints ---
    log_dir: str = "./td_runs"
    run_name: Optional[str] = None
    checkpoint_freq: int = 50_000      # env steps between checkpoints (0 disables)
    eval_freq: int = 0                 # env steps between evals (0 disables; needs a free container)
    n_eval_episodes: int = 3
    use_wandb: bool = False
    wandb_project: str = "factorio-td"
    resume: Optional[str] = None       # path to a saved model .zip to resume from
