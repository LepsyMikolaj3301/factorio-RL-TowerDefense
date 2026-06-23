"""Training configuration for the Tower Defense RL pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """All knobs for `fle.rl.train`. Sensible defaults for a first real run."""

    # --- Environment / scenario ---
    # Difficulty is driven by two engine knobs applied on world init + every
    # reset (None = keep the save's value). See TDScenarioConfig.
    evolution_factor: Optional[float] = None   # enemy evolution (0..1)
    max_unit_group_size: Optional[int] = None  # cap on biters per attack group
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

    # --- Debug ---
    # "" = off  |  "info" = one line per step  |  "verbose" = per-call timings
    debug_env: str = ""


@dataclass
class EvalConfig:
    """All knobs for `fle.rl.eval`. Runs one episode of a trained model (no learning)."""

    model_path: str                      # required: path to a saved .zip (final_model/best/checkpoint)
    # Difficulty knobs applied on world init + every reset (None = save default).
    evolution_factor: Optional[float] = None   # enemy evolution (0..1)
    max_unit_group_size: Optional[int] = None  # cap on biters per attack group
    save_path: Optional[str] = None      # prebuilt Factorio save (else container default)
    run_idx: int = 0                     # which container to connect to
    game_speed: Optional[float] = None   # override scenario game_speed (e.g. 1.0 to watch); None = config default
    deterministic: bool = True           # argmax over masked logits vs sample
    max_steps: int = 100_000             # safety cap so a stuck episode can't run forever
    device: str = "auto"                 # "auto" -> CUDA if available, else CPU
    seed: int = 0

    # --- Output ---
    log_dir: str = "./td_runs"
    run_name: Optional[str] = None       # output dir name; default derived from model name
    output_json: Optional[str] = None    # explicit path; default <run_dir>/eval_results.json
