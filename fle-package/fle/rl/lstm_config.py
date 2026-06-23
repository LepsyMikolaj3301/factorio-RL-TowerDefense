"""Training/eval configuration for the *recurrent* Tower Defense RL pipeline.

This is the v2 (masked PPO-LSTM) sibling of :mod:`fle.rl.config`. It is a
**separate** dataclass so the feedforward v1 config/pipeline is left completely
untouched — the recurrent model is a side-by-side variant, never an overwrite.

Hyperparameter defaults are tuned for a slow, throughput-bound env (RCON
round-trips dominate, not GPU): short full-sequence rollouts, env-wise
minibatching, and fewer update epochs than the feedforward baseline. VRAM is not
the binding constraint, so ``n_steps`` / ``lstm_hidden`` are sized for credit
assignment and capacity rather than to fit a memory budget.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RecurrentTrainConfig:
    """All knobs for ``fle.rl.train_lstm`` (masked PPO-LSTM, v2 model)."""

    # --- Environment / scenario (mirrors TrainConfig) ---
    evolution_factor: Optional[float] = None   # enemy evolution (0..1); None keeps save value
    max_unit_group_size: Optional[int] = None  # cap on biters per attack group
    game_speed: Optional[float] = None          # override scenario game_speed; None keeps default
    save_path: Optional[str] = None     # prebuilt Factorio save (else container default)
    num_envs: int = 4                   # parallel containers (throughput-bound)
    use_subproc: bool = True            # SubprocVecEnv when num_envs > 1

    # --- PPO / rollout ---
    total_timesteps: int = 1_000_000
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True              # linear decay of lr -> 0 over training
    n_steps: int = 128                  # rollout / BPTT sequence length per env
    num_minibatches: int = 2            # env-wise minibatches (sequences kept whole)
    update_epochs: int = 4              # recurrent overfits faster than feedforward's 10
    gamma: float = 0.997               # long survival horizon; slightly below 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    norm_adv: bool = True
    ent_coef: float = 0.01             # entropy is over *valid* actions only (masking)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None  # early-stop an update if approx_kl exceeds this

    # --- Model (TDExtractor + LSTM) ---
    features_dim: int = 256            # TDExtractor output (LSTM input size)
    cnn_dim: int = 128
    set_dim: int = 64
    lstm_hidden: int = 256            # LSTM hidden/cell size (can grow; VRAM is not the limit)

    # --- Infra ---
    device: str = "auto"             # "auto" -> CUDA if available, else CPU
    seed: int = 0

    # --- Logging / checkpoints ---
    log_dir: str = "./td_runs"
    run_name: Optional[str] = None    # output dir name; default derived below
    checkpoint_freq: int = 50_000     # env steps between checkpoints (0 disables)
    use_wandb: bool = False
    wandb_project: str = "factorio-td"
    resume: Optional[str] = None      # path to a saved .pt checkpoint to resume from

    # --- Debug ---
    debug_env: str = ""               # "" | "info" | "verbose"

    def resolved_run_name(self) -> str:
        # Always suffix with ``_lstm`` so a recurrent run can never collide with a
        # feedforward run dir / checkpoint.
        base = self.run_name or f"td_{self.num_envs}env"
        return base if base.endswith("_lstm") else f"{base}_lstm"


@dataclass
class RecurrentEvalConfig:
    """All knobs for ``fle.rl.eval_lstm`` (one inference episode, no learning)."""

    model_path: str                      # required: path to a saved .pt checkpoint
    evolution_factor: Optional[float] = None
    max_unit_group_size: Optional[int] = None
    save_path: Optional[str] = None
    run_idx: int = 0
    game_speed: Optional[float] = None
    deterministic: bool = True           # argmax over masked logits vs sample
    max_steps: int = 100_000
    device: str = "auto"
    seed: int = 0

    # --- Output ---
    log_dir: str = "./td_runs"
    run_name: Optional[str] = None
    output_json: Optional[str] = None
