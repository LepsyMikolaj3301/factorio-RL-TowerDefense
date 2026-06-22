"""Tower Defense reinforcement learning: the v1 model + real training pipeline.

    from fle.rl import TrainConfig, train

    train(TrainConfig(num_envs=4, total_timesteps=1_000_000))

Or from the CLI: ``python -m fle.rl.train --num-envs 4``.
"""

from fle.rl.config import TrainConfig
from fle.rl.policy import (
    PointerHead,
    TDExtractor,
    make_td_policy_kwargs,
)

__all__ = [
    "TrainConfig",
    "TDExtractor",
    "PointerHead",
    "make_td_policy_kwargs",
    "train",
]


def train(cfg: "TrainConfig"):
    """Run the training pipeline (imports sb3-contrib lazily)."""
    from fle.rl.train import train as _train

    return _train(cfg)
