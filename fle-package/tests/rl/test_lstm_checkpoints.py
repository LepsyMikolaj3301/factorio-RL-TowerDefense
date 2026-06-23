"""Checkpoint tests for the custom masked PPO-LSTM trainer."""

import pytest

torch = pytest.importorskip("torch")

from fle.rl.lstm_config import RecurrentTrainConfig
from fle.rl.lstm_policy import ACTION_FACTORS, ACTION_MASK_DIM
from fle.rl.train_lstm import _save_checkpoint


def test_lstm_checkpoint_includes_optimizer_state(tmp_path):
    agent = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    cfg = RecurrentTrainConfig(features_dim=32, cnn_dim=16, set_dim=8, lstm_hidden=16)
    path = tmp_path / "checkpoints" / "td_lstm_128_steps.pt"

    _save_checkpoint(agent, cfg, str(path), 128, optimizer=optimizer)

    ckpt = torch.load(path, map_location="cpu")
    assert ckpt["checkpoint_version"] == 2
    assert ckpt["action_factors"] == list(ACTION_FACTORS)
    assert ckpt["action_mask_dim"] == ACTION_MASK_DIM
    assert ckpt["global_step"] == 128
    assert "model_state" in ckpt
    assert "optimizer_state" in ckpt
    assert ckpt["model_kwargs"] == {
        "features_dim": 32,
        "cnn_dim": 16,
        "set_dim": 8,
        "lstm_hidden": 16,
    }
    assert ckpt["config"]["checkpoint_freq"] == cfg.checkpoint_freq
