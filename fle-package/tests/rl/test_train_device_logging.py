"""Tests for RL training device visibility helpers."""

import pytest

torch = pytest.importorskip("torch")

from fle.rl.train import _print_device_summary


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class _Model:
    def __init__(self):
        self.device = torch.device("cpu")
        self.policy = _Policy()


def test_print_device_summary_reports_resolved_device(capsys):
    _print_device_summary(_Model(), requested_device="auto")

    out = capsys.readouterr().out
    assert "requested_device=auto" in out
    assert "resolved_device=cpu" in out
    assert "torch_cuda_available=" in out
    assert "policy_param_device=cpu" in out
