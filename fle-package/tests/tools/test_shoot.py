"""Unit tests for Shoot tool client."""

import inspect
import pytest
from unittest.mock import MagicMock

from fle.env.tools.agent.shoot.client import Shoot


class TestShootClient:
    def _make_tool(self):
        conn = MagicMock()
        game_state = MagicMock()
        return Shoot(conn, game_state)

    def test_instantiation(self):
        tool = self._make_tool()
        assert isinstance(tool, Shoot)

    def test_required_params(self):
        """target_x and target_y are required, duration_ticks has default."""
        sig = inspect.signature(Shoot.__call__)
        params = sig.parameters
        assert params["target_x"].default is inspect.Parameter.empty
        assert params["target_y"].default is inspect.Parameter.empty
        assert params["duration_ticks"].default == 30

    def test_call_delegates_to_execute(self):
        tool = self._make_tool()
        tool.execute = MagicMock(return_value=("ok", None))
        tool.clean_response = MagicMock(return_value="ok")

        result = tool(target_x=5.0, target_y=-3.0, duration_ticks=60)

        tool.execute.assert_called_once()
        args = tool.execute.call_args[0]
        assert 5.0 in args
        assert -3.0 in args
        assert 60 in args
        assert result == "ok"
