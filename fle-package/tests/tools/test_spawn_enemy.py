"""Unit tests for SpawnEnemy tool client."""

import pytest
from unittest.mock import MagicMock, patch

from fle.env.tools.admin.spawn_enemy.client import SpawnEnemy


class TestSpawnEnemyClient:
    """Test SpawnEnemy tool construction and parameter validation."""

    def _make_tool(self):
        conn = MagicMock()
        game_state = MagicMock()
        tool = SpawnEnemy(conn, game_state)
        return tool

    def test_instantiation(self):
        tool = self._make_tool()
        assert isinstance(tool, SpawnEnemy)

    def test_default_params(self):
        """Verify default parameters match expected values."""
        import inspect

        sig = inspect.signature(SpawnEnemy.__call__)
        params = sig.parameters

        assert params["entity_type"].default == "small-biter"
        assert params["x"].default == 0.0
        assert params["y"].default == 0.0
        assert params["count"].default == 1

    def test_call_delegates_to_execute(self):
        tool = self._make_tool()
        tool.execute = MagicMock(return_value=("5", None))
        tool.clean_response = MagicMock(return_value=5)

        result = tool(entity_type="medium-biter", x=10.0, y=-5.0, count=5)

        tool.execute.assert_called_once()
        args = tool.execute.call_args[0]
        assert "medium-biter" in args
        assert 10.0 in args
        assert -5.0 in args
        assert 5 in args
        assert result == 5
