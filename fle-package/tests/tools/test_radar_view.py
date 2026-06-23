"""Unit tests for RadarView tool client - base64 decode logic."""

import base64
import numpy as np
import pytest
from unittest.mock import MagicMock

from fle.env.tools.admin.radar_view.client import RadarView, NUM_CHANNELS


class TestRadarViewDecode:
    """Test RadarView base64 → numpy decoding."""

    def _make_tool(self):
        conn = MagicMock()
        conn.rcon_client = MagicMock()
        game_state = MagicMock()
        return RadarView(conn, game_state)

    def test_valid_b64_decode(self):
        """Test decoding a valid base64-encoded grid."""
        tool = self._make_tool()
        radius = 4
        cell_size = 1.0
        grid_size = int(2 * radius / cell_size)  # 8

        # Create a known array
        arr = np.arange(NUM_CHANNELS * grid_size * grid_size, dtype=np.uint8)
        encoded = "b64:" + base64.b64encode(arr.tobytes()).decode()

        tool.connection.rcon_client.send_command = MagicMock(return_value=encoded)

        result = tool(center_x=0, center_y=0, radius=radius, cell_size=cell_size)

        assert result.shape == (NUM_CHANNELS, grid_size, grid_size)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(
            result.flatten(), arr
        )

    def test_wrong_size_returns_zeros(self):
        """Test that wrong-sized data returns zeros."""
        tool = self._make_tool()
        radius = 4
        cell_size = 1.0
        grid_size = int(2 * radius / cell_size)

        # Wrong size data
        bad_arr = np.zeros(10, dtype=np.uint8)
        encoded = "b64:" + base64.b64encode(bad_arr.tobytes()).decode()

        tool.connection.rcon_client.send_command = MagicMock(return_value=encoded)

        result = tool(center_x=0, center_y=0, radius=radius, cell_size=cell_size)
        assert result.shape == (NUM_CHANNELS, grid_size, grid_size)
        assert np.all(result == 0)

    def test_non_b64_response_returns_zeros(self):
        """Test that non-base64 response returns zeros."""
        tool = self._make_tool()
        radius = 4

        tool.connection.rcon_client.send_command = MagicMock(return_value="some error")

        result = tool(center_x=0, center_y=0, radius=radius)
        grid_size = int(2 * radius / 1.0)
        assert result.shape == (NUM_CHANNELS, grid_size, grid_size)
        assert np.all(result == 0)

    def test_channel_count(self):
        assert NUM_CHANNELS == 8
