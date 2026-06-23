"""Unit tests for the read_td_events tool client."""

from unittest.mock import MagicMock

from fle.env.tools.admin.read_td_events.client import ReadTdEvents


def test_read_td_events_parses_boiler_counter():
    conn = MagicMock()
    conn.rcon_client = MagicMock()
    conn.rcon_client.send_command.return_value = (
        "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    )
    game_state = MagicMock()
    game_state.agent_index = 0

    tool = ReadTdEvents(conn, game_state)
    events = tool()

    assert events["kills"] == 1
    assert events["buildings_lost"] == 4
    assert events["boilers_lost"] == 5
    assert events["radar_lost"] == 6
    assert events["turret_w"] == 15
