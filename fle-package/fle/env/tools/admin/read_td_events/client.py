from fle.env.tools import Tool


# Order must match server.lua's CSV output.
EVENT_KEYS = [
    "kills",
    "turrets_lost",
    "walls_lost",
    "buildings_lost",
    "boilers_lost",
    "radar_lost",
    "char_died",
    "wall_n",
    "wall_e",
    "wall_s",
    "wall_w",
    "turret_n",
    "turret_e",
    "turret_s",
    "turret_w",
]


class ReadTdEvents(Tool):
    def __init__(self, connection, game_state):
        super().__init__(connection, game_state)

    def __call__(self) -> dict:
        """
        Read and reset the tower-defense reward event counters accumulated
        server-side since the last call (one decision window's worth of deaths).

        :return: dict with death/loss counters including boilers_lost.
        """
        # Returns a small CSV string; read it directly over RCON (mirrors
        # radar_view) so we never depend on the pcall+dump table round-trip.
        cmd = (
            "/silent-command rcon.print(storage.actions.read_td_events("
            f"{self.player_index}))"
        )
        resp = self.connection.rcon_client.send_command(cmd)
        out = {k: 0 for k in EVENT_KEYS}
        if not isinstance(resp, str):
            return out
        parts = resp.strip().split(",")
        for key, value in zip(EVENT_KEYS, parts):
            try:
                out[key] = int(float(value))
            except (ValueError, TypeError):
                out[key] = 0
        return out
