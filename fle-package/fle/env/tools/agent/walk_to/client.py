from typing import List

from fle.env.entities import Position
from fle.env.tools import Tool


class WalkTo(Tool):
    def __init__(self, connection, game_state):
        super().__init__(connection, game_state)

    def __call__(
        self,
        waypoints: List[Position],
        target_anchor: int = -1,
        speed: float = 0.2,
    ) -> int:
        """
        Start an asynchronous walk along `waypoints`. The scenario's on_tick
        walker moves the agent character along the polyline at `speed` tiles/tick
        while the game runs; the env polls is_moving via storage.td_walk.active.

        :param waypoints: ordered list of Positions (an A* path).
        :param target_anchor: anchor index this walk is heading to (for obs).
        :param speed: tiles advanced per game tick.
        :return: number of waypoints accepted.
        """
        coords = ";".join(
            f"{float(w.x):.3f},{float(w.y):.3f}" for w in waypoints
        )
        cmd = (
            "/silent-command rcon.print(storage.actions.walk_to("
            f"{self.player_index}, \"{coords}\", {int(target_anchor)}, "
            f"{float(speed)}))"
        )
        resp = self.connection.rcon_client.send_command(cmd)
        try:
            return int(float(resp))
        except (ValueError, TypeError):
            return 0
