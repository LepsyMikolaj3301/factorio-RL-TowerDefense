from fle.env.tools import Tool
from fle.env.entities import Position


class Shoot(Tool):
    def __init__(self, connection, game_state):
        super().__init__(connection, game_state)

    def __call__(
        self,
        target_x: float,
        target_y: float,
        duration_ticks: int = 30,
    ):
        """
        Command the player character to shoot towards a target position.
        :param target_x: Target X coordinate
        :param target_y: Target Y coordinate
        :param duration_ticks: How many ticks to sustain shooting
        :return: Result string
        """
        response, _ = self.execute(
            self.player_index, target_x, target_y, duration_ticks
        )
        return self.clean_response(response)
