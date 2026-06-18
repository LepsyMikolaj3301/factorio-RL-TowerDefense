from typing import List, Sequence, Tuple, Union
from fle.env.tools import Tool

PositionLike = Union[Tuple[float, float], Sequence[float]]


class DestroyTurrets(Tool):
    def __init__(self, connection, game_state):
        super().__init__(connection, game_state)

    def __call__(
        self,
        positions: List[PositionLike],
        radius: float = 0.6,
    ):
        """
        Destroy the gun-turrets standing at the given slot positions.

        Used by the tower-defense env to vacate a random subset of turret slots
        at the start of each episode. Destroyed turrets are NOT returned to any
        inventory -- the slot simply becomes empty.

        :param positions: List of (x, y) slot centers to clear.
        :param radius: Search radius around each position to match a turret.
        :return: Number of turrets destroyed.
        """
        if not positions:
            return 0

        # Convert positions to a Lua-friendly list: {x=..,y=..}, {x=..,y=..}
        positions_str = ", ".join(
            f"{{x={float(p[0])},y={float(p[1])}}}" for p in positions
        )
        response, _ = self.execute(self.player_index, positions_str, float(radius))
        return self.clean_response(response)
