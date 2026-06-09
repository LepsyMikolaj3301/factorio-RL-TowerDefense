from fle.env.tools import Tool


class SpawnEnemy(Tool):
    def __init__(self, connection, game_state):
        super().__init__(connection, game_state)

    def __call__(
        self,
        entity_type: str = "small-biter",
        x: float = 0.0,
        y: float = 0.0,
        count: int = 1,
    ):
        """
        Spawn enemy units at a given position.
        :param entity_type: Type of enemy (small-biter, medium-biter, big-biter, behemoth-biter,
                            small-spitter, medium-spitter, big-spitter, behemoth-spitter,
                            biter-spawner, spitter-spawner)
        :param x: X coordinate
        :param y: Y coordinate
        :param count: Number of enemies to spawn
        :return: Number of enemies successfully spawned
        """
        response, _ = self.execute(self.player_index, entity_type, x, y, count)
        return self.clean_response(response)
