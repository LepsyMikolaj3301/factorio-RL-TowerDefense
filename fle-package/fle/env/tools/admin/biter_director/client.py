from fle.env.tools import Tool


class BiterDirector(Tool):
    def __init__(self, connection, game_state):
        super().__init__(connection, game_state)

    def __call__(
        self,
        wave_number: int = 1,
        center_x: float = 0.0,
        center_y: float = 0.0,
        spawn_radius: float = 50.0,
        base_count: int = 5,
        escalation_factor: float = 1.5,
        use_spawners: bool = False,
        target_x: float = 0.0,
        target_y: float = 0.0,
    ):
        """
        Spawn a wave of enemies.

        Wave size scales with wave_number: count = base_count * escalation_factor^(wave_number-1).

        When use_spawners=True, biters are created near the map's baked-in unit-spawner
        entities (enemy force) and commanded to attack (target_x, target_y). Falls back to
        a geometric ring around (center_x, center_y) when no spawners exist.

        :param wave_number: Current wave number (1-indexed)
        :param center_x: Center X of the defense area (ring fallback origin)
        :param center_y: Center Y of the defense area (ring fallback origin)
        :param spawn_radius: Distance from center to spawn enemies (ring fallback)
        :param base_count: Number of enemies in wave 1
        :param escalation_factor: Multiplier per wave
        :param use_spawners: When True, spawn near the map's unit-spawner entities
        :param target_x: Attack destination X (defaults to center_x)
        :param target_y: Attack destination Y (defaults to center_y)
        :return: Number of enemies spawned
        """
        response, _ = self.execute(
            self.player_index,
            wave_number,
            center_x,
            center_y,
            spawn_radius,
            base_count,
            escalation_factor,
            use_spawners,
            target_x,
            target_y,
        )
        return self.clean_response(response)
