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
    ):
        """
        Spawn a wave of enemies around a center point.
        Wave size scales with wave_number: count = base_count * escalation_factor^(wave_number-1)
        Enemies spawn in a ring at spawn_radius distance.
        :param wave_number: Current wave number (1-indexed)
        :param center_x: Center X of the defense area
        :param center_y: Center Y of the defense area
        :param spawn_radius: Distance from center to spawn enemies
        :param base_count: Number of enemies in wave 1
        :param escalation_factor: Multiplier per wave
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
        )
        return self.clean_response(response)
