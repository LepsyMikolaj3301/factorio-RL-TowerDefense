import base64
import numpy as np
from fle.env.tools import Tool


# Channel indices for the symbolic grid
CHANNEL_EMPTY = 0
CHANNEL_WALL = 1
CHANNEL_TURRET = 2
CHANNEL_AMMO_PCT = 3
CHANNEL_BITER = 4
CHANNEL_SPITTER = 5
CHANNEL_SPAWNER = 6
CHANNEL_CHARACTER = 7
NUM_CHANNELS = 8


class RadarView(Tool):
    def __init__(self, connection, game_state):
        super().__init__(connection, game_state)

    def __call__(
        self,
        center_x: float = 0.0,
        center_y: float = 0.0,
        radius: int = 32,
        cell_size: float = 1.0,
        charted_only: bool = True,
    ) -> np.ndarray:
        """
        Get a symbolic grid observation around a center point.
        Returns uint8[C,H,W] tensor where C=8 channels:
          0: empty, 1: wall, 2: turret, 3: ammo_loaded_pct,
          4: biter, 5: spitter, 6: spawner, 7: character

        :param center_x: Center X coordinate
        :param center_y: Center Y coordinate
        :param radius: Radius in tiles (grid will be 2*radius x 2*radius)
        :param cell_size: Size of each grid cell in tiles
        :param charted_only: Only include entities in charted chunks
        :return: numpy uint8 array of shape (NUM_CHANNELS, H, W)
        """
        response, _ = self.execute(
            self.player_index, center_x, center_y, radius, cell_size, charted_only
        )
        cleaned = self.clean_response(response)

        if isinstance(cleaned, str) and cleaned.startswith("b64:"):
            grid_size = int(2 * radius / cell_size)
            raw = base64.b64decode(cleaned[4:])
            arr = np.frombuffer(raw, dtype=np.uint8)
            expected = NUM_CHANNELS * grid_size * grid_size
            if len(arr) == expected:
                return arr.reshape(NUM_CHANNELS, grid_size, grid_size)
            else:
                # Fallback: return zeros
                return np.zeros(
                    (NUM_CHANNELS, grid_size, grid_size), dtype=np.uint8
                )
        else:
            grid_size = int(2 * radius / cell_size)
            return np.zeros(
                (NUM_CHANNELS, grid_size, grid_size), dtype=np.uint8
            )
