"""Tower Defense action and observation space definitions."""

import numpy as np
import gymnasium
from gymnasium import spaces

# --- Action constants ---
ACTION_NOOP = 0
ACTION_PLACE_WALL = 1
ACTION_PLACE_TURRET = 2
ACTION_MOVE_WALL = 3
ACTION_MOVE_TURRET = 4
ACTION_REFILL_TURRET = 5
ACTION_SHOOT = 6
NUM_ACTION_TYPES = 7

# --- Observation grid ---
NUM_CHANNELS = 8  # empty, wall, turret, ammo_pct, biter, spitter, spawner, character
DEFAULT_GRID_SIZE = 64  # 64x64 tiles

# --- Inventory items tracked ---
TRACKED_ITEMS = [
    "firearm-magazine",
    "piercing-rounds-magazine",
    "gun-turret",
    "stone-wall",
]

# Max entities for padding
MAX_TURRETS = 50
MAX_WALLS = 200


def make_observation_space(grid_size: int = DEFAULT_GRID_SIZE) -> spaces.Dict:
    """Create the tower defense observation space."""
    return spaces.Dict(
        {
            "map": spaces.Box(
                low=0,
                high=255,
                shape=(NUM_CHANNELS, grid_size, grid_size),
                dtype=np.uint8,
            ),
            "inventory": spaces.Box(
                low=0,
                high=10000,
                shape=(len(TRACKED_ITEMS),),
                dtype=np.int32,
            ),
            "turrets": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(MAX_TURRETS, 4),  # x, y, ammo, health
                dtype=np.float32,
            ),
            "walls": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(MAX_WALLS, 3),  # x, y, health
                dtype=np.float32,
            ),
            "character": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),  # x, y, health, weapon_ammo
                dtype=np.float32,
            ),
            "radar": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),  # x, y, health
                dtype=np.float32,
            ),
            "game": spaces.Box(
                low=0,
                high=np.inf,
                shape=(2,),  # elapsed_ticks, wave_intensity_proxy
                dtype=np.float32,
            ),
        }
    )


def make_action_space(grid_size: int = DEFAULT_GRID_SIZE) -> spaces.Dict:
    """Create the tower defense action space."""
    return spaces.Dict(
        {
            "action_type": spaces.Discrete(NUM_ACTION_TYPES),
            "target_x": spaces.Discrete(grid_size),
            "target_y": spaces.Discrete(grid_size),
            "source_x": spaces.Discrete(grid_size),
            "source_y": spaces.Discrete(grid_size),
            "ammo_amount": spaces.Discrete(51),  # 0-50
        }
    )


def flatten_action_space(grid_size: int = DEFAULT_GRID_SIZE) -> spaces.MultiDiscrete:
    """Alternative flat action space for algorithms that don't support Dict."""
    return spaces.MultiDiscrete(
        [NUM_ACTION_TYPES, grid_size, grid_size, grid_size, grid_size, 51]
    )


class FlatTDActionWrapper(gymnasium.Wrapper):
    """Wraps TowerDefenseEnv to expose a flat MultiDiscrete action space.

    Order: [action_type, target_x, target_y, source_x, source_y, ammo_amount]
    Compatible with SB3 PPO / any algorithm that needs a non-Dict action space.
    """

    def __init__(self, env):
        super().__init__(env)
        grid_size = env.unwrapped.grid_size
        self.action_space = flatten_action_space(grid_size)

    def step(self, action):
        dict_action = {
            "action_type": int(action[0]),
            "target_x": int(action[1]),
            "target_y": int(action[2]),
            "source_x": int(action[3]),
            "source_y": int(action[4]),
            "ammo_amount": int(action[5]),
        }
        return self.env.step(dict_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
