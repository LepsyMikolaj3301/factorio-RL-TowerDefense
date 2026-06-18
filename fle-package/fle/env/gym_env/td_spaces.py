"""Tower Defense action and observation space definitions.

The environment uses a *slot-based* turret model: the loaded map defines a fixed
set of turret positions ("slots"), and the agent can only place turrets into
those slots or refill the turrets standing in them. There is no free-coordinate
placement, so the action space is small and well-structured.
"""

import numpy as np
import gymnasium
from gymnasium import spaces

# --- Action constants ---
# The action set is deliberately limited to the two meaningful turret decisions
# (where to place, where to feed ammo) plus a no-op. Walls and direct shooting
# are intentionally not exposed to the policy.
ACTION_NOOP = 0
ACTION_PLACE_TURRET = 1
ACTION_REFILL_TURRET = 2
NUM_ACTION_TYPES = 3

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

# Max turret slots read from the map (padding target for the slot arrays).
MAX_SLOTS = 64

# Per-slot feature width: [x, y, occupied, ammo, health]
SLOT_FEATURES = 5


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
            # One row per canonical turret slot. occupied is 1 (turret present),
            # 0 (empty slot) or -1 (padding row beyond the real slot count).
            "turret_slots": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(MAX_SLOTS, SLOT_FEATURES),  # x, y, occupied, ammo, health
                dtype=np.float32,
            ),
            # Real (non-padding) slots.
            "slot_valid_mask": spaces.MultiBinary(MAX_SLOTS),
            # Real slots that are currently empty (valid PLACE_TURRET targets).
            "place_slot_mask": spaces.MultiBinary(MAX_SLOTS),
            # Real slots that currently hold a turret (valid REFILL_TURRET targets).
            "refill_slot_mask": spaces.MultiBinary(MAX_SLOTS),
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
    """Create the tower defense action space.

    grid_size is accepted for signature compatibility but no longer affects the
    action space, which is purely slot-index based.
    """
    return spaces.Dict(
        {
            "action_type": spaces.Discrete(NUM_ACTION_TYPES),
            "slot_index": spaces.Discrete(MAX_SLOTS),
            "ammo_amount": spaces.Discrete(51),  # 0-50
        }
    )


def flatten_action_space(grid_size: int = DEFAULT_GRID_SIZE) -> spaces.MultiDiscrete:
    """Alternative flat action space for algorithms that don't support Dict."""
    return spaces.MultiDiscrete([NUM_ACTION_TYPES, MAX_SLOTS, 51])


class FlatTDActionWrapper(gymnasium.Wrapper):
    """Wraps TowerDefenseEnv to expose a flat MultiDiscrete action space.

    Order: [action_type, slot_index, ammo_amount]
    Compatible with SB3 PPO / any algorithm that needs a non-Dict action space.
    """

    def __init__(self, env):
        super().__init__(env)
        grid_size = env.unwrapped.grid_size
        self.action_space = flatten_action_space(grid_size)

    def step(self, action):
        dict_action = {
            "action_type": int(action[0]),
            "slot_index": int(action[1]),
            "ammo_amount": int(action[2]),
        }
        return self.env.step(dict_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
