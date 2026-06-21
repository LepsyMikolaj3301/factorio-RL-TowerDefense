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
# The action set is limited to the three meaningful turret decisions
# (pick a turret up, place one down, feed ammo to one) plus a no-op. Walls and
# direct shooting are intentionally not exposed to the policy.
ACTION_NOOP = 0
ACTION_PICK_TURRET = 1     # mine the turret in a slot back into the agent inventory
ACTION_PLACE_TURRET = 2    # place an inventory turret into an empty slot
ACTION_REFILL_TURRET = 3   # insert ammo into the turret standing in a slot
ACTION_MOVE_ANCHOR = 4     # teleport the character onto a chosen anchor tile
NUM_ACTION_TYPES = 5

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

# Max anchor tiles read from the map (padding target for the anchor arrays).
# The character may only ever stand on one of these positions.
MAX_ANCHORS = 32

# Character interaction range in tiles. Slots within this distance of the
# character's current anchor are valid PLACE/PICK/REFILL targets.
REACH_DISTANCE = 10.0


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
            # Real slots that currently hold a turret (valid PICK_TURRET targets).
            "pick_slot_mask": spaces.MultiBinary(MAX_SLOTS),
            # One row per anchor tile read from the map. (x, y) is the tile
            # center; padding rows beyond the real anchor count are zero.
            "anchors": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(MAX_ANCHORS, 2),  # x, y
                dtype=np.float32,
            ),
            # Real (non-padding) anchors. Doubles as the MOVE_TO_ANCHOR mask.
            "anchor_valid_mask": spaces.MultiBinary(MAX_ANCHORS),
            # Which turret slots are within REACH_DISTANCE of the character's
            # current anchor. The PPO policy uses this to filter valid slot actions.
            "reach_slot_mask": spaces.MultiBinary(MAX_SLOTS),
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
            "anchor_index": spaces.Discrete(MAX_ANCHORS),
        }
    )


def flatten_action_space(grid_size: int = DEFAULT_GRID_SIZE) -> spaces.MultiDiscrete:
    """Alternative flat action space for algorithms that don't support Dict."""
    return spaces.MultiDiscrete([NUM_ACTION_TYPES, MAX_SLOTS, 51, MAX_ANCHORS])


class FlatTDActionWrapper(gymnasium.Wrapper):
    """Wraps TowerDefenseEnv to expose a flat MultiDiscrete action space.

    Order: [action_type, slot_index, ammo_amount, anchor_index]
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
            "anchor_index": int(action[3]),
        }
        return self.env.step(dict_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
