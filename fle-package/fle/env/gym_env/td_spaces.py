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

# --- Threat tracking (biter groups + nests) ---
# Max biter *groups* (swarms / clusters) tracked. Each is a cluster of live
# enemy units, distinct from a nest (spawner). Padding target for the array.
MAX_GROUPS = 12
# Per-group feature width:
#   [rel_cx, rel_cy, count_norm, spread_norm, head_dx, head_dy,
#    dist_radar_norm, dist_turret_norm, eta_norm, is_swarm]
GROUP_FEATURES = 10

# Max biter nests (unit-spawners) tracked. Padding target for the array.
MAX_NESTS = 8
# Per-nest feature width: [rel_cx, rel_cy, health_norm, dist_radar_norm, dir_dx, dir_dy]
NEST_FEATURES = 6

# Max boilers tracked as critical base structures.
MAX_BOILERS = 32
# Per-boiler feature width: [rel_x, rel_y, alive, health_norm]
BOILER_FEATURES = 4

# A cluster of at least this many live enemies is flagged as a "swarm"
# (is_swarm=1). The big groups (~60) you described read very differently from
# a nest at the symbolic level because of this flag + the count feature.
SWARM_THRESHOLD = 60

# --- Movement / transit state ---
# [is_moving, target_anchor_norm, remaining_dist_norm, head_dx, head_dy]
MOVEMENT_FEATURES = 5

# --- Recent directional structure losses ---
# [wall_n, wall_e, wall_s, wall_w, turret_n, turret_e, turret_s, turret_w]
RECENT_LOSS_FEATURES = 8

# --- Normalization constants (used by the env to condition obs for the NN) ---
COUNT_NORM = 100.0       # divide enemy counts by this
ETA_NORM = 600.0         # ticks-to-base normalizer (~10s at 60tps)
NEST_HP_NORM = 350.0     # spawner max HP ballpark
CHAR_HP_NORM = 250.0     # character max HP ballpark
RADAR_HP_NORM = 250.0    # radar max HP ballpark
BOILER_HP_NORM = 200.0   # boiler max HP ballpark
TURRET_HP_NORM = 400.0   # gun-turret max HP ballpark
TURRET_AMMO_NORM = 10.0  # gun-turret ammo slot capacity ballpark
LOSS_COUNT_NORM = 10.0   # recent loss count normalizer for one decision window


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
            # Action-conditioned reach masks: the per-action slot mask AND'd with
            # reach_slot_mask. The pointer head consumes these directly so it can
            # never select an out-of-reach slot for the chosen action type.
            "place_reach_mask": spaces.MultiBinary(MAX_SLOTS),
            "refill_reach_mask": spaces.MultiBinary(MAX_SLOTS),
            "pick_reach_mask": spaces.MultiBinary(MAX_SLOTS),
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
            # Biter groups (swarms): clusters of live enemies, distinct from nests.
            # All positions are relative to the radar center and normalized.
            "biter_groups": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(MAX_GROUPS, GROUP_FEATURES),
                dtype=np.float32,
            ),
            "group_valid_mask": spaces.MultiBinary(MAX_GROUPS),
            # Biter nests (unit-spawners): the origins, even when at the grid edge.
            "nests": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(MAX_NESTS, NEST_FEATURES),
                dtype=np.float32,
            ),
            "nest_valid_mask": spaces.MultiBinary(MAX_NESTS),
            # Boilers are critical power targets read once from the map.
            "boilers": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(MAX_BOILERS, BOILER_FEATURES),
                dtype=np.float32,
            ),
            "boiler_valid_mask": spaces.MultiBinary(MAX_BOILERS),
            # Transit state for the async A* walk (see TowerDefenseEnv._move_to_anchor).
            "movement": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(MOVEMENT_FEATURES,),
                dtype=np.float32,
            ),
            # Last decision-window directional losses relative to the radar:
            # wall N/E/S/W then turret N/E/S/W, clipped and normalized.
            "recent_losses": spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(RECENT_LOSS_FEATURES,),
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
