"""Scenario configuration for TowerDefenseEnv.

Usage:
    from fle.env.gym_env.td_config import TDScenarioConfig

    env = TowerDefenseEnv(instance, config=TDScenarioConfig.HARD)
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple


@dataclass
class TDScenarioConfig:
    """All tunable parameters for a tower-defense scenario.

    Named presets are available as class attributes:
        TDScenarioConfig.EASY    — gentle waves, generous inventory
        TDScenarioConfig.MEDIUM  — balanced defaults
        TDScenarioConfig.HARD    — fast escalation, sparse resources
    """

    # --- Wave escalation ---
    base_enemy_count: int = 5
    escalation_factor: float = 1.5
    ticks_per_wave: int = 3600       # 60 s at 60 tps

    # --- Environment cadence ---
    decision_cadence: int = 60       # ticks between agent steps
    game_speed: float = 10.0
    max_ticks: int = 108_000         # 30 min at 60 tps

    # --- Grid / observation ---
    grid_size: int = 64
    grid_radius: float = 32.0

    # --- Turret slots ---
    # Fraction of the map's pre-placed turrets to remove at the start of every
    # episode, leaving empty slots for the agent to (re)fill. Re-rolled per reset.
    turret_deletion_percentage: float = 0.5
    # Upper bound on the number of slots tracked (must match td_spaces.MAX_SLOTS).
    max_turret_slots: int = 64
    # Optional explicit slot positions. If the loaded map has no pre-placed
    # turrets, the env seeds turrets at these positions on first reset. If None,
    # slots are read purely from the map's existing turrets.
    turret_slot_positions: Optional[List[Tuple[float, float]]] = None

    # --- Anchor tiles ---
    # The character may only stand on "anchor" positions, which the map author
    # paints with the `hazard-concrete-left` tile texture. The env scans for
    # these tiles once on first reset and lets the policy move between them.
    # Upper bound on anchors tracked (must match td_spaces.MAX_ANCHORS).
    max_anchor_slots: int = 32
    # Half-width (tiles) of the square area scanned around the radar center for
    # anchor tiles. Matches the chart radius used by _read_radar.
    anchor_scan_radius: float = 128.0

    # --- Spawn & reset behaviour ---
    # Where the agent character is placed after each reset. The radar occupies
    # (0,0) by convention; (0,10) keeps the agent inside the walled ring but
    # off the radar tile.
    player_spawn_position: Tuple[float, float] = (0.0, 10.0)
    # When False, _spawn_wave() is a no-op — biters must originate from the
    # map's own Factorio AI (useful for debugging or map-only configs).
    spawn_waves_at_runtime: bool = True
    # When True, waves are sent from the map's baked-in unit-spawners (enemy
    # force). Falls back to the geometric-ring origin if no spawners exist.
    spawn_from_map_spawners: bool = True
    # Destroy stale enemy *units* at the start of each episode so the agent
    # always begins with a clean board. Spawner entities are never touched.
    clear_biters_on_reset: bool = True

    # --- Reward weights ---
    alpha_survive: float = 0.01      # per step
    beta_kills: float = 1.0          # per kill
    delta_damage: float = 0.5        # per HP of damage taken
    epsilon_invalid: float = 0.5     # per invalid action
    terminal_bonus: float = 100.0
    terminal_penalty: float = -100.0

    # --- Starting inventory ---
    # Only ammo is given at the start. Turrets come from the map: N are
    # removed and placed in the agent's inventory by _delete_turret_subset().
    starting_inventory: Dict[str, int] = field(default_factory=lambda: {
        "piercing-rounds-magazine": 500,
    })

    # Named presets (assigned after class definition below)
    EASY: ClassVar["TDScenarioConfig"]
    MEDIUM: ClassVar["TDScenarioConfig"]
    HARD: ClassVar["TDScenarioConfig"]


TDScenarioConfig.EASY = TDScenarioConfig(
    base_enemy_count=3,
    escalation_factor=1.2,
    ticks_per_wave=5400,   # 90 s between waves
    turret_deletion_percentage=0.3,  # most turrets stay; light replanning
    starting_inventory={
        "piercing-rounds-magazine": 1000,
    },
)

TDScenarioConfig.MEDIUM = TDScenarioConfig()

TDScenarioConfig.HARD = TDScenarioConfig(
    base_enemy_count=10,
    escalation_factor=2.0,
    ticks_per_wave=2400,   # 40 s between waves
    turret_deletion_percentage=0.7,  # many empty slots, scarce turrets to fill them
    starting_inventory={
        "piercing-rounds-magazine": 200,
    },
)
