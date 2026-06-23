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

    # --- Difficulty: biter strength & attack-group size ---
    # Enemy evolution factor (0..1). Applied to the enemy force when the world is
    # initialized and re-applied after every reset (a snapshot restore reverts
    # the force's evolution to the captured value). None = keep the save's value.
    evolution_factor: Optional[float] = None
    # Factorio map_settings.unit_group.max_unit_group_size — the cap on biters per
    # attack group. Larger = bigger swarms = harder. Applied on init + every reset.
    # None = leave the engine/map default untouched.
    max_unit_group_size: Optional[int] = None

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

    # --- Boilers ---
    # Boilers are critical power structures read once from the loaded map. If at
    # least this fraction of the initial boiler set is destroyed, the episode
    # terminates and the next reset restores the map.
    max_boiler_slots: int = 32
    boiler_scan_radius: float = 128.0
    boiler_loss_fraction: float = 1.0 / 3.0

    # --- Spawn & reset behaviour ---
    # Where the agent character is placed after each reset. The radar occupies
    # (0,0) by convention; (0,10) keeps the agent inside the walled ring but
    # off the radar tile.
    player_spawn_position: Tuple[float, float] = (0.0, 10.0)
    # Runtime wave spawning is disabled for predefined TD saves; attacks come
    # from the map's own unit-spawners.
    spawn_waves_at_runtime: bool = False
    # When True, waves are sent from the map's baked-in unit-spawners (enemy
    # force). Falls back to the geometric-ring origin if no spawners exist.
    spawn_from_map_spawners: bool = True
    # Destroy stale enemy *units* at the start of each episode so the agent
    # always begins with a clean board. Spawner entities are never touched.
    clear_biters_on_reset: bool = True
    # Rebuild the enemy nest layout (unit-spawners + worm turrets) to the map's
    # original starting set on every reset, removing nests created by expansion
    # during the previous episode. Recorded once on first reset.
    restore_starting_nests_on_reset: bool = True
    # Disable Factorio's enemy expansion so biters never create new nests beyond
    # the ones baked into the save. The scenario control.lua sets this on a fresh
    # map, but on_init never runs on a loaded save, so the env enforces it via
    # RCON on the first reset (map_settings persist across snapshot restores).
    disable_enemy_expansion: bool = True

    # --- Ammo ---
    # The magazine type the agent uses everywhere (starting inventory + the
    # REFILL action). Must match the ammo granted in `starting_inventory`,
    # otherwise refill inserts an item the agent does not carry and turrets
    # never get ammo.
    ammo_type: str = "piercing-rounds-magazine"

    # --- Threat tracking ---
    # A live enemy cluster of >= this many units is flagged is_swarm in the obs.
    swarm_threshold: int = 60
    # Radius (tiles) scanned for biter groups (larger than the obs grid so the
    # agent sees a swarm forming on the approach). Nests are searched surface-wide.
    threat_scan_radius: float = 96.0
    # Coarse cell size (tiles) used to bucket enemies before clustering.
    threat_cell_size: float = 6.0

    # --- Movement ---
    # Tiles the async on_tick walker advances the character per game tick.
    walk_speed: float = 0.2

    # --- Reward weights ---
    alpha_survive: float = 0.01      # per step survived
    beta_kills: float = 1.0          # per enemy killed
    delta_damage: float = 0.5        # per HP of character damage taken
    epsilon_invalid: float = 0.5     # per invalid action
    # Destruction penalties (per entity lost to the enemy this step).
    p_wall_destroyed: float = 12.0         # strong; biters destroy walls in packs
    p_turret_destroyed: float = 20.0       # very strong (turret permanently lost)
    p_building_destroyed: float = 20.0     # very strong
    p_radar_damage: float = 0.2            # per HP the radar (the "heart") loses
    p_empty_turret: float = 0.05           # per occupied turret with no usable ammo
    p_move_command: float = 0.25            # per valid MOVE_ANCHOR command
    p_move_transit: float = 0.05            # per step spent walking/in transit
    terminal_bonus: float = 100.0          # survived to max_ticks
    terminal_penalty: float = -200.0       # radar destroyed or character died (very strong)

    # --- Dense shaping (annealed to 0 over shaping_decay_steps env steps) ---
    w_coverage: float = 0.0    # legacy absolute coverage shaping (kept disabled)
    w_ammo: float = 0.0        # legacy absolute ammo shaping (kept disabled)
    w_coverage_delta: float = 0.5  # reward when filled slot coverage improves
    w_ammo_delta: float = 0.2      # reward when loaded-turret fraction improves
    # Reward when armed turret coverage improves on the side where live threats
    # or, before contact, visible nests indicate the next attack is likely from.
    w_threatened_turret_delta: float = 0.75
    w_early_place_turret: float = 0.10
    w_early_refill_turret: float = 0.20
    early_turret_setup_steps: int = 200
    w_threat: float = 0.05     # penalty scaling with closeness of the nearest swarm
    shaping_decay_steps: int = 200_000

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
