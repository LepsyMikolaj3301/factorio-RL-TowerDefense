"""Tower Defense Gymnasium environment for Factorio."""

import copy
import logging
import math
import time as _time_module
from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np

from fle.commons.models.game_state import GameState
from fle.env import FactorioInstance
from fle.env.gym_env.td_config import TDScenarioConfig
from fle.env.gym_env.td_spaces import (
    ACTION_MOVE_ANCHOR,
    ACTION_NOOP,
    ACTION_PICK_TURRET,
    ACTION_PLACE_TURRET,
    ACTION_REFILL_TURRET,
    ACTION_TAKE_AMMO,
    BOILER_FEATURES,
    BOILER_HP_NORM,
    CHAR_HP_NORM,
    COUNT_NORM,
    ETA_NORM,
    GROUP_FEATURES,
    LOSS_COUNT_NORM,
    MAX_ANCHORS,
    MAX_BOILERS,
    MAX_GROUPS,
    MAX_NESTS,
    MAX_SLOTS,
    MOVEMENT_FEATURES,
    NEST_FEATURES,
    NEST_HP_NORM,
    NUM_CHANNELS,
    RADAR_HP_NORM,
    REACH_DISTANCE,
    SLOT_FEATURES,
    TRACKED_ITEMS,
    TURRET_AMMO_NORM,
    TURRET_HP_NORM,
    make_action_space,
    make_observation_space,
)
from fle.env.game_types import Prototype
from fle.env.entities import Direction, Position

logger = logging.getLogger(__name__)

# RCON network errors (socket dropped / not connected). Factorio closes the
# RCON socket when a human client joins to spectate, and a server crash+restart
# also surfaces here. We catch the common base so step()/reset() can reconnect
# instead of killing the training run. Fall back to Exception if the lib layout
# changes.
try:
    from factorio_rcon.factorio_rcon import RCONNetworkError as _RCONNetworkError
except ImportError:  # pragma: no cover
    _RCONNetworkError = Exception


class TowerDefenseEnv(gymnasium.Env):
    """
    Gymnasium-compatible Tower Defense environment for Factorio.

    The agent defends a radar from waves of biters using walls, turrets,
    and direct shooting. Observation is a symbolic grid + structured data.

    Pass a TDScenarioConfig to tune difficulty:
        env = TowerDefenseEnv(instance, config=TDScenarioConfig.HARD)

    Common parameters can be overridden directly without building a config:
        env = TowerDefenseEnv(instance, turret_deletion_percentage=0.3)
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        instance: FactorioInstance,
        config: Optional[TDScenarioConfig] = None,
        render_mode: Optional[str] = None,
        *,
        turret_deletion_percentage: Optional[float] = None,
    ):
        super().__init__()

        self.instance = instance
        # Copy so per-instance overrides never mutate shared preset objects.
        self.config = copy.copy(config or TDScenarioConfig.MEDIUM)
        if turret_deletion_percentage is not None:
            self.config.turret_deletion_percentage = float(turret_deletion_percentage)
        self.render_mode = render_mode

        # Convenience aliases for readability inside the class
        cfg = self.config
        self.grid_size = cfg.grid_size
        self.cell_size = 1.0
        self.decision_cadence = cfg.decision_cadence
        self.max_ticks = cfg.max_ticks
        self.game_speed = cfg.game_speed
        self.max_slots = min(cfg.max_turret_slots, MAX_SLOTS)
        self.max_anchors = min(cfg.max_anchor_slots, MAX_ANCHORS)
        self.max_boilers = min(cfg.max_boiler_slots, MAX_BOILERS)

        # Distance (tiles) within which a live turret is considered to occupy a
        # slot, or two read positions are considered the same slot.
        self._slot_epsilon = 0.6

        # Spaces
        self.observation_space = make_observation_space(self.grid_size)
        self.action_space = make_action_space(self.grid_size)

        # Internal state
        self._step_count = 0
        self._total_steps = 0  # lifetime steps (for shaping decay)
        self._prev_kills = 0
        self._prev_health = 0.0
        self._prev_radar_hp = 0.0
        self._wave_number = 0
        self._center_x = 0.0
        self._center_y = 0.0
        self._radius = self.grid_size * self.cell_size / 2.0
        # Position normalizer (radar-relative coords are divided by this).
        self._norm = max(1.0, self._radius)

        # Async-movement state (set by _move_to_anchor, advanced in step()).
        # The loaded predefined save's control.lua has no on_tick walker, so the
        # env drives movement itself: it glides the character along _walk_path by
        # (decision_cadence * walk_speed) tiles each step, during the unpaused
        # window, so travel costs real game-time and biters attack mid-transit.
        self._moving = False
        self._move_target = -1
        self._walk_path: list = []   # list of (x, y) world waypoints
        self._walk_idx = 0           # index of the next waypoint to reach

        # Last good observation, returned if RCON drops mid-step so the episode
        # can terminate with a valid (if stale) obs instead of crashing.
        self._last_obs: Optional[Dict[str, np.ndarray]] = None

        # Raw (un-normalized) values stashed during _get_observation so the
        # reward/termination logic doesn't re-query the game.
        self._cur_char_hp = 0.0
        self._cur_radar_hp = 0.0
        self._cur_coverage = 0.0       # filled_reachable / total_reachable
        self._cur_ammo_frac = 0.0      # loaded_turrets / live_turrets
        self._cur_empty_turrets = 0    # live gun-turrets with no configured ammo
        self._cur_threat_closeness = 0.0  # 0 (far) .. 1 (on top of base)
        self._cur_threatened_turret_readiness = 0.0
        self._last_events: Dict[str, int] = self._zero_events()
        self._episode_events: Dict[str, int] = self._zero_events()
        self._prev_coverage = 0.0
        self._prev_ammo_frac = 0.0
        self._prev_threatened_turret_readiness = 0.0
        self._last_early_setup_reward = 0.0
        self._last_ammo_taken = 0
        self._cur_slot_threat_alignment = np.ones(MAX_SLOTS, dtype=np.float32)
        self._cur_has_threat_direction = False

        # Canonical turret slots, read once from the map on first reset.
        # Shape (N, 2): the (x, y) center of every slot. Never changes for a run.
        self._turret_slots: Optional[np.ndarray] = None

        # Canonical boiler positions, read once from the map on first reset.
        # Shape (N, 2). If one third of this initial set is destroyed, the
        # episode terminates and training resets from the snapshot.
        self._boiler_slots: Optional[np.ndarray] = None

        # Anchor tiles, read once from the map on first reset. Shape (N, 2): the
        # (x, y) center of every `hazard-concrete-left` tile. These are the only
        # positions the character is allowed to move to. Never changes for a run.
        self._anchor_slots: Optional[np.ndarray] = None

        # Precomputed (MAX_ANCHORS, MAX_SLOTS) reach matrix. Entry [i, j] is 1
        # when anchor i is within REACH_DISTANCE of turret slot j, 0 otherwise.
        # Built once after the first reset and reused every step.
        self._anchor_reach_matrix: Optional[np.ndarray] = None

        # Index of the anchor the character last moved to (updated by _move_to_anchor).
        # Used to emit reach_slot_mask without an extra argmin every observation.
        self._current_anchor_index: int = 0

        # Save-state reset
        self._initial_snapshot: Optional[GameState] = None
        self._initial_evolution_factor: float = 0.0

        # Radar state (read from map on first reset; never placed by the env)
        self._has_radar: bool = False
        self._radar_position: Optional[Position] = None

        # Starting enemy nest/worm layout, recorded once on first reset as a
        # list of (name, x, y). Used to rebuild nests on every reset.
        self._starting_nests: Optional[list] = None

        # Infinity chest settings are Factorio-specific entity state. Keep a
        # Python-side copy so episode resets can repair filters even if the
        # generic snapshot loader drops them.
        self._infinity_chest_settings: Optional[list] = None

        # Resolve the configured ammo name to a Prototype once. Used by the
        # REFILL action so it inserts the same magazine the agent actually
        # carries (see TDScenarioConfig.ammo_type).
        self._ammo_prototype = next(
            (p for p in Prototype if p.value[0] == self.config.ammo_type),
            Prototype.FirearmMagazine,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        t_reset = _time_module.perf_counter()

        if self._initial_snapshot is None:
            logger.info("reset() — FIRST RESET: reading map entities and capturing snapshot")
            # First reset: server already loaded the save file.
            # Configure the running game, then capture the snapshot.
            self.instance.set_speed(self.game_speed)
            self.instance.pause()
            self.instance._reset_elapsed_ticks()
            self._reset_td_lua_storage()
            if self.config.disable_enemy_expansion:
                self._disable_enemy_expansion()

            self._read_radar()
            self._read_boiler_slots()
            # Record the pristine nest layout before any episode mutates it.
            self._record_starting_nests()
            self._spawn_player()
            self._clear_resource_entities()
            if self.config.clear_biters_on_reset:
                self._clear_live_biters()
            self._set_starting_inventory()

            # If the map ships without turrets, seed them from config so the
            # slot system still has positions to work with.
            self._seed_turret_positions_if_needed()

            # Read the canonical slot geometry from every turret on the map.
            self._read_turret_slots()

            # Read the anchor tiles (hazard-concrete-left) the character may
            # stand on. Tiles are static, so this is read once and reused.
            self._read_anchor_slots()

            # Precompute which anchors can reach which slots. Static for the run.
            self._build_anchor_reach_matrix()

            self._report_save_state()

            # Capture the snapshot with ALL slot turrets present, so each episode
            # restores the full set before re-rolling which slots start empty.
            self._record_infinity_chest_settings()
            self._capture_save_state()

        else:
            logger.info("reset() — EPISODE RESET #%d: restoring from snapshot", self._step_count)
            # Subsequent resets: wipe and restore from snapshot.
            t_snap = _time_module.perf_counter()
            self.instance.reset(game_state=self._initial_snapshot)
            logger.debug("reset() — snapshot restore took %.3fs", _time_module.perf_counter() - t_snap)
            # GameState captures entities/research/inventories but not the Lua
            # storage global. Reset TD-specific fields explicitly so walk state,
            # event counters, and threat cache don't bleed across episodes —
            # matching the clean state of a freshly-loaded save.
            self._reset_td_lua_storage()
            self.instance.set_speed(self.game_speed)
            self.instance.pause()
            self.instance._reset_elapsed_ticks()

            self._read_radar()
            self._spawn_player()
            if self.config.clear_biters_on_reset:
                self._clear_live_biters()
            if self.config.restore_starting_nests_on_reset:
                self._restore_starting_nests()

            self._restore_infinity_chest_settings()
            self._assert_snapshot_restored()

        # Apply difficulty knobs (evolution factor + biter group size). On the
        # first reset this initializes the world; on later resets it re-applies
        # the settings a snapshot restore would otherwise revert.
        self._apply_difficulty_settings()

        # Re-roll which slots start empty for this episode (seeded by reset seed).
        self._delete_turret_subset()

        self._step_count = 0
        self._prev_kills = 0
        self._prev_health = self._get_character_health()
        self._prev_radar_hp = self._get_radar_hp()
        self._wave_number = 0
        self._current_anchor_index = self._nearest_anchor_index()
        self._moving = False
        self._move_target = -1
        # Flush any death events accumulated outside an episode so the first
        # step's reward only reflects in-episode deaths.
        self._read_events()
        self._last_events = self._zero_events()
        self._episode_events = self._zero_events()

        obs = self._get_observation()
        self._prev_coverage = self._cur_coverage
        self._prev_ammo_frac = self._cur_ammo_frac
        self._prev_threatened_turret_readiness = self._cur_threatened_turret_readiness
        logger.info("reset() — done in %.2fs  slots=%d  anchors=%d  "
                    "char_hp=%.0f  radar_hp=%.0f",
                    _time_module.perf_counter() - t_reset,
                    0 if self._turret_slots is None else len(self._turret_slots),
                    0 if self._anchor_slots is None else len(self._anchor_slots),
                    self._cur_char_hp, self._cur_radar_hp)
        info = {"step": 0, "elapsed_ticks": 0, "wave": 0}
        return obs, info

    def _capture_save_state(self) -> None:
        self._initial_snapshot = GameState.from_instance(self.instance)
        raw = self.instance.rcon_client.send_command(
            "/sc rcon.print(game.forces['enemy'].get_evolution_factor(game.surfaces[1]))"
        )
        try:
            self._initial_evolution_factor = float(raw.strip())
        except (ValueError, AttributeError):
            self._initial_evolution_factor = 0.0

    def _apply_difficulty_settings(self) -> None:
        """Apply the scenario's difficulty knobs to the live game.

        Evolution factor is enemy-force state that GameState restores from the
        snapshot, so it must be re-applied every reset. Group size is a
        map_setting that persists across snapshot restores, but we set it each
        reset too so the world always matches the config. Both knobs default to
        None, in which case the save's own values are preserved (evolution is
        restored to the factor captured on the first reset; group size is left
        at the engine/map default).
        """
        evo = self.config.evolution_factor
        if evo is None:
            evo = self._initial_evolution_factor
        self.instance.rcon_client.send_command(
            f"/sc game.forces['enemy'].set_evolution_factor(game.surfaces[1], {float(evo)})"
        )
        group_size = self.config.max_unit_group_size
        if group_size is not None:
            self.instance.rcon_client.send_command(
                f"/sc game.map_settings.unit_group.max_unit_group_size = {int(group_size)}"
            )

    def _assert_snapshot_restored(self) -> None:
        """Fail fast if GameState restore did not recreate core TD entities."""
        try:
            raw = self.instance.rcon_client.send_command(
                "/sc local s=game.surfaces[1] "
                "local r=#s.find_entities_filtered{name='radar'} "
                "local t=#s.find_entities_filtered{name='gun-turret'} "
                "rcon.print(r..','..t)"
            ).strip()
            radar_count, turret_count = [int(float(x)) for x in raw.split(",", 1)]
        except Exception as e:
            raise RuntimeError(f"Could not validate TD snapshot restore: {e}") from e

        if radar_count <= 0 or turret_count <= 0:
            raise RuntimeError(
                "TD snapshot restore failed: "
                f"radars={radar_count}, gun_turrets={turret_count}. "
                "The entity-state loader did not recreate the saved map."
            )

        if self._infinity_chest_settings:
            try:
                raw = self.instance.rcon_client.send_command(
                    "/sc local n=0 "
                    "for _,e in pairs(game.surfaces[1].find_entities_filtered{name='infinity-chest'}) do "
                    "local ok,filters=pcall(function() return e.infinity_container_filters end) "
                    "if (not ok) or (not filters) then "
                    "ok,filters=pcall(function() return e.infinity_container_filter end) "
                    "end "
                    "if ok and filters then "
                    "if filters.name then filters={filters} end "
                    "for _,f in pairs(filters) do if f and f.name then n=n+1 end end "
                    "end "
                    "end "
                    "rcon.print(n)"
                ).strip()
                filter_count = int(float(raw or 0))
            except Exception as e:
                raise RuntimeError(f"Could not validate infinity-chest filters: {e}") from e
            if filter_count <= 0:
                raise RuntimeError(
                    "TD snapshot restore failed: infinity-chest filters were not restored."
                )

    @staticmethod
    def _lua_string(value: Any) -> str:
        text = str(value)
        text = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{text}"'

    @staticmethod
    def _lua_bool(value: Any) -> str:
        return "true" if bool(value) else "false"

    def _record_infinity_chest_settings(self) -> None:
        """Capture infinity-chest filter settings from the pristine TD save."""
        raw = self.instance.rcon_client.send_command(
            "/sc local out={} "
            "for _,e in pairs(game.surfaces[1].find_entities_filtered{name='infinity-chest'}) do "
            "local remove=false "
            "local ok_remove,value=pcall(function() return e.remove_unfiltered_items end) "
            "if ok_remove and value ~= nil then remove=value end "
            "local filters={} "
            "local ok,fs=pcall(function() return e.infinity_container_filters end) "
            "if (not ok) or (not fs) then "
            "ok,fs=pcall(function() return e.infinity_container_filter end) "
            "end "
            "if ok and fs then "
            "if fs.name then fs={fs} end "
            "for idx,f in pairs(fs) do "
            "if f and f.name then "
            "table.insert(filters, tostring(f.index or idx or #filters+1)..':'.."
            "tostring(f.name)..':'..tostring(f.count or 0)..':'.."
            "tostring(f.mode or 'at-least')) "
            "end "
            "end "
            "end "
            "table.insert(out, tostring(e.position.x)..','..tostring(e.position.y)..','.."
            "tostring(remove)..','..table.concat(filters,'|')) "
            "end "
            "rcon.print(table.concat(out,';'))"
        ).strip()

        settings = []
        if raw:
            for chest_blob in raw.split(";"):
                if not chest_blob:
                    continue
                chest_parts = chest_blob.split(",", 3)
                while len(chest_parts) < 4:
                    chest_parts.append("")
                x_str, y_str, remove_str, filter_str = chest_parts[:4]
                filters = []
                for filter_blob in filter(None, filter_str.split("|")):
                    parts = filter_blob.split(":", 3)
                    if len(parts) != 4:
                        continue
                    index_str, name, count_str, mode = parts
                    try:
                        index = int(float(index_str))
                    except ValueError:
                        index = len(filters) + 1
                    try:
                        count = int(float(count_str))
                    except ValueError:
                        count = 0
                    if name:
                        filters.append({
                            "index": index,
                            "name": name,
                            "count": count,
                            "mode": mode or "at-least",
                        })
                try:
                    settings.append({
                        "x": float(x_str),
                        "y": float(y_str),
                        "remove_unfiltered_items": remove_str == "true",
                        "filters": filters,
                    })
                except ValueError:
                    continue
        self._infinity_chest_settings = settings

    def _restore_infinity_chest_settings(self) -> None:
        """Reapply recorded infinity-chest filter settings after snapshot reset."""
        if not self._infinity_chest_settings:
            return

        entries = []
        for chest in self._infinity_chest_settings:
            filters = []
            for filter_data in chest.get("filters", []):
                filters.append(
                    "{index=%d,name=%s,count=%d,mode=%s}" % (
                        int(filter_data["index"]),
                        self._lua_string(filter_data["name"]),
                        int(filter_data["count"]),
                        self._lua_string(filter_data["mode"]),
                    )
                )
            entries.append(
                "{x=%.6f,y=%.6f,remove=%s,filters={%s}}" % (
                    float(chest["x"]),
                    float(chest["y"]),
                    self._lua_bool(chest.get("remove_unfiltered_items", False)),
                    ",".join(filters),
                )
            )

        self.instance.rcon_client.send_command(
            "/sc local entries={%s} local s=game.surfaces[1] "
            "for _,entry in ipairs(entries) do "
            "local e=s.find_entities_filtered{name='infinity-chest', position={entry.x,entry.y}, radius=0.7}[1] "
            "if e then "
            "pcall(function() e.remove_unfiltered_items=entry.remove end) "
            "local sparse={} local packed={} local slot={} "
            "for _,f in ipairs(entry.filters) do "
            "local with_index={index=f.index,name=f.name,count=f.count,mode=f.mode} "
            "local no_index={name=f.name,count=f.count,mode=f.mode} "
            "sparse[f.index]=with_index table.insert(packed,with_index) slot[f.index]=no_index "
            "end "
            "pcall(function() e.infinity_container_filters=sparse end) "
            "pcall(function() e.infinity_container_filters=packed end) "
            "for index,filter in pairs(sparse) do "
            "pcall(function() e.set_infinity_container_filter(index,filter) end) "
            "if slot[index] then pcall(function() e.set_infinity_container_filter(index,slot[index]) end) end "
            "end "
            "if packed[1] then pcall(function() e.infinity_container_filter=packed[1] end) end "
            "if slot[1] then pcall(function() e.infinity_container_filter=slot[1] end) end "
            "end "
            "end" % ",".join(entries)
        )

    def _reset_td_lua_storage(self) -> None:
        """Reset the TD-specific Lua storage fields to their save-load initial values.

        GameState snapshots capture entities/research/inventories but not the
        Lua `storage` global. These fields must be explicitly wiped each episode
        so walk state, event counters, and threat cache don't carry over from
        the previous episode (matching a freshly-loaded save as used in tests).
        """
        self.instance.rcon_client.send_command(
            "/sc "
            "storage.td_walk = {active=false, path={}, idx=1, player_index=1, target_anchor=-1, speed=0.2}; "
            "storage.td_events = {kills=0, turrets_lost=0, walls_lost=0, "
            "buildings_lost=0, boilers_lost=0, radar_lost=0, char_died=0, "
            "wall_n=0, wall_e=0, wall_s=0, wall_w=0, "
            "turret_n=0, turret_e=0, turret_s=0, turret_w=0}; "
            "storage.td_threat_prev = {tick=0, groups={}}"
        )

    def _disable_enemy_expansion(self) -> None:
        """Stop biters creating new nests beyond the ones baked into the save.

        The scenario control.lua sets this in on_init, but on_init never runs
        when a save is loaded, so the loaded map keeps expansion enabled. We
        enforce it here; map_settings persist across snapshot restores, so doing
        it once on the first reset is enough to keep the nest count stable.
        """
        self.instance.rcon_client.send_command(
            "/sc game.map_settings.enemy_expansion.enabled = false"
        )

    def _read_radar(self) -> None:
        """Find the radar on the map, set the grid center, and chart the area.

        Uses RCON directly (bypasses the force-filtered get_entities tool) so it
        works regardless of which force owns the radar. After locating the radar
        it charts a generous radius around it so _radar_view and get_entities
        return live data on the very first observation.
        """
        try:
            raw = self.instance.rcon_client.send_command(
                "/sc local e=game.surfaces[1].find_entities_filtered{name='radar'}[1];"
                "if e then rcon.print(e.position.x..','..e.position.y) "
                "else rcon.print('nil') end"
            ).strip()
            if raw and raw != "nil":
                cx, cy = map(float, raw.split(","))
                self._has_radar = True
                self._radar_position = Position(x=cx, y=cy)
                self._center_x = cx
                self._center_y = cy
                # Chart a generous area around the radar so the observation grid
                # and get_entities both see live data from the first step.
                r = 128
                self.instance.rcon_client.send_command(
                    f"/sc game.forces.player.chart(game.surfaces[1],"
                    f"{{{{{cx-r},{cy-r}}},{{{cx+r},{cy+r}}}}}"
                    f")"
                )
                return
        except Exception:
            pass
        self._has_radar = False
        self._radar_position = None
        self._center_x = 0.0
        self._center_y = 0.0
        logger.warning(
            "No radar found on map within 128 tiles of (0,0). "
            "The env will not terminate on radar destruction."
        )

    def _read_boiler_slots(self) -> None:
        """Read initial boiler positions from the map into the canonical list."""
        r = float(self.config.boiler_scan_radius)
        cx, cy = self._center_x, self._center_y
        boilers: list = []
        try:
            raw = self.instance.rcon_client.send_command(
                f"/sc local out={{}} "
                f"for _,e in pairs(game.surfaces[1].find_entities_filtered{{"
                f"name='boiler', area={{{{{cx-r},{cy-r}}},{{{cx+r},{cy+r}}}}}"
                f"}}) do "
                f"out[#out+1]=e.position.x..','..e.position.y end "
                f"rcon.print(table.concat(out,';'))"
            ).strip()
            if raw:
                for pair in raw.split(";"):
                    if not pair:
                        continue
                    x, y = pair.split(",", 1)
                    boilers.append((float(x), float(y)))
                    if len(boilers) >= self.max_boilers:
                        break
        except Exception as e:
            logger.warning(f"Failed to read boiler positions: {e}")
        self._boiler_slots = np.array(boilers, dtype=np.float32).reshape(-1, 2)
        if len(boilers) == 0:
            logger.warning(
                "No boilers found near the radar; boiler-loss termination is disabled."
            )

    def _current_boilers(self) -> list:
        """Return live boilers near the radar as (x, y, health)."""
        r = float(self.config.boiler_scan_radius)
        cx, cy = self._center_x, self._center_y
        try:
            raw = self.instance.rcon_client.send_command(
                f"/sc local out={{}} "
                f"for _,e in pairs(game.surfaces[1].find_entities_filtered{{"
                f"name='boiler', area={{{{{cx-r},{cy-r}}},{{{cx+r},{cy+r}}}}}"
                f"}}) do "
                f"out[#out+1]=e.position.x..','..e.position.y..','..(e.health or 0) end "
                f"rcon.print(table.concat(out,';'))"
            ).strip()
        except Exception:
            return []
        boilers = []
        if raw:
            for part in raw.split(";"):
                if not part:
                    continue
                try:
                    x, y, hp = part.split(",")
                    boilers.append((float(x), float(y), float(hp)))
                except ValueError:
                    continue
        return boilers

    def _boiler_status(self, boilers: list) -> Tuple[np.ndarray, np.ndarray]:
        """Match canonical boiler slots against live boilers."""
        n = 0 if self._boiler_slots is None else len(self._boiler_slots)
        alive = np.zeros(n, dtype=bool)
        health = np.zeros(n, dtype=np.float32)
        eps = self._slot_epsilon
        for i in range(n):
            sx, sy = self._boiler_slots[i]
            for (bx, by, hp) in boilers:
                if abs(bx - sx) <= eps and abs(by - sy) <= eps:
                    alive[i] = True
                    health[i] = hp
                    break
        return alive, health

    def _spawn_player(self) -> None:
        """Teleport the agent character to the configured spawn position."""
        x, y = self.config.player_spawn_position
        self.instance.rcon_client.send_command(
            f"/sc storage.agent_characters[1].teleport({{{x},{y}}})"
        )
        try:
            self.instance.first_namespace.player_location = Position(x=float(x), y=float(y))
        except Exception:
            pass

    def _clear_live_biters(self) -> None:
        """Destroy all stale enemy units (biters/spitters). Spawners are preserved."""
        self.instance.rcon_client.send_command(
            "/sc for _,e in pairs(game.surfaces[1].find_entities_filtered("
            "{type='unit', force='enemy'})) do e.destroy() end"
        )

    def _record_starting_nests(self) -> None:
        """Record the map's pristine nest/worm layout (called once, first reset).

        Captures every enemy-force unit-spawner and worm turret as (name, x, y)
        so the layout can be rebuilt exactly on every subsequent reset.
        """
        raw = self.instance.rcon_client.send_command(
            "/sc local o={} "
            "for _,e in pairs(game.surfaces[1].find_entities_filtered{force='enemy'}) do "
            "if e.type=='unit-spawner' or e.type=='turret' then "
            "o[#o+1]=e.name..','..e.position.x..','..e.position.y end end "
            "rcon.print(table.concat(o,';'))"
        ).strip()
        self._starting_nests = []
        if raw and raw != "nil":
            for part in raw.split(";"):
                if not part:
                    continue
                name, x, y = part.rsplit(",", 2)
                self._starting_nests.append((name, float(x), float(y)))

    def _restore_starting_nests(self) -> None:
        """Rebuild the enemy nest layout to the recorded starting set.

        Destroys every current enemy nest/worm (including those created by
        expansion during the previous episode), then recreates the original
        starting nests. Self-healing: also restores starting nests the agent
        destroyed last episode.
        """
        if not self._starting_nests:
            return
        self.instance.rcon_client.send_command(
            "/sc for _,e in pairs(game.surfaces[1].find_entities_filtered{force='enemy'}) do "
            "if e.type=='unit-spawner' or e.type=='turret' then e.destroy() end end"
        )
        parts = "".join(
            f"s.create_entity{{name='{n}',position={{{x},{y}}},force='enemy'}} "
            for (n, x, y) in self._starting_nests
        )
        self.instance.rcon_client.send_command(f"/sc local s=game.surfaces[1] {parts}")

    def _clear_resource_entities(self) -> None:
        """Remove ore deposits from the map — not needed in tower defense."""
        self.instance.rcon_client.send_command(
            "/sc for _,e in pairs(game.surfaces[1].find_entities_filtered("
            "{type='resource'})) do e.destroy() end"
        )

    def _set_starting_inventory(self) -> None:
        """Set the agent character's inventory to the configured starting items."""
        self.instance.first_namespace._set_inventory(self.config.starting_inventory)

    # ------------------------------------------------------------------
    # Turret slot management
    # ------------------------------------------------------------------
    def _current_turrets(self) -> list:
        """Return current gun-turrets as a list of (x, y, ammo, health).

        Uses a direct RCON query filtered to gun-turrets only so that walls,
        infinity-pipes, chests, and other map entities are never serialised over
        RCON (doing a broad get_entities on a map with 500+ walls hangs the step).
        """
        cx, cy, r = self._center_x, self._center_y, self._radius
        t0 = _time_module.perf_counter()
        try:
            ammo_name = self.config.ammo_type.replace("\\", "\\\\").replace("'", "\\'")
            raw = self.instance.rcon_client.send_command(
                f"/sc local out={{}} "
                f"for _,e in pairs(game.surfaces[1].find_entities_filtered{{"
                f"name='gun-turret',"
                f"area={{{{{cx-r},{cy-r}}},{{{cx+r},{cy+r}}}}}"
                f"}}) do "
                f"local ammo=0 "
                f"if e.get_inventory(defines.inventory.turret_ammo) then "
                f"  ammo=e.get_inventory(defines.inventory.turret_ammo).get_item_count('{ammo_name}') "
                f"end "
                f"out[#out+1]=e.position.x..','..e.position.y..','..ammo..','..e.health "
                f"end rcon.print(table.concat(out,';'))"
            ).strip()
        except Exception:
            return []
        turrets = []
        if raw:
            for part in raw.split(";"):
                if not part:
                    continue
                try:
                    x, y, ammo, hp = part.split(",")
                    turrets.append((float(x), float(y), float(ammo), float(hp)))
                except ValueError:
                    continue
        logger.debug("_current_turrets: found %d gun-turrets in %.3fs",
                     len(turrets), _time_module.perf_counter() - t0)
        return turrets

    def _seed_turret_positions_if_needed(self) -> None:
        """Place turrets at configured positions if the map ships with none."""
        positions = self.config.turret_slot_positions
        if not positions:
            return
        if self._current_turrets():
            return  # Map already has turrets; trust the map's layout.
        ns = self.instance.first_namespace
        for (px, py) in positions:
            try:
                ns.place_entity(
                    Prototype.GunTurret,
                    Direction.UP,
                    Position(x=float(px), y=float(py)),
                )
            except Exception as e:
                logger.debug(f"Could not seed turret at ({px},{py}): {e}")

    def _read_turret_slots(self) -> None:
        """Read every turret center on the map into the canonical slot list."""
        slots: list = []
        for (x, y, _ammo, _health) in self._current_turrets():
            # Dedup positions that fall within epsilon of an existing slot.
            is_dup = any(
                abs(x - sx) <= self._slot_epsilon and abs(y - sy) <= self._slot_epsilon
                for (sx, sy) in slots
            )
            if not is_dup and len(slots) < self.max_slots:
                slots.append((x, y))
        self._turret_slots = np.array(slots, dtype=np.float32).reshape(-1, 2)
        if len(slots) == 0:
            logger.warning(
                "No turrets found on map and no turret_slot_positions configured; "
                "the agent will have no slots to act on."
            )

    def _read_anchor_slots(self) -> None:
        """Scan the map for `hazard-concrete-left` tiles and record their centers.

        Read once on first reset (tiles are static). Each tile is a 1x1 cell
        whose position is its top-left corner; we store the tile center
        (corner + 0.5) so the value is a valid teleport target. The result is a
        canonical (N, 2) array, mirroring `_turret_slots`.
        """
        r = self.config.anchor_scan_radius
        cx, cy = self._center_x, self._center_y
        anchors: list = []
        try:
            raw = self.instance.rcon_client.send_command(
                "/sc local out={} "
                "for _,t in pairs(game.surfaces[1].find_tiles_filtered{"
                "name='hazard-concrete-left', area={"
                f"{{{cx - r},{cy - r}}},{{{cx + r},{cy + r}}}"
                "}}) do out[#out+1]=t.position.x..','..t.position.y end "
                "rcon.print(table.concat(out, ';'))"
            ).strip()
            if raw:
                for pair in raw.split(";"):
                    if not pair:
                        continue
                    tx, ty = pair.split(",")
                    # Tile position is the top-left corner; store the center.
                    anchors.append((float(tx) + 0.5, float(ty) + 0.5))
                    if len(anchors) >= self.max_anchors:
                        break
        except Exception as e:
            logger.warning(f"Failed to read anchor tiles: {e}")
        self._anchor_slots = np.array(anchors, dtype=np.float32).reshape(-1, 2)
        if len(anchors) == 0:
            logger.warning(
                "No 'hazard-concrete-left' anchor tiles found on map; the agent "
                "will have no anchor positions to move to."
            )

    def _build_anchor_reach_matrix(self) -> None:
        """Precompute which anchor positions can reach which turret slots.

        Iterates over all (anchor, slot) pairs once and records whether the
        Euclidean distance is within REACH_DISTANCE. The resulting
        (MAX_ANCHORS, MAX_SLOTS) matrix is static for the lifetime of the env
        and is sliced every observation to produce `reach_slot_mask`.
        """
        mat = np.zeros((MAX_ANCHORS, MAX_SLOTS), dtype=np.int8)
        na = 0 if self._anchor_slots is None else len(self._anchor_slots)
        ns = 0 if self._turret_slots is None else len(self._turret_slots)
        for i in range(min(na, MAX_ANCHORS)):
            ax = float(self._anchor_slots[i, 0])
            ay = float(self._anchor_slots[i, 1])
            for j in range(min(ns, MAX_SLOTS)):
                sx = float(self._turret_slots[j, 0])
                sy = float(self._turret_slots[j, 1])
                if math.hypot(ax - sx, ay - sy) <= REACH_DISTANCE:
                    mat[i, j] = 1
        self._anchor_reach_matrix = mat

    def _report_save_state(self) -> None:
        """Print a one-time save validation report to stdout on first reset.

        Covers every entity the scenario depends on (including those without a
        Python Prototype such as infinity-pipe / infinity-chest), the radar_view
        grid channel occupancy, slot/anchor geometry, reach matrix, and obs-space
        conformance. Intended to be read before training starts.
        """
        rc = self.instance.rcon_client
        lines: list = ["", "=" * 60, "  TowerDefenseEnv — save validation report", "=" * 60]

        def rcon_int(cmd: str, default: int = 0) -> int:
            try:
                return int(float(rc.send_command(cmd).strip()))
            except Exception:
                return default

        def ok(val, good: bool) -> str:
            return f"{val:>6}  {'[OK]' if good else '[WARN]'}"

        # --- Entity census (direct RCON, sees ALL entities regardless of Prototype) ---
        # "unit-spawner" is an entity TYPE (biter nests: biter-spawner, spitter-spawner,
        # etc.) — must be queried with {type=...}. Everything else is queried by name.
        lines.append("  Entities on surface:")
        all_ok = True

        # Entries: (display_label, rcon_filter_fragment, min_count, required)
        # rcon_filter_fragment is inserted verbatim into find_entities_filtered{...}
        checks = [
            ("radar",            "name='radar'",           1, True),
            ("boiler",           "name='boiler'",          1, True),
            ("gun-turret",       "name='gun-turret'",      1, True),
            ("stone-wall",       "name='stone-wall'",      0, False),
            ("unit-spawner",     "type='unit-spawner'",    1, False),  # biter nests (ok if absent — place in map editor)
            ("infinity-chest",   "name='infinity-chest'",  1, True),
            ("infinity-pipe",    "name='infinity-pipe'",   1, True),
        ]
        for label, flt, min_count, required in checks:
            n = rcon_int(
                f"/sc local n=0 for _,e in pairs(game.surfaces[1]"
                f".find_entities_filtered{{{flt}}}) do n=n+1 end rcon.print(n)"
            )
            good = n >= min_count
            if required and not good:
                all_ok = False
            marker = "[OK]" if good else ("[WARN — missing!]" if required else "[WARN]")
            lines.append(f"    {label:<20} {n:>4}   {marker}")

        # Evolution factor
        evo = 0.0
        try:
            evo = float(rc.send_command(
                "/sc rcon.print(game.forces['enemy'].get_evolution_factor(game.surfaces[1]))"
            ).strip())
        except Exception:
            pass
        lines.append(f"    {'evolution factor':<20} {evo:.4f}")
        evo_target = (
            "save default" if self.config.evolution_factor is None
            else f"{self.config.evolution_factor:.4f}"
        )
        group_target = (
            "engine default" if self.config.max_unit_group_size is None
            else str(self.config.max_unit_group_size)
        )
        lines.append(f"    {'evolution target':<20} {evo_target}")
        lines.append(f"    {'max group size':<20} {group_target}")

        # --- Grid channel occupancy (radar_view) ---------------------------------
        lines.append("  Grid channels (radar_view):")
        channel_names = [
            "0 empty", "1 wall", "2 turret", "3 ammo%",
            "4 biter", "5 spitter", "6 spawner", "7 character",
        ]
        try:
            grid = self.instance.first_namespace._radar_view(
                center_x=self._center_x,
                center_y=self._center_y,
                radius=int(self._radius),
                cell_size=self.cell_size,
                charted_only=True,
            )
            for ch, ch_name in enumerate(channel_names):
                cells = int(np.count_nonzero(grid[ch]))
                total = grid.shape[1] * grid.shape[2]
                lines.append(f"    ch{ch_name:<12}  {cells:>5} / {total} cells non-zero")
        except Exception as e:
            lines.append(f"    [radar_view failed: {e}]")

        # --- Slot / anchor / reach geometry -------------------------------------
        n_slots = 0 if self._turret_slots is None else len(self._turret_slots)
        n_anchors = 0 if self._anchor_slots is None else len(self._anchor_slots)
        n_boilers = 0 if self._boiler_slots is None else len(self._boiler_slots)
        lines.append(f"  Turret slots  : {n_slots} (cap {self.max_slots})")
        lines.append(f"  Anchor tiles  : {n_anchors} (cap {self.max_anchors})")
        lines.append(f"  Boilers       : {n_boilers} (cap {self.max_boilers})")

        if self._anchor_reach_matrix is not None and n_slots > 0 and n_anchors > 0:
            mat = self._anchor_reach_matrix
            reachable_slots = sum(
                1 for j in range(n_slots)
                if any(mat[i, j] for i in range(n_anchors))
            )
            orphan_slots = n_slots - reachable_slots
            lines.append(
                f"  Reach matrix  : {reachable_slots}/{n_slots} slots reachable from ≥1 anchor"
                + ("  [OK]" if orphan_slots == 0 else f"  [WARN — {orphan_slots} orphan slot(s)]")
            )
            if orphan_slots:
                all_ok = False
        else:
            lines.append("  Reach matrix  : not built")

        # --- Observation space conformance --------------------------------------
        try:
            obs_space = self.observation_space
            sample = obs_space.sample()
            obs_ok = obs_space.contains(sample)
            lines.append(f"  Obs space sample conformance: {'[OK]' if obs_ok else '[FAIL]'}")
        except Exception as e:
            lines.append(f"  Obs space check: [FAIL] {e}")
            all_ok = False

        # --- Vector env compatibility (spaces are serialisable) -----------------
        try:
            import pickle
            pickle.dumps(self.observation_space)
            pickle.dumps(self.action_space)
            lines.append("  VecEnv pickle  : [OK]  (spaces are serialisable)")
        except Exception as e:
            lines.append(f"  VecEnv pickle  : [FAIL] {e}")
            all_ok = False

        lines.append("=" * 60)
        lines.append(f"  Overall: {'READY' if all_ok else 'WARNINGS — check above'}")
        lines.append("=" * 60)
        lines.append("")
        print("\n".join(lines))

    def _delete_turret_subset(self) -> None:
        """Destroy a random subset of slot turrets and give them to the agent.

        The removed turrets are inserted into the agent's inventory so the
        agent has exactly the right number of turrets to refill empty slots.
        All slot positions (including emptied ones) remain in _turret_slots,
        so the model always knows the full layout.
        """
        if self._turret_slots is None or len(self._turret_slots) == 0:
            return
        n = len(self._turret_slots)
        pct = max(0.0, min(1.0, self.config.turret_deletion_percentage))
        n_delete = int(np.floor(n * pct))
        if n_delete <= 0:
            return
        idx = self.np_random.choice(n, size=n_delete, replace=False)
        positions = [
            (float(self._turret_slots[i][0]), float(self._turret_slots[i][1]))
            for i in idx
        ]
        # Destroy each selected turret via RCON (find by position + small radius).
        positions_lua = (
            "{"
            + ",".join(f"{{{px},{py}}}" for (px, py) in positions)
            + "}"
        )
        result = self.instance.rcon_client.send_command(
            f"/sc local pos={positions_lua}; local n=0;"
            f"for _,p in ipairs(pos) do "
            f"  local e=game.surfaces[1].find_entities_filtered"
            f"{{name='gun-turret', position=p, radius=0.7}}[1];"
            f"  if e then e.destroy(); n=n+1 end "
            f"end rcon.print(n)"
        ).strip()
        try:
            n_destroyed = int(float(result))
        except (ValueError, TypeError):
            n_destroyed = n_delete
            logger.warning(f"Could not parse turret-destroy count: {result!r}")
        if n_destroyed != n_delete:
            logger.warning(
                f"_delete_turret_subset: expected to destroy {n_delete}, "
                f"actually destroyed {n_destroyed}"
            )
        # Give the removed turrets to the agent so it can re-place them.
        if n_destroyed > 0:
            self.instance.rcon_client.send_command(
                f"/sc local c=storage.agent_characters[1];"
                f"if c and c.valid then"
                f"  c.insert{{name='gun-turret', count={n_destroyed}}}"
                f"end"
            )

    def _slot_occupancy(self, turrets: list) -> Tuple[np.ndarray, np.ndarray]:
        """Match canonical slots against live turrets.

        Returns (occupied, slot_features) where occupied is a bool array of
        length N and slot_features is (N, 3) of [occupied(0/1), ammo, health].
        """
        n = 0 if self._turret_slots is None else len(self._turret_slots)
        occupied = np.zeros(n, dtype=bool)
        features = np.zeros((n, 3), dtype=np.float32)
        eps = self._slot_epsilon
        for i in range(n):
            sx, sy = self._turret_slots[i]
            for (tx, ty, ammo, health) in turrets:
                if abs(tx - sx) <= eps and abs(ty - sy) <= eps:
                    occupied[i] = True
                    features[i] = [1.0, ammo, health]
                    break
        return occupied, features

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Resilient wrapper around _step_impl.

        If RCON drops mid-step (human client joined to spectate, or the server
        crashed and Docker restarted it), reconnect and end the episode cleanly
        so SB3 calls reset() rather than the whole training run dying.
        """
        try:
            obs, reward, terminated, truncated, info = self._step_impl(action)
            self._last_obs = obs
            return obs, reward, terminated, truncated, info
        except _RCONNetworkError as e:
            logger.warning(
                "[step %d] RCON dropped (%s) — reconnecting and ending episode",
                self._step_count, e,
            )
            try:
                self.instance.reconnect_rcon(pause_after=True)
                logger.warning("[step %d] RCON reconnected; episode terminated", self._step_count)
            except Exception as re:
                logger.error("[step %d] RCON reconnect FAILED: %s", self._step_count, re)
            obs = self._last_obs if self._last_obs is not None else self._zero_obs()
            info = {
                "step": self._step_count,
                "elapsed_ticks": -1,
                "wave": self._wave_number,
                "invalid_action": False,
                "is_moving": False,
                "kills": 0, "turrets_lost": 0, "walls_lost": 0,
                "buildings_lost": 0, "boilers_lost": 0,
                "coverage": self._cur_coverage,
                "rcon_dropped": True,
            }
            # Terminate so SB3 resets; penalize like a death (state was lost).
            return obs, self.config.terminal_penalty, True, False, info

    def _zero_obs(self) -> Dict[str, np.ndarray]:
        """A valid all-zeros observation, used when no prior obs is cached."""
        return {k: np.zeros(space.shape, dtype=space.dtype)
                for k, space in self.observation_space.spaces.items()}

    def _step_impl(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        self._total_steps += 1

        t_step_start = _time_module.perf_counter()
        logger.debug("[step %d] ── BEGIN ──────────────────────────────", self._step_count)

        # Execute action
        t_act = _time_module.perf_counter()
        invalid = self._execute_action(action)
        logger.debug("[step %d] action=%s  invalid=%s  took=%.3fs",
                     self._step_count, self._action_name(action), invalid,
                     _time_module.perf_counter() - t_act)

        # Unpause, let game run for decision_cadence ticks, then pause.
        # If RCON drops while the game is running (e.g. a human client joins),
        # reconnect_rcon() immediately pauses the game to cap extra elapsed ticks.
        try:
            from factorio_rcon.factorio_rcon import RCONClosed
        except ImportError:
            RCONClosed = Exception  # pragma: no cover

        t0 = _time_module.perf_counter()
        self.instance.unpause()
        sleep_seconds = self.decision_cadence / 60.0 / self.game_speed
        logger.debug(
            "[step %d] UNPAUSE — cadence=%d ticks, sleeping %.3fs (game_speed=%.1fx)",
            self._step_count, self.decision_cadence, sleep_seconds, self.game_speed,
        )
        if sleep_seconds > 0:
            _time_module.sleep(sleep_seconds)
        try:
            self.instance.pause()
        except RCONClosed:
            logger.warning(
                "RCON dropped during step (game was running); "
                "reconnecting and pausing — some ticks may have elapsed."
            )
            self.instance.reconnect_rcon(pause_after=True)
        t1 = _time_module.perf_counter()

        elapsed_ticks = self.instance.get_elapsed_ticks()
        logger.debug(
            "[step %d] PAUSE   — elapsed_ticks=%d  wall=%.3fs",
            self._step_count, elapsed_ticks, t1 - t0,
        )

        # Poll the async walker: it advanced the character during the unpaused
        # window. Update transit state; snap the current anchor on arrival so the
        # reach mask re-enables turret actions for the reached anchor.
        moving, target = self._read_walk_state()
        self._moving = moving
        if moving:
            self._move_target = target
        elif self._move_target >= 0:
            self._current_anchor_index = self._move_target
            self._move_target = -1

        # Read and clear TD reward/loss events immediately after the game has
        # advanced. The current observation includes these directional losses,
        # and reward/termination use the same event window.
        ev = self._read_events()
        self._last_events = ev
        self._accumulate_episode_events(ev)

        # Get observation
        t_obs = _time_module.perf_counter()
        obs = self._get_observation()
        logger.debug("[step %d] OBS     — char_hp=%.1f  radar_hp=%.1f  "
                     "coverage=%.2f  threats=%d  took=%.3fs",
                     self._step_count,
                     self._cur_char_hp, self._cur_radar_hp,
                     self._cur_coverage,
                     int(obs["group_valid_mask"].sum()),
                     _time_module.perf_counter() - t_obs)

        # Compute reward
        t_rew = _time_module.perf_counter()
        reward = self._compute_reward(invalid, action)
        logger.debug("[step %d] REWARD  — r=%.4f  took=%.3fs",
                     self._step_count, reward, _time_module.perf_counter() - t_rew)

        # Check termination
        terminated = self._check_terminated(obs)
        truncated = elapsed_ticks >= self.max_ticks

        if terminated:
            reward += self.config.terminal_penalty
        elif truncated:
            reward += self.config.terminal_bonus

        ev = getattr(self, "_last_events", {}) or {}
        episode_ev = getattr(self, "_episode_events", {}) or {}
        action_type = int(action.get("action_type", ACTION_NOOP))
        action_names = {
            ACTION_NOOP: "noop",
            ACTION_PICK_TURRET: "pick",
            ACTION_PLACE_TURRET: "place",
            ACTION_REFILL_TURRET: "refill",
            ACTION_MOVE_ANCHOR: "move",
            ACTION_TAKE_AMMO: "take_ammo",
        }
        info = {
            "step": self._step_count,
            "elapsed_ticks": elapsed_ticks,
            "wave": self._wave_number,
            "invalid_action": invalid,
            "is_moving": self._moving,
            "kills": ev.get("kills", 0),
            "turrets_lost": ev.get("turrets_lost", 0),
            "walls_lost": ev.get("walls_lost", 0),
            "buildings_lost": ev.get("buildings_lost", 0),
            "boilers_lost": ev.get("boilers_lost", 0),
            "walls_lost_step": ev.get("walls_lost", 0),
            "turrets_lost_step": ev.get("turrets_lost", 0),
            "boilers_lost_step": ev.get("boilers_lost", 0),
            "walls_lost_episode": episode_ev.get("walls_lost", 0),
            "turrets_lost_episode": episode_ev.get("turrets_lost", 0),
            "boilers_lost_episode": episode_ev.get("boilers_lost", 0),
            "action_type_id": action_type,
            "action_name": action_names.get(action_type, "unknown"),
            "coverage": self._cur_coverage,
            "empty_turrets": self._cur_empty_turrets,
            "threatened_turret_readiness": self._cur_threatened_turret_readiness,
            "early_setup_reward": self._last_early_setup_reward,
            "ammo_taken": self._last_ammo_taken,
        }
        for key in (
            "wall_n", "wall_e", "wall_s", "wall_w",
            "turret_n", "turret_e", "turret_s", "turret_w",
        ):
            info[key] = ev.get(key, 0)
            info[f"{key}_episode"] = episode_ev.get(key, 0)

        # ── per-step summary line (INFO so it shows in normal runs) ──────────
        logger.info(
            "step %4d | ticks=%6d | act=%-14s | r=%+7.3f | "
            "char=%.0f  radar=%.0f  cov=%.0f%%  kills=%d  wall_lost=%d  "
            "moving=%s  invalid=%s  wall=%.2fs",
            self._step_count, elapsed_ticks,
            self._action_name(action),
            reward,
            self._cur_char_hp, self._cur_radar_hp,
            self._cur_coverage * 100,
            ev.get("kills", 0), ev.get("walls_lost", 0),
            "Y" if self._moving else "N",
            "Y" if invalid else "N",
            _time_module.perf_counter() - t_step_start,
        )

        if terminated or truncated:
            logger.info("──── EPISODE END ──── terminated=%s truncated=%s "
                        "ticks=%d  total_reward_contribution=%.3f",
                        terminated, truncated, elapsed_ticks, reward)

        return obs, reward, terminated, truncated, info

    @staticmethod
    def _action_name(action: Dict[str, Any]) -> str:
        """Human-readable action label for logging."""
        _NAMES = {
            ACTION_NOOP: "NOOP",
            ACTION_PICK_TURRET: "PICK_TURRET",
            ACTION_PLACE_TURRET: "PLACE_TURRET",
            ACTION_REFILL_TURRET: "REFILL_TURRET",
            ACTION_MOVE_ANCHOR: "MOVE_ANCHOR",
            ACTION_TAKE_AMMO: "TAKE_AMMO",
        }
        atype = int(action.get("action_type", ACTION_NOOP))
        name = _NAMES.get(atype, f"?({atype})")
        slot = int(action.get("slot_index", 0))
        anchor = int(action.get("anchor_index", 0))
        ammo = int(action.get("ammo_amount", 0))
        if atype == ACTION_MOVE_ANCHOR:
            return f"{name}[{anchor}]"
        if atype in (ACTION_PLACE_TURRET, ACTION_PICK_TURRET):
            return f"{name}[s{slot}]"
        if atype in (ACTION_REFILL_TURRET, ACTION_TAKE_AMMO):
            return f"{name}[s{slot},a{ammo}]"
        return name

    def _execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute the given slot-based action. Returns True if invalid."""
        self._last_ammo_taken = 0
        action_type = int(action.get("action_type", ACTION_NOOP))
        slot_index = int(action.get("slot_index", 0))
        ammo_amount = int(action.get("ammo_amount", 0))
        anchor_index = int(action.get("anchor_index", 0))

        if action_type == ACTION_NOOP:
            return False

        # Movement is anchor-indexed (not slot-indexed); handle it before the
        # turret-slot resolution below. Re-routing mid-transit is always allowed.
        if action_type == ACTION_MOVE_ANCHOR:
            return self._move_to_anchor(anchor_index)

        # While walking, the only legal actions are re-route (MOVE_ANCHOR) or
        # NOOP — you cannot service a turret while in transit. Enforce it here so
        # the rule holds even without the action-mask wrapper.
        if self._moving:
            return True

        # Resolve the slot index to a real slot center.
        n = 0 if self._turret_slots is None else len(self._turret_slots)
        if slot_index < 0 or slot_index >= n:
            return True  # padding / out-of-range slot
        slot_x = float(self._turret_slots[slot_index][0])
        slot_y = float(self._turret_slots[slot_index][1])

        ns = self.instance.first_namespace
        # Is there a live turret in this slot right now?
        turret_here = self._find_turret_at(slot_x, slot_y)

        try:
            if action_type == ACTION_PICK_TURRET:
                if turret_here is None:
                    return True  # empty slot, nothing to pick up
                # Mine the turret (and any contained ammo) back into the agent's
                # inventory so it can be re-placed into another slot later.
                ns.pickup_entity(turret_here)
                return False

            elif action_type == ACTION_PLACE_TURRET:
                if turret_here is not None:
                    return True  # slot already occupied
                ns.place_entity(
                    Prototype.GunTurret,
                    Direction.UP,
                    Position(x=slot_x, y=slot_y),
                )
                return False

            elif action_type == ACTION_REFILL_TURRET:
                if turret_here is None:
                    return True  # empty slot, nothing to refill
                amount = max(1, ammo_amount)
                ns.insert_item(
                    self._ammo_prototype,
                    turret_here,
                    amount,
                )
                return False

            elif action_type == ACTION_TAKE_AMMO:
                if turret_here is None:
                    return True  # empty slot, nothing to take from
                amount = max(1, ammo_amount)
                taken = self._take_configured_ammo_from_turret(
                    slot_x,
                    slot_y,
                    amount,
                )
                if taken <= 0:
                    return True
                self._last_ammo_taken = taken
                return False

            else:
                return True  # Unknown action type

        except Exception as e:
            logger.debug(f"Action failed: {e}")
            return True

    def _take_configured_ammo_from_turret(
        self,
        slot_x: float,
        slot_y: float,
        amount: int,
    ) -> int:
        """Move configured ammo from the selected turret to the agent inventory."""
        ammo_name = self.config.ammo_type.replace("\\", "\\\\").replace("'", "\\'")
        amount = max(1, int(amount))
        raw = self.instance.rcon_client.send_command(
            f"/sc local s=game.surfaces[1] "
            f"local item='{ammo_name}' "
            f"local inserted=0 "
            f"local e=s.find_entities_filtered{{"
            f"name='gun-turret', position={{{slot_x},{slot_y}}}, "
            f"radius={float(self._slot_epsilon)}}}[1] "
            f"if e and e.valid then "
            f"  local inv=e.get_inventory(defines.inventory.turret_ammo) "
            f"  if inv then "
            f"    local available=inv.get_item_count(item) "
            f"    local take=math.min({amount}, available) "
            f"    if take > 0 then "
            f"      local removed=inv.remove{{name=item,count=take}} "
            f"      if removed > 0 then "
            f"        local c=storage.agent_characters and storage.agent_characters[1] "
            f"        if c and c.valid then inserted=c.insert{{name=item,count=removed}} end "
            f"        if inserted < removed then "
            f"          inv.insert{{name=item,count=removed-inserted}} "
            f"        end "
            f"      end "
            f"    end "
            f"  end "
            f"end "
            f"rcon.print(inserted)"
        ).strip()
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            logger.debug("Could not parse ammo-take count: %r", raw)
            return 0

    def _find_turret_at(self, x: float, y: float):
        """Return a GunTurret entity at slot (x, y), or None.

        Filters by entity type so only ≤1 gun-turret is serialised over RCON
        (instead of the full 500+ entity list that get_entities returns without
        a filter, which causes the step to hang on maps with many walls).
        """
        ns = self.instance.first_namespace
        try:
            entities = ns.get_entities(
                entities={Prototype.GunTurret},
                position=Position(x=x, y=y),
                radius=self._slot_epsilon,
            )
            for e in entities:
                if getattr(e, "name", None) == "gun-turret":
                    return e
        except Exception:
            pass
        return None

    def _move_to_anchor(self, anchor_index: int) -> bool:
        """Start an asynchronous walk to the chosen anchor tile.

        Hands a path to the scenario's on_tick walker (via the walk_to tool); the
        character then travels over the next decision window(s) while biters keep
        attacking. Does NOT block. Returns True if the anchor index is invalid.
        Issuing this while already walking simply re-routes to the new anchor.
        """
        n = 0 if self._anchor_slots is None else len(self._anchor_slots)
        if anchor_index < 0 or anchor_index >= n:
            return True  # padding / out-of-range anchor
        if self._moving and anchor_index == self._move_target:
            return True  # already moving there
        if not self._moving and anchor_index == self._current_anchor_index:
            return True  # already standing there
        ax = float(self._anchor_slots[anchor_index][0])
        ay = float(self._anchor_slots[anchor_index][1])
        try:
            cx, cy = self._get_char_pos()
            waypoints = self._plan_path(Position(x=cx, y=cy), Position(x=ax, y=ay))
            self.instance.first_namespace.walk_to(
                waypoints, anchor_index, speed=self.config.walk_speed
            )
            self._moving = True
            self._move_target = anchor_index
            return False
        except Exception as e:
            logger.debug(f"Move-to-anchor failed: {e}")
            return True

    def _nearest_anchor_index(self) -> int:
        """Return the anchor nearest to the current character position."""
        n = 0 if self._anchor_slots is None else len(self._anchor_slots)
        if n <= 0:
            return 0
        chx, chy = self._get_char_pos()
        best_idx = 0
        best_dist = float("inf")
        for i in range(n):
            ax = float(self._anchor_slots[i, 0])
            ay = float(self._anchor_slots[i, 1])
            dist = math.hypot(ax - chx, ay - chy)
            if dist < best_dist:
                best_idx = i
                best_dist = dist
        return best_idx

    def _plan_path(self, start: Position, finish: Position) -> list:
        """Return an ordered waypoint list from start to finish.

        Tries Factorio A* (request_path/get_path) for obstacle-aware routing and
        falls back to a straight line. The teleport-based walker tolerates either,
        so the straight line is a safe fallback (e.g. when the pathfinder is busy
        or the game is paused). Swapping in richer routing later is transparent.
        """
        ns = self.instance.first_namespace
        was_paused = True
        try:
            try:
                was_paused = self.instance.game_control.is_paused()
            except Exception:
                was_paused = True
            if was_paused:
                self.instance.unpause()
            handle = ns._request_path(
                start, finish, allow_paths_through_own_entities=True, resolution=-1
            )
            waypoints = ns._get_path(handle, max_attempts=4)
            if waypoints:
                return waypoints
        except Exception as e:
            logger.debug(f"A* path failed ({e}); using straight-line walk")
        finally:
            if was_paused:
                try:
                    self.instance.pause()
                except Exception:
                    pass
        return [finish]

    def _get_char_pos(self) -> Tuple[float, float]:
        """Read the agent character's live (x, y) from the game."""
        try:
            raw = self.instance.rcon_client.send_command(
                "/sc local c=storage.agent_characters and storage.agent_characters[1]; "
                "if c and c.valid then rcon.print(c.position.x..','..c.position.y) "
                "else rcon.print('nil') end"
            ).strip()
            if raw and raw != "nil":
                x, y = raw.split(",")
                return float(x), float(y)
        except Exception:
            pass
        return self._center_x, self._center_y

    def _read_walk_state(self) -> Tuple[bool, int]:
        """Return (is_moving, target_anchor) from the scenario walk controller."""
        try:
            raw = self.instance.rcon_client.send_command(
                "/sc local w=storage.td_walk; "
                "if w and w.active then rcon.print(tostring(w.target_anchor)) "
                "else rcon.print('done') end"
            ).strip()
            if raw == "done" or not raw:
                return False, self._move_target
            return True, int(float(raw))
        except Exception:
            return False, self._move_target

    def _read_events(self) -> Dict[str, int]:
        """Read & clear the server-side death/kill counters for this window."""
        try:
            return self.instance.first_namespace._read_td_events()
        except Exception:
            return self._zero_events()

    @staticmethod
    def _zero_events() -> Dict[str, int]:
        return {
            "kills": 0,
            "turrets_lost": 0,
            "walls_lost": 0,
            "buildings_lost": 0,
            "boilers_lost": 0,
            "radar_lost": 0,
            "char_died": 0,
            "wall_n": 0,
            "wall_e": 0,
            "wall_s": 0,
            "wall_w": 0,
            "turret_n": 0,
            "turret_e": 0,
            "turret_s": 0,
            "turret_w": 0,
        }

    def _accumulate_episode_events(self, ev: Dict[str, int]) -> None:
        """Add the latest event window into per-episode totals."""
        if not self._episode_events:
            self._episode_events = self._zero_events()
        for key in self._episode_events:
            self._episode_events[key] = self._episode_events.get(key, 0) + int(
                ev.get(key, 0)
            )

    def _get_radar_hp(self) -> float:
        """Read the radar's current HP directly (0 if no radar / destroyed)."""
        if not self._has_radar:
            return 0.0
        try:
            raw = self.instance.rcon_client.send_command(
                "/sc local e=game.surfaces[1].find_entities_filtered{name='radar'}[1]; "
                "rcon.print(e and e.valid and e.health or 0)"
            ).strip()
            return float(raw)
        except Exception:
            return 0.0

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Build the observation dict from game state.

        All positions are emitted relative to the radar center and normalized by
        the grid radius (self._norm) so the network sees well-conditioned inputs.
        Raw values needed by the reward are stashed on self._cur_* to avoid extra
        RCON round-trips in _compute_reward.
        """
        ns = self.instance.first_namespace
        cx0, cy0 = self._center_x, self._center_y
        norm = self._norm

        # Live character position (the async walker may have moved it).
        chx, chy = self._get_char_pos()
        try:
            ns.player_location = Position(x=chx, y=chy)
        except Exception:
            pass

        # Map grid from radar_view
        try:
            map_grid = ns._radar_view(
                center_x=cx0,
                center_y=cy0,
                radius=int(self._radius),
                cell_size=self.cell_size,
                charted_only=True,
            )
            if not isinstance(map_grid, np.ndarray):
                map_grid = np.zeros(
                    (NUM_CHANNELS, self.grid_size, self.grid_size), dtype=np.uint8
                )
        except Exception:
            map_grid = np.zeros(
                (NUM_CHANNELS, self.grid_size, self.grid_size), dtype=np.uint8
            )

        # Inventory
        try:
            inv = ns.inspect_inventory()
            inv_arr = np.array(
                [inv.get(item, 0) for item in TRACKED_ITEMS], dtype=np.int32
            )
        except Exception:
            inv_arr = np.zeros(len(TRACKED_ITEMS), dtype=np.int32)

        # Turret occupancy — use a targeted RCON query (gun-turrets only) so
        # walls, infinity-pipes, chests, etc. are never serialised over RCON.
        turrets = self._current_turrets()
        boilers = self._current_boilers()

        # Turret slots + masks (positions normalized; reach from LIVE char pos)
        slots_arr = np.zeros((MAX_SLOTS, SLOT_FEATURES), dtype=np.float32)
        slot_valid_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        place_slot_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        refill_slot_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        pick_slot_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        take_ammo_slot_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        reach_slot_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        slots_arr[:, 2] = -1.0  # padding rows: occupied = -1
        live_turrets = 0
        loaded_turrets = 0
        empty_turrets = 0
        n = 0 if self._turret_slots is None else len(self._turret_slots)
        if n > 0:
            occupied, feats = self._slot_occupancy(turrets)
            for i in range(min(n, MAX_SLOTS)):
                sx, sy = self._turret_slots[i]
                occ, ammo, hp = feats[i][0], feats[i][1], feats[i][2]
                slots_arr[i] = [
                    (sx - cx0) / norm, (sy - cy0) / norm,
                    occ, ammo / TURRET_AMMO_NORM, hp / TURRET_HP_NORM,
                ]
                slot_valid_mask[i] = 1
                if math.hypot(sx - chx, sy - chy) <= REACH_DISTANCE:
                    reach_slot_mask[i] = 1
                if occupied[i]:
                    refill_slot_mask[i] = 1
                    pick_slot_mask[i] = 1
                    live_turrets += 1
                    if ammo > 0:
                        take_ammo_slot_mask[i] = 1
                        loaded_turrets += 1
                    else:
                        empty_turrets += 1
                else:
                    place_slot_mask[i] = 1

        # Action-conditioned reach masks. While in transit no turret action is
        # legal, so they are all zero (the policy may only re-route / noop).
        if self._moving:
            place_reach_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
            refill_reach_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
            pick_reach_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
            take_ammo_reach_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        else:
            place_reach_mask = (place_slot_mask & reach_slot_mask).astype(np.int8)
            refill_reach_mask = (refill_slot_mask & reach_slot_mask).astype(np.int8)
            pick_reach_mask = (pick_slot_mask & reach_slot_mask).astype(np.int8)
            take_ammo_reach_mask = (
                take_ammo_slot_mask & reach_slot_mask
            ).astype(np.int8)

        # Anchors (normalized)
        anchors_arr = np.zeros((MAX_ANCHORS, 2), dtype=np.float32)
        anchor_valid_mask = np.zeros(MAX_ANCHORS, dtype=np.int8)
        na = 0 if self._anchor_slots is None else len(self._anchor_slots)
        for i in range(min(na, MAX_ANCHORS)):
            ax, ay = self._anchor_slots[i]
            anchors_arr[i] = [(ax - cx0) / norm, (ay - cy0) / norm]
            anchor_valid_mask[i] = 1

        # Character (normalized)
        char_hp = self._get_character_health()
        ammo_total = float(inv_arr[0] + inv_arr[1]) if len(inv_arr) >= 2 else 0.0
        char_arr = np.array(
            [
                (chx - cx0) / norm, (chy - cy0) / norm,
                char_hp / CHAR_HP_NORM, ammo_total / 500.0,
            ],
            dtype=np.float32,
        )

        # Radar (normalized; the "heart")
        radar_hp = self._get_radar_hp()
        rx = self._radar_position.x if self._radar_position is not None else cx0
        ry = self._radar_position.y if self._radar_position is not None else cy0
        radar_arr = np.array(
            [(rx - cx0) / norm, (ry - cy0) / norm, radar_hp / RADAR_HP_NORM],
            dtype=np.float32,
        )

        # Biter groups (swarms) + nests via threat_view
        (
            groups_arr, group_valid_mask, nests_arr, nest_valid_mask,
            threat_closeness,
        ) = self._build_threat_obs(cx0, cy0, norm, turrets)
        (
            slot_threat_alignment,
            has_threat_direction,
        ) = self._compute_slot_threat_alignment(
            slots_arr,
            slot_valid_mask,
            groups_arr,
            group_valid_mask,
            nests_arr,
            nest_valid_mask,
        )
        threatened_turret_readiness = self._compute_threatened_turret_readiness(
            slots_arr,
            slot_valid_mask,
            groups_arr,
            group_valid_mask,
            nests_arr,
            nest_valid_mask,
        )

        # Boilers (critical power targets).
        boilers_arr = np.zeros((MAX_BOILERS, BOILER_FEATURES), dtype=np.float32)
        boiler_valid_mask = np.zeros(MAX_BOILERS, dtype=np.int8)
        nb = 0 if self._boiler_slots is None else len(self._boiler_slots)
        if nb > 0:
            boiler_alive, boiler_health = self._boiler_status(boilers)
            for i in range(min(nb, MAX_BOILERS)):
                bx, by = self._boiler_slots[i]
                boilers_arr[i] = [
                    (bx - cx0) / norm,
                    (by - cy0) / norm,
                    1.0 if boiler_alive[i] else 0.0,
                    boiler_health[i] / BOILER_HP_NORM,
                ]
                boiler_valid_mask[i] = 1

        # Movement / transit state
        movement_arr = np.zeros(MOVEMENT_FEATURES, dtype=np.float32)
        movement_arr[0] = 1.0 if self._moving else 0.0
        if self._moving and 0 <= self._move_target < na:
            tx, ty = self._anchor_slots[self._move_target]
            rem = math.hypot(tx - chx, ty - chy)
            rmag = rem if rem > 1e-6 else 1.0
            movement_arr[1] = self._move_target / float(MAX_ANCHORS)
            movement_arr[2] = rem / norm
            movement_arr[3] = (tx - chx) / rmag
            movement_arr[4] = (ty - chy) / rmag

        # Directional losses from the decision window that just elapsed.
        ev = getattr(self, "_last_events", {}) or {}
        recent_losses = np.array(
            [
                ev.get("wall_n", 0), ev.get("wall_e", 0),
                ev.get("wall_s", 0), ev.get("wall_w", 0),
                ev.get("turret_n", 0), ev.get("turret_e", 0),
                ev.get("turret_s", 0), ev.get("turret_w", 0),
            ],
            dtype=np.float32,
        )
        recent_losses = np.clip(recent_losses, 0.0, LOSS_COUNT_NORM) / LOSS_COUNT_NORM

        # Game (elapsed normalized to [0, 1] over the episode budget)
        elapsed_ticks = self.instance.get_elapsed_ticks()
        game_arr = np.array(
            [elapsed_ticks / max(1, self.max_ticks), self._wave_number],
            dtype=np.float32,
        )

        # Stash raw values for the reward / termination logic.
        self._cur_char_hp = char_hp
        self._cur_radar_hp = radar_hp
        self._cur_coverage = (live_turrets / n) if n > 0 else 0.0
        self._cur_ammo_frac = (loaded_turrets / live_turrets) if live_turrets > 0 else 0.0
        self._cur_empty_turrets = empty_turrets
        self._cur_threat_closeness = threat_closeness
        self._cur_threatened_turret_readiness = threatened_turret_readiness
        self._cur_slot_threat_alignment = slot_threat_alignment
        self._cur_has_threat_direction = has_threat_direction

        return {
            "map": map_grid,
            "inventory": inv_arr,
            "turret_slots": slots_arr,
            "slot_valid_mask": slot_valid_mask,
            "place_slot_mask": place_slot_mask,
            "refill_slot_mask": refill_slot_mask,
            "pick_slot_mask": pick_slot_mask,
            "take_ammo_slot_mask": take_ammo_slot_mask,
            "place_reach_mask": place_reach_mask,
            "refill_reach_mask": refill_reach_mask,
            "pick_reach_mask": pick_reach_mask,
            "take_ammo_reach_mask": take_ammo_reach_mask,
            "anchors": anchors_arr,
            "anchor_valid_mask": anchor_valid_mask,
            "reach_slot_mask": reach_slot_mask,
            "biter_groups": groups_arr,
            "group_valid_mask": group_valid_mask,
            "nests": nests_arr,
            "nest_valid_mask": nest_valid_mask,
            "boilers": boilers_arr,
            "boiler_valid_mask": boiler_valid_mask,
            "movement": movement_arr,
            "recent_losses": recent_losses,
            "character": char_arr,
            "radar": radar_arr,
            "game": game_arr,
        }

    def _build_threat_obs(self, cx0, cy0, norm, turrets):
        """Query threat_view and build the biter_groups / nests obs arrays.

        Returns (groups_arr, group_valid_mask, nests_arr, nest_valid_mask,
        threat_closeness) where threat_closeness is in [0, 1] (1 = a swarm is on
        top of the base, 0 = none within threat_scan_radius).
        """
        groups_arr = np.zeros((MAX_GROUPS, GROUP_FEATURES), dtype=np.float32)
        group_valid_mask = np.zeros(MAX_GROUPS, dtype=np.int8)
        nests_arr = np.zeros((MAX_NESTS, NEST_FEATURES), dtype=np.float32)
        nest_valid_mask = np.zeros(MAX_NESTS, dtype=np.int8)
        scan_r = float(self.config.threat_scan_radius)
        closeness = 0.0
        try:
            threat = self.instance.first_namespace._threat_view(
                center_x=cx0,
                center_y=cy0,
                radius=scan_r,
                cell=self.config.threat_cell_size,
                charted_only=True,
                max_groups=MAX_GROUPS,
                max_nests=MAX_NESTS,
            )
        except Exception:
            return groups_arr, group_valid_mask, nests_arr, nest_valid_mask, closeness

        min_dist = float("inf")
        for i, g in enumerate(threat.get("groups", [])[:MAX_GROUPS]):
            try:
                gx, gy, count, spread, vx, vy = g
            except (ValueError, TypeError):
                continue
            rcx, rcy = gx - cx0, gy - cy0
            dist = math.hypot(rcx, rcy)
            speed = math.hypot(vx, vy)
            if speed > 1e-6:
                hdx, hdy = vx / speed, vy / speed
                eta = dist / speed
            else:
                hdx, hdy, eta = 0.0, 0.0, 2.0 * ETA_NORM
            d_turret = self._nearest_turret_dist(gx, gy, turrets, scan_r)
            is_swarm = 1.0 if count >= self.config.swarm_threshold else 0.0
            groups_arr[i] = [
                rcx / norm, rcy / norm, count / COUNT_NORM, spread / norm,
                hdx, hdy, dist / norm, d_turret / norm,
                min(eta, 2.0 * ETA_NORM) / ETA_NORM, is_swarm,
            ]
            group_valid_mask[i] = 1
            if dist < min_dist:
                min_dist = dist
        if min_dist < float("inf") and scan_r > 0:
            closeness = max(0.0, min(1.0, 1.0 - min_dist / scan_r))

        for i, nst in enumerate(threat.get("nests", [])[:MAX_NESTS]):
            try:
                nx, ny, nhp = nst
            except (ValueError, TypeError):
                continue
            rnx, rny = nx - cx0, ny - cy0
            dist = math.hypot(rnx, rny)
            dmag = dist if dist > 1e-6 else 1.0
            nests_arr[i] = [
                rnx / norm, rny / norm, nhp / NEST_HP_NORM, dist / norm,
                rnx / dmag, rny / dmag,
            ]
            nest_valid_mask[i] = 1

        n_groups = int(group_valid_mask.sum())
        n_nests = int(nest_valid_mask.sum())
        logger.debug("_build_threat_obs: %d biter group(s), %d nest(s), "
                     "closeness=%.3f", n_groups, n_nests, closeness)
        return groups_arr, group_valid_mask, nests_arr, nest_valid_mask, closeness

    @staticmethod
    def _nearest_turret_dist(gx, gy, turrets, default):
        """Distance from (gx, gy) to the nearest live turret (default if none)."""
        best = float("inf")
        for (tx, ty, _ammo, _hp) in turrets:
            d = math.hypot(tx - gx, ty - gy)
            if d < best:
                best = d
        return best if best < float("inf") else default

    def _compute_threatened_turret_readiness(
        self,
        slots_arr: np.ndarray,
        slot_valid_mask: np.ndarray,
        groups_arr: np.ndarray,
        group_valid_mask: np.ndarray,
        nests_arr: np.ndarray,
        nest_valid_mask: np.ndarray,
    ) -> float:
        """Score armed turret coverage on the side threats are coming from.

        A threat east of the radar should reward occupied, ammo-loaded turret
        slots east of the radar. Live biter groups are preferred; visible nests
        act as a weaker fallback before a marching group is present.
        """
        slot_valid = slot_valid_mask.astype(bool)
        if slots_arr.size == 0 or not np.any(slot_valid):
            return 0.0

        slots = slots_arr[slot_valid]
        slot_xy = slots[:, :2].astype(np.float32)
        slot_mag = np.linalg.norm(slot_xy, axis=1)
        non_center = slot_mag > 1e-6
        if not np.any(non_center):
            return 0.0

        slot_unit = np.zeros_like(slot_xy, dtype=np.float32)
        slot_unit[non_center] = slot_xy[non_center] / slot_mag[non_center, None]
        occupied = np.clip(slots[:, 2], 0.0, 1.0)
        ammo_ready = np.clip(slots[:, 3], 0.0, 1.0)
        # Placement gives partial credit; ammo makes it a real defense.
        slot_readiness = occupied * (0.4 + 0.6 * ammo_ready)

        def readiness_for(side_xy: np.ndarray) -> Optional[float]:
            side_mag = float(np.linalg.norm(side_xy))
            if side_mag <= 1e-6:
                return None
            side_unit = side_xy / side_mag
            side_weights = np.clip(slot_unit @ side_unit, 0.0, 1.0) ** 2
            side_weights[~non_center] = 0.0
            total_weight = float(side_weights.sum())
            if total_weight <= 1e-6:
                return 0.0
            return float(np.dot(side_weights, slot_readiness) / total_weight)

        weighted_score = 0.0
        total_threat_weight = 0.0
        scan_norm = float(self.config.threat_scan_radius) / max(1.0, self._norm)
        group_valid = group_valid_mask.astype(bool)
        for group in groups_arr[group_valid]:
            score = readiness_for(group[:2].astype(np.float32))
            if score is None:
                continue
            dist_norm = max(0.0, float(group[6]))
            closeness = 0.0
            if scan_norm > 1e-6:
                closeness = max(0.0, 1.0 - min(1.0, dist_norm / scan_norm))
            count_weight = max(0.1, float(group[2]))
            swarm_bonus = 1.5 if float(group[9]) > 0.5 else 1.0
            threat_weight = count_weight * (0.25 + closeness) * swarm_bonus
            weighted_score += threat_weight * score
            total_threat_weight += threat_weight

        if total_threat_weight > 1e-6:
            return max(0.0, min(1.0, weighted_score / total_threat_weight))

        nest_valid = nest_valid_mask.astype(bool)
        for nest in nests_arr[nest_valid]:
            score = readiness_for(nest[:2].astype(np.float32))
            if score is None:
                continue
            dist_norm = max(0.0, float(nest[3]))
            threat_weight = 1.0 / (1.0 + dist_norm)
            weighted_score += threat_weight * score
            total_threat_weight += threat_weight

        if total_threat_weight <= 1e-6:
            return 0.0
        return max(0.0, min(1.0, weighted_score / total_threat_weight))

    def _compute_slot_threat_alignment(
        self,
        slots_arr: np.ndarray,
        slot_valid_mask: np.ndarray,
        groups_arr: np.ndarray,
        group_valid_mask: np.ndarray,
        nests_arr: np.ndarray,
        nest_valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """Return per-slot alignment with the currently suspected attack side."""
        alignment = np.ones(MAX_SLOTS, dtype=np.float32)
        slot_valid = slot_valid_mask.astype(bool)
        if slots_arr.size == 0 or not np.any(slot_valid):
            return alignment, False

        threat_vectors = []
        scan_norm = float(self.config.threat_scan_radius) / max(1.0, self._norm)
        group_valid = group_valid_mask.astype(bool)
        for group in groups_arr[group_valid]:
            side_xy = group[:2].astype(np.float32)
            side_mag = float(np.linalg.norm(side_xy))
            if side_mag <= 1e-6:
                continue
            dist_norm = max(0.0, float(group[6]))
            closeness = 0.0
            if scan_norm > 1e-6:
                closeness = max(0.0, 1.0 - min(1.0, dist_norm / scan_norm))
            count_weight = max(0.1, float(group[2]))
            swarm_bonus = 1.5 if float(group[9]) > 0.5 else 1.0
            weight = count_weight * (0.25 + closeness) * swarm_bonus
            threat_vectors.append((side_xy / side_mag, weight))

        if not threat_vectors:
            nest_valid = nest_valid_mask.astype(bool)
            for nest in nests_arr[nest_valid]:
                side_xy = nest[:2].astype(np.float32)
                side_mag = float(np.linalg.norm(side_xy))
                if side_mag <= 1e-6:
                    continue
                dist_norm = max(0.0, float(nest[3]))
                weight = 1.0 / (1.0 + dist_norm)
                threat_vectors.append((side_xy / side_mag, weight))

        total_weight = sum(weight for _unit, weight in threat_vectors)
        if total_weight <= 1e-6:
            return alignment, False

        alignment[:] = 0.0
        for idx in np.flatnonzero(slot_valid):
            slot_xy = slots_arr[idx, :2].astype(np.float32)
            slot_mag = float(np.linalg.norm(slot_xy))
            if slot_mag <= 1e-6:
                continue
            slot_unit = slot_xy / slot_mag
            score = 0.0
            for side_unit, weight in threat_vectors:
                score += weight * (max(0.0, float(np.dot(slot_unit, side_unit))) ** 2)
            alignment[idx] = max(0.0, min(1.0, score / total_weight))

        return alignment, True

    def _get_character_health(self) -> float:
        """Get the agent character's health via storage.agent_characters[1]."""
        try:
            response = self.instance.rcon_client.send_command(
                "/sc local c=storage.agent_characters and storage.agent_characters[1]; "
                "rcon.print(c and c.valid and c.health or 0)"
            )
            return float(response)
        except Exception:
            return 0.0

    def _compute_reward(self, invalid: bool, action: Optional[Dict[str, Any]] = None) -> float:
        """Compute step reward from server-side death events + stashed raw state.

        Uses the already-read death/kill counters accumulated during the last
        decision window, applies destruction, damage, movement, and invalid
        penalties, then adds light delta shaping that anneals to zero over
        cfg.shaping_decay_steps.
        """
        cfg = self.config
        reward = 0.0

        # Survival.
        reward += cfg.alpha_survive

        # Event-based kills + destruction (read-and-cleared before observation).
        ev = getattr(self, "_last_events", None) or self._zero_events()
        reward += cfg.beta_kills * ev.get("kills", 0)
        reward -= cfg.p_wall_destroyed * ev.get("walls_lost", 0)
        reward -= cfg.p_turret_destroyed * ev.get("turrets_lost", 0)
        reward -= cfg.p_building_destroyed * ev.get("buildings_lost", 0)

        # Character damage (raw HP, stashed during _get_observation).
        damage_taken = max(0.0, self._prev_health - self._cur_char_hp)
        self._prev_health = self._cur_char_hp
        reward -= cfg.delta_damage * damage_taken

        # Radar damage (the heart).
        radar_damage = max(0.0, self._prev_radar_hp - self._cur_radar_hp)
        self._prev_radar_hp = self._cur_radar_hp
        reward -= cfg.p_radar_damage * radar_damage

        # Invalid action penalty.
        if invalid:
            reward -= cfg.epsilon_invalid

        # Empty gun-turrets look like defenses but cannot shoot, so penalize each
        # occupied slot that has no configured ammo loaded this step.
        reward -= cfg.p_empty_turret * self._cur_empty_turrets

        # Movement penalties. A move command has a small fixed cost; remaining
        # in transit costs a little each decision window so repeated/long moves
        # do not dominate training.
        action_type = int((action or {}).get("action_type", ACTION_NOOP))
        if not invalid and action_type == ACTION_MOVE_ANCHOR:
            reward -= cfg.p_move_command
        if self._moving:
            reward -= cfg.p_move_transit

        early_setup_reward = 0.0
        if not invalid and self._step_count <= cfg.early_turret_setup_steps:
            if action_type == ACTION_PLACE_TURRET:
                early_setup_reward = cfg.w_early_place_turret
            elif action_type == ACTION_REFILL_TURRET:
                early_setup_reward = cfg.w_early_refill_turret
            if early_setup_reward > 0.0 and self._cur_has_threat_direction:
                slot_index = int((action or {}).get("slot_index", 0))
                if 0 <= slot_index < len(self._cur_slot_threat_alignment):
                    early_setup_reward *= float(self._cur_slot_threat_alignment[slot_index])
                else:
                    early_setup_reward = 0.0
        reward += early_setup_reward
        self._last_early_setup_reward = early_setup_reward

        # Dense shaping, annealed to 0 over training.
        decay = max(0.0, 1.0 - self._total_steps / max(1, cfg.shaping_decay_steps))
        if decay > 0.0:
            coverage_delta = max(0.0, self._cur_coverage - self._prev_coverage)
            ammo_delta = max(0.0, self._cur_ammo_frac - self._prev_ammo_frac)
            threatened_delta = max(
                0.0,
                self._cur_threatened_turret_readiness
                - self._prev_threatened_turret_readiness,
            )
            reward += decay * cfg.w_coverage_delta * coverage_delta
            reward += decay * cfg.w_ammo_delta * ammo_delta
            reward += decay * cfg.w_threatened_turret_delta * threatened_delta
            reward -= decay * cfg.w_threat * self._cur_threat_closeness
        self._prev_coverage = self._cur_coverage
        self._prev_ammo_frac = self._cur_ammo_frac
        self._prev_threatened_turret_readiness = self._cur_threatened_turret_readiness

        return reward

    def _check_terminated(self, obs: Dict[str, np.ndarray]) -> bool:
        """Terminate on radar destruction or character death."""
        if self._step_count <= 1:
            return False
        ev = getattr(self, "_last_events", {}) or {}
        if self._has_radar and (self._cur_radar_hp <= 0 or ev.get("radar_lost", 0) > 0):
            return True
        boiler_valid = obs.get("boiler_valid_mask")
        boiler_rows = obs.get("boilers")
        if boiler_valid is not None and boiler_rows is not None:
            n_boilers = int(np.asarray(boiler_valid).sum())
            if n_boilers > 0:
                alive_count = int(np.count_nonzero(np.asarray(boiler_rows)[:n_boilers, 2] > 0.5))
                lost_count = n_boilers - alive_count
                threshold = max(1, int(math.ceil(n_boilers * self.config.boiler_loss_fraction)))
                if lost_count >= threshold:
                    return True
        if self._cur_char_hp <= 0 or ev.get("char_died", 0) > 0:
            return True
        return False

    def render(self):
        """Render is optional — can be implemented to visualize the map grid."""
        pass

    def close(self):
        """Clean up."""
        try:
            self.instance.cleanup()
        except Exception:
            pass
