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
    MAX_ANCHORS,
    MAX_SLOTS,
    NUM_CHANNELS,
    REACH_DISTANCE,
    SLOT_FEATURES,
    TRACKED_ITEMS,
    make_action_space,
    make_observation_space,
)
from fle.env.game_types import Prototype
from fle.env.entities import Direction, Position

logger = logging.getLogger(__name__)


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

        # Distance (tiles) within which a live turret is considered to occupy a
        # slot, or two read positions are considered the same slot.
        self._slot_epsilon = 0.6

        # Spaces
        self.observation_space = make_observation_space(self.grid_size)
        self.action_space = make_action_space(self.grid_size)

        # Internal state
        self._step_count = 0
        self._prev_kills = 0
        self._prev_health = 0.0
        self._wave_number = 0
        self._center_x = 0.0
        self._center_y = 0.0
        self._radius = self.grid_size * self.cell_size / 2.0

        # Canonical turret slots, read once from the map on first reset.
        # Shape (N, 2): the (x, y) center of every slot. Never changes for a run.
        self._turret_slots: Optional[np.ndarray] = None

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

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        if self._initial_snapshot is None:
            # First reset: server already loaded the save file.
            # Configure the running game, then capture the snapshot.
            self.instance.set_speed(self.game_speed)
            self.instance.pause()
            self.instance._reset_elapsed_ticks()

            self._read_radar()
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

            # Capture the snapshot with ALL slot turrets present, so each episode
            # restores the full set before re-rolling which slots start empty.
            self._capture_save_state()

        else:
            # Subsequent resets: wipe and restore from snapshot.
            self.instance.reset(game_state=self._initial_snapshot)
            self._restore_evolution_factor()
            self.instance.set_speed(self.game_speed)
            self.instance.pause()
            self.instance._reset_elapsed_ticks()

            self._read_radar()
            self._spawn_player()
            if self.config.clear_biters_on_reset:
                self._clear_live_biters()

        # Re-roll which slots start empty for this episode (seeded by reset seed).
        self._delete_turret_subset()

        self._step_count = 0
        self._prev_kills = 0
        self._prev_health = self._get_character_health()
        self._wave_number = 0
        self._current_anchor_index = 0

        obs = self._get_observation()
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

    def _restore_evolution_factor(self) -> None:
        self.instance.rcon_client.send_command(
            f"/sc game.forces['enemy'].set_evolution_factor(game.surfaces[1], {self._initial_evolution_factor})"
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
        """Return current gun-turrets as a list of (x, y, ammo, health)."""
        ns = self.instance.first_namespace
        turrets = []
        try:
            entities = ns.get_entities(
                position=Position(x=self._center_x, y=self._center_y),
                radius=self._radius,
            )
            for e in entities:
                if e.name == "gun-turret":
                    turrets.append(
                        (
                            float(e.position.x),
                            float(e.position.y),
                            float(getattr(e, "ammo_count", 0) or 0),
                            float(getattr(e, "health", 0) or 0),
                        )
                    )
        except Exception:
            pass
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
        self._step_count += 1

        # Execute action
        invalid = self._execute_action(action)

        # Unpause, let game run for decision_cadence ticks, then pause.
        # If RCON drops while the game is running (e.g. a human client joins),
        # reconnect_rcon() immediately pauses the game to cap extra elapsed ticks.
        try:
            from factorio_rcon.factorio_rcon import RCONClosed
        except ImportError:
            RCONClosed = Exception  # pragma: no cover

        self.instance.unpause()
        sleep_seconds = self.decision_cadence / 60.0 / self.game_speed
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

        elapsed_ticks = self.instance.get_elapsed_ticks()

        # --- Runtime wave triggering (DISABLED) ---------------------------------
        # The map already ships with baked-in biter nests that spawn and path to
        # the base on their own, so we don't drive waves from Python for now.
        # Kept here (and in _spawn_wave / biter_director) for later development.
        #
        # new_wave = int(elapsed_ticks / self.config.ticks_per_wave)
        # if new_wave > self._wave_number:
        #     self._wave_number = new_wave
        #     self._spawn_wave()
        # -----------------------------------------------------------------------

        # Get observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward(invalid)

        # Check termination
        terminated = self._check_terminated(obs)
        truncated = elapsed_ticks >= self.max_ticks

        if terminated:
            reward += self.config.terminal_penalty
        elif truncated:
            reward += self.config.terminal_bonus

        info = {
            "step": self._step_count,
            "elapsed_ticks": elapsed_ticks,
            "wave": self._wave_number,
            "invalid_action": invalid,
        }

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute the given slot-based action. Returns True if invalid."""
        action_type = int(action.get("action_type", ACTION_NOOP))
        slot_index = int(action.get("slot_index", 0))
        ammo_amount = int(action.get("ammo_amount", 0))
        anchor_index = int(action.get("anchor_index", 0))

        if action_type == ACTION_NOOP:
            return False

        # Movement is anchor-indexed (not slot-indexed); handle it before the
        # turret-slot resolution below.
        if action_type == ACTION_MOVE_ANCHOR:
            return self._move_to_anchor(anchor_index)

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
                    Prototype.FirearmMagazine,
                    turret_here,
                    amount,
                )
                return False

            else:
                return True  # Unknown action type

        except Exception as e:
            logger.debug(f"Action failed: {e}")
            return True

    def _find_turret_at(self, x: float, y: float):
        """Return the gun-turret entity occupying slot (x, y), or None."""
        ns = self.instance.first_namespace
        try:
            entities = ns.get_entities(
                position=Position(x=x, y=y),
                radius=self._slot_epsilon,
            )
            for e in entities:
                if e.name == "gun-turret":
                    return e
        except Exception:
            pass
        return None

    def _move_to_anchor(self, anchor_index: int) -> bool:
        """Teleport the character onto the chosen anchor tile. Returns True if invalid."""
        n = 0 if self._anchor_slots is None else len(self._anchor_slots)
        if anchor_index < 0 or anchor_index >= n:
            return True  # padding / out-of-range anchor
        ax = float(self._anchor_slots[anchor_index][0])
        ay = float(self._anchor_slots[anchor_index][1])
        try:
            self.instance.rcon_client.send_command(
                f"/sc storage.agent_characters[1].teleport({{{ax},{ay}}})"
            )
            self.instance.first_namespace.player_location = Position(x=ax, y=ay)
            self._current_anchor_index = anchor_index
            return False
        except Exception as e:
            logger.debug(f"Move-to-anchor failed: {e}")
            return True

    def _spawn_wave(self):
        """Spawn a wave of enemies using biter_director."""
        if not self.config.spawn_waves_at_runtime:
            return
        try:
            self.instance.first_namespace._biter_director(
                wave_number=self._wave_number,
                center_x=self._center_x,
                center_y=self._center_y,
                spawn_radius=self._radius * 1.5,
                base_count=self.config.base_enemy_count,
                escalation_factor=self.config.escalation_factor,
                use_spawners=self.config.spawn_from_map_spawners,
                target_x=self._center_x,
                target_y=self._center_y,
            )
        except Exception as e:
            logger.warning(f"Failed to spawn wave {self._wave_number}: {e}")

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Build the observation dict from game state."""
        ns = self.instance.first_namespace

        # Map grid from radar_view
        try:
            map_grid = ns._radar_view(
                center_x=self._center_x,
                center_y=self._center_y,
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

        # Entities (used for turret occupancy and the radar lookup)
        entities = []
        try:
            entities = ns.get_entities(
                position=Position(x=self._center_x, y=self._center_y),
                radius=self._radius,
            )
        except Exception:
            pass
        turrets = [
            (
                float(e.position.x),
                float(e.position.y),
                float(getattr(e, "ammo_count", 0) or 0),
                float(getattr(e, "health", 0) or 0),
            )
            for e in entities
            if getattr(e, "name", None) == "gun-turret"
        ]

        # Turret slots + masks
        slots_arr = np.zeros((MAX_SLOTS, SLOT_FEATURES), dtype=np.float32)
        slot_valid_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        place_slot_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        refill_slot_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        pick_slot_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        # Mark padding rows as occupied = -1
        slots_arr[:, 2] = -1.0
        n = 0 if self._turret_slots is None else len(self._turret_slots)
        if n > 0:
            occupied, feats = self._slot_occupancy(turrets)
            for i in range(min(n, MAX_SLOTS)):
                sx, sy = self._turret_slots[i]
                slots_arr[i] = [sx, sy, feats[i][0], feats[i][1], feats[i][2]]
                slot_valid_mask[i] = 1
                if occupied[i]:
                    refill_slot_mask[i] = 1
                    pick_slot_mask[i] = 1
                else:
                    place_slot_mask[i] = 1

        # Anchors (static positions the character may stand on)
        anchors_arr = np.zeros((MAX_ANCHORS, 2), dtype=np.float32)
        anchor_valid_mask = np.zeros(MAX_ANCHORS, dtype=np.int8)
        na = 0 if self._anchor_slots is None else len(self._anchor_slots)
        for i in range(min(na, MAX_ANCHORS)):
            anchors_arr[i] = self._anchor_slots[i]
            anchor_valid_mask[i] = 1

        # Reach mask: which slots are within REACH_DISTANCE of the current anchor
        reach_slot_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        if self._anchor_reach_matrix is not None:
            reach_slot_mask[:] = self._anchor_reach_matrix[self._current_anchor_index]

        # Character
        char_arr = np.zeros(4, dtype=np.float32)
        try:
            health = self._get_character_health()
            char_arr[0] = ns.player_location.x
            char_arr[1] = ns.player_location.y
            char_arr[2] = health
            char_arr[3] = inv_arr[0]  # firearm-magazine count as proxy
        except Exception:
            pass

        # Radar
        radar_arr = np.zeros(3, dtype=np.float32)
        try:
            for e in entities:
                if e.name == "radar":
                    radar_arr[0] = e.position.x
                    radar_arr[1] = e.position.y
                    radar_arr[2] = getattr(e, "health", 0)
                    break
        except Exception:
            pass

        # Game
        elapsed_ticks = self.instance.get_elapsed_ticks()
        game_arr = np.array(
            [elapsed_ticks, self._wave_number], dtype=np.float32
        )

        return {
            "map": map_grid,
            "inventory": inv_arr,
            "turret_slots": slots_arr,
            "slot_valid_mask": slot_valid_mask,
            "place_slot_mask": place_slot_mask,
            "refill_slot_mask": refill_slot_mask,
            "pick_slot_mask": pick_slot_mask,
            "anchors": anchors_arr,
            "anchor_valid_mask": anchor_valid_mask,
            "reach_slot_mask": reach_slot_mask,
            "character": char_arr,
            "radar": radar_arr,
            "game": game_arr,
        }

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

    def _get_kill_count(self) -> int:
        """Get total enemy kills."""
        try:
            response = self.instance.rcon_client.send_command(
                "/sc rcon.print(game.forces['player'].kill_count_statistics.input_counts['small-biter'] or 0)"
            )
            return int(response)
        except Exception:
            return 0

    def _compute_reward(self, invalid: bool) -> float:
        """Compute step reward."""
        cfg = self.config
        reward = 0.0

        # Survival bonus
        reward += cfg.alpha_survive

        # Kill bonus
        kills = self._get_kill_count()
        delta_kills = kills - self._prev_kills
        self._prev_kills = kills
        reward += cfg.beta_kills * delta_kills

        # Damage penalty
        health = self._get_character_health()
        damage_taken = max(0, self._prev_health - health)
        self._prev_health = health
        reward -= cfg.delta_damage * damage_taken

        # Invalid action penalty
        if invalid:
            reward -= cfg.epsilon_invalid

        return reward

    def _check_terminated(self, obs: Dict[str, np.ndarray]) -> bool:
        """Check if episode should terminate."""
        if self._has_radar and obs["radar"][2] <= 0 and self._step_count > 1:
            return True
        if obs["character"][2] <= 0 and self._step_count > 1:
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
