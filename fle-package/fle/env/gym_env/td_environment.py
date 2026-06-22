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
    CHAR_HP_NORM,
    COUNT_NORM,
    ETA_NORM,
    GROUP_FEATURES,
    MAX_ANCHORS,
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

        # Async-movement state (set by _move_to_anchor, polled in step()).
        self._moving = False
        self._move_target = -1

        # Raw (un-normalized) values stashed during _get_observation so the
        # reward/termination logic doesn't re-query the game.
        self._cur_char_hp = 0.0
        self._cur_radar_hp = 0.0
        self._cur_coverage = 0.0       # filled_reachable / total_reachable
        self._cur_ammo_frac = 0.0      # loaded_turrets / live_turrets
        self._cur_threat_closeness = 0.0  # 0 (far) .. 1 (on top of base)
        self._last_events: Dict[str, int] = {}

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

        # Starting enemy nest/worm layout, recorded once on first reset as a
        # list of (name, x, y). Used to rebuild nests on every reset.
        self._starting_nests: Optional[list] = None

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
            if self.config.restore_starting_nests_on_reset:
                self._restore_starting_nests()

        # Re-roll which slots start empty for this episode (seeded by reset seed).
        self._delete_turret_subset()

        self._step_count = 0
        self._prev_kills = 0
        self._prev_health = self._get_character_health()
        self._prev_radar_hp = self._get_radar_hp()
        self._wave_number = 0
        self._current_anchor_index = 0
        self._moving = False
        self._move_target = -1
        # Flush any death events accumulated outside an episode so the first
        # step's reward only reflects in-episode deaths.
        self._read_events()

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
        self._total_steps += 1

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

        ev = getattr(self, "_last_events", {}) or {}
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
            "coverage": self._cur_coverage,
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
        """Start an asynchronous walk to the chosen anchor tile.

        Hands a path to the scenario's on_tick walker (via the walk_to tool); the
        character then travels over the next decision window(s) while biters keep
        attacking. Does NOT block. Returns True if the anchor index is invalid.
        Issuing this while already walking simply re-routes to the new anchor.
        """
        n = 0 if self._anchor_slots is None else len(self._anchor_slots)
        if anchor_index < 0 or anchor_index >= n:
            return True  # padding / out-of-range anchor
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

    def _plan_path(self, start: Position, finish: Position) -> list:
        """Return an ordered waypoint list from start to finish.

        Tries Factorio A* (request_path/get_path) for obstacle-aware routing and
        falls back to a straight line. The teleport-based walker tolerates either,
        so the straight line is a safe fallback (e.g. when the pathfinder is busy
        or the game is paused). Swapping in richer routing later is transparent.
        """
        ns = self.instance.first_namespace
        try:
            handle = ns._request_path(
                start, finish, allow_paths_through_own_entities=True, resolution=-1
            )
            waypoints = ns._get_path(handle)
            if waypoints:
                return waypoints
        except Exception as e:
            logger.debug(f"A* path failed ({e}); using straight-line walk")
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
            return {
                "kills": 0, "turrets_lost": 0, "walls_lost": 0,
                "buildings_lost": 0, "radar_lost": 0, "char_died": 0,
            }

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

    # --- Runtime wave spawning (DISABLED) -----------------------------------
    # The map ships with baked-in biter nests that spawn and path to the base on
    # their own, so we never drive waves from Python. The only call site (in
    # step()) is already commented out; the method body is kept here, commented,
    # for later development. Re-enable both together if runtime waves are wanted.
    #
    # def _spawn_wave(self):
    #     """Spawn a wave of enemies using biter_director."""
    #     if not self.config.spawn_waves_at_runtime:
    #         return
    #     try:
    #         self.instance.first_namespace._biter_director(
    #             wave_number=self._wave_number,
    #             center_x=self._center_x,
    #             center_y=self._center_y,
    #             spawn_radius=self._radius * 1.5,
    #             base_count=self.config.base_enemy_count,
    #             escalation_factor=self.config.escalation_factor,
    #             use_spawners=self.config.spawn_from_map_spawners,
    #             target_x=self._center_x,
    #             target_y=self._center_y,
    #         )
    #     except Exception as e:
    #         logger.warning(f"Failed to spawn wave {self._wave_number}: {e}")
    # ------------------------------------------------------------------------

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

        # Entities (used for turret occupancy)
        entities = []
        try:
            entities = ns.get_entities(
                position=Position(x=cx0, y=cy0),
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

        # Turret slots + masks (positions normalized; reach from LIVE char pos)
        slots_arr = np.zeros((MAX_SLOTS, SLOT_FEATURES), dtype=np.float32)
        slot_valid_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        place_slot_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        refill_slot_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        pick_slot_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        reach_slot_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        slots_arr[:, 2] = -1.0  # padding rows: occupied = -1
        live_turrets = 0
        loaded_turrets = 0
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
                        loaded_turrets += 1
                else:
                    place_slot_mask[i] = 1

        # Action-conditioned reach masks. While in transit no turret action is
        # legal, so they are all zero (the policy may only re-route / noop).
        if self._moving:
            place_reach_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
            refill_reach_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
            pick_reach_mask = np.zeros(MAX_SLOTS, dtype=np.int8)
        else:
            place_reach_mask = (place_slot_mask & reach_slot_mask).astype(np.int8)
            refill_reach_mask = (refill_slot_mask & reach_slot_mask).astype(np.int8)
            pick_reach_mask = (pick_slot_mask & reach_slot_mask).astype(np.int8)

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
        self._cur_threat_closeness = threat_closeness

        return {
            "map": map_grid,
            "inventory": inv_arr,
            "turret_slots": slots_arr,
            "slot_valid_mask": slot_valid_mask,
            "place_slot_mask": place_slot_mask,
            "refill_slot_mask": refill_slot_mask,
            "pick_slot_mask": pick_slot_mask,
            "place_reach_mask": place_reach_mask,
            "refill_reach_mask": refill_reach_mask,
            "pick_reach_mask": pick_reach_mask,
            "anchors": anchors_arr,
            "anchor_valid_mask": anchor_valid_mask,
            "reach_slot_mask": reach_slot_mask,
            "biter_groups": groups_arr,
            "group_valid_mask": group_valid_mask,
            "nests": nests_arr,
            "nest_valid_mask": nest_valid_mask,
            "movement": movement_arr,
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

    def _compute_reward(self, invalid: bool) -> float:
        """Compute step reward from server-side death events + stashed raw state.

        Reads (and clears) the death/kill counters accumulated during the last
        decision window, applies the destruction penalties you specified, the
        character/radar damage penalties, and light dense shaping that anneals to
        zero over cfg.shaping_decay_steps. Stashes the events for _check_terminated.
        """
        cfg = self.config
        reward = 0.0

        # Survival.
        reward += cfg.alpha_survive

        # Event-based kills + destruction (read-and-clear for this window).
        ev = self._read_events()
        self._last_events = ev
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

        # Dense shaping, annealed to 0 over training.
        decay = max(0.0, 1.0 - self._total_steps / max(1, cfg.shaping_decay_steps))
        if decay > 0.0:
            reward += decay * cfg.w_coverage * self._cur_coverage
            reward += decay * cfg.w_ammo * self._cur_ammo_frac
            reward -= decay * cfg.w_threat * self._cur_threat_closeness

        return reward

    def _check_terminated(self, obs: Dict[str, np.ndarray]) -> bool:
        """Terminate on radar destruction or character death."""
        if self._step_count <= 1:
            return False
        ev = getattr(self, "_last_events", {}) or {}
        if self._has_radar and (self._cur_radar_hp <= 0 or ev.get("radar_lost", 0) > 0):
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
