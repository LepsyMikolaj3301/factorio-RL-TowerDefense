"""Tower Defense Gymnasium environment for Factorio."""

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np

from fle.commons.models.game_state import GameState
from fle.env import FactorioInstance
from fle.env.gym_env.td_config import TDScenarioConfig
from fle.env.gym_env.td_spaces import (
    ACTION_NOOP,
    ACTION_PLACE_TURRET,
    ACTION_REFILL_TURRET,
    MAX_SLOTS,
    NUM_CHANNELS,
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
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        instance: FactorioInstance,
        config: Optional[TDScenarioConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.instance = instance
        self.config = config or TDScenarioConfig.MEDIUM
        self.render_mode = render_mode

        # Convenience aliases for readability inside the class
        cfg = self.config
        self.grid_size = cfg.grid_size
        self.cell_size = 1.0
        self.decision_cadence = cfg.decision_cadence
        self.max_ticks = cfg.max_ticks
        self.game_speed = cfg.game_speed
        self.max_slots = min(cfg.max_turret_slots, MAX_SLOTS)

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

        # Save-state reset
        self._initial_snapshot: Optional[GameState] = None
        self._initial_evolution_factor: float = 0.0

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

            try:
                loc = self.instance.first_namespace.player_location
                self._center_x = float(loc.x)
                self._center_y = float(loc.y)
            except Exception:
                self._center_x = 0.0
                self._center_y = 0.0

            try:
                self.instance.first_namespace.place_entity(
                    Prototype.Radar,
                    position=self.instance.first_namespace.player_location,
                )
            except Exception:
                pass

            # If the map ships without turrets, seed them from config so the
            # slot system still has positions to work with.
            self._seed_turret_positions_if_needed()

            # Read the canonical slot geometry from every turret on the map.
            self._read_turret_slots()

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

            try:
                loc = self.instance.first_namespace.player_location
                self._center_x = float(loc.x)
                self._center_y = float(loc.y)
            except Exception:
                self._center_x = 0.0
                self._center_y = 0.0

            # Radar is restored by _load_entity_state; place_entity no-ops on collision.
            try:
                self.instance.first_namespace.place_entity(
                    Prototype.Radar,
                    position=self.instance.first_namespace.player_location,
                )
            except Exception:
                pass

        # Re-roll which slots start empty for this episode (seeded by reset seed).
        self._delete_turret_subset()

        self._step_count = 0
        self._prev_kills = 0
        self._prev_health = self._get_character_health()
        self._wave_number = 0

        obs = self._get_observation()
        info = {"step": 0, "elapsed_ticks": 0, "wave": 0}
        return obs, info

    def _capture_save_state(self) -> None:
        self._initial_snapshot = GameState.from_instance(self.instance)
        raw = self.instance.rcon_client.send_command(
            "/sc rcon.print(game.forces['player'].evolution_factor)"
        )
        try:
            self._initial_evolution_factor = float(raw.strip())
        except (ValueError, AttributeError):
            self._initial_evolution_factor = 0.0

    def _restore_evolution_factor(self) -> None:
        self.instance.rcon_client.send_command(
            f"/sc game.forces['player'].evolution_factor = {self._initial_evolution_factor}"
        )

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

    def _delete_turret_subset(self) -> None:
        """Destroy a seeded random subset of slot turrets for this episode."""
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
        try:
            self.instance.first_namespace._destroy_turrets(positions=positions)
        except Exception as e:
            logger.warning(f"Failed to delete turret subset: {e}")

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

        # Unpause, let game run for decision_cadence ticks, then pause
        self.instance.unpause()
        sleep_seconds = self.decision_cadence / 60.0 / self.game_speed
        if sleep_seconds > 0:
            import time
            time.sleep(sleep_seconds)
        self.instance.pause()

        # Check wave spawning
        elapsed_ticks = self.instance.get_elapsed_ticks()
        new_wave = int(elapsed_ticks / self.config.ticks_per_wave)
        if new_wave > self._wave_number:
            self._wave_number = new_wave
            self._spawn_wave()

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

        if action_type == ACTION_NOOP:
            return False

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
            if action_type == ACTION_PLACE_TURRET:
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

    def _spawn_wave(self):
        """Spawn a wave of enemies using biter_director."""
        try:
            self.instance.first_namespace._biter_director(
                wave_number=self._wave_number,
                center_x=self._center_x,
                center_y=self._center_y,
                spawn_radius=self._radius * 1.5,
                base_count=self.config.base_enemy_count,
                escalation_factor=self.config.escalation_factor,
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
                else:
                    place_slot_mask[i] = 1

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
            "character": char_arr,
            "radar": radar_arr,
            "game": game_arr,
        }

    def _get_character_health(self) -> float:
        """Get the player character's health."""
        try:
            response = self.instance.rcon_client.send_command(
                "/sc rcon.print(game.get_player(1).character and game.get_player(1).character.health or 0)"
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
        if obs["radar"][2] <= 0 and self._step_count > 1:
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
