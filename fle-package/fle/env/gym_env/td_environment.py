"""Tower Defense Gymnasium environment for Factorio."""

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np

from fle.env import FactorioInstance
from fle.env.gym_env.td_spaces import (
    ACTION_NOOP,
    ACTION_PLACE_WALL,
    ACTION_PLACE_TURRET,
    ACTION_MOVE_WALL,
    ACTION_MOVE_TURRET,
    ACTION_REFILL_TURRET,
    ACTION_SHOOT,
    DEFAULT_GRID_SIZE,
    MAX_TURRETS,
    MAX_WALLS,
    NUM_CHANNELS,
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
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        instance: FactorioInstance,
        grid_size: int = DEFAULT_GRID_SIZE,
        cell_size: float = 1.0,
        decision_cadence: int = 60,
        max_ticks: int = 108000,  # 30 minutes at 60 ticks/sec
        game_speed: float = 10.0,
        # Reward coefficients
        alpha_survive: float = 0.01,
        beta_kills: float = 1.0,
        gamma_ammo: float = 0.1,
        delta_damage: float = 0.5,
        epsilon_invalid: float = 0.5,
        terminal_bonus: float = 100.0,
        terminal_penalty: float = -100.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.instance = instance
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.decision_cadence = decision_cadence
        self.max_ticks = max_ticks
        self.game_speed = game_speed
        self.render_mode = render_mode

        # Reward coefficients
        self.alpha = alpha_survive
        self.beta = beta_kills
        self.gamma = gamma_ammo
        self.delta = delta_damage
        self.epsilon = epsilon_invalid
        self.terminal_bonus = terminal_bonus
        self.terminal_penalty = terminal_penalty

        # Spaces
        self.observation_space = make_observation_space(grid_size)
        self.action_space = make_action_space(grid_size)

        # Internal state
        self._step_count = 0
        self._prev_kills = 0
        self._prev_health = 0.0
        self._wave_number = 0
        self._ticks_per_wave = 3600  # 60 seconds per wave
        self._center_x = 0.0
        self._center_y = 0.0
        self._radius = grid_size * cell_size / 2.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        # Reset the Factorio instance
        self.instance.reset()
        self.instance.set_speed(self.game_speed)
        self.instance.pause()
        self.instance._reset_elapsed_ticks()

        # Anchor the grid center to the player's spawn position
        try:
            loc = self.instance.first_namespace.player_location
            self._center_x = float(loc.x)
            self._center_y = float(loc.y)
        except Exception:
            self._center_x = 0.0
            self._center_y = 0.0

        # Place radar at the center now that terrain is ready
        try:
            from fle.env.game_types import Prototype
            self.instance.first_namespace.place_entity(
                Prototype.Radar,
                position=self.instance.first_namespace.player_location,
            )
        except Exception:
            pass

        self._step_count = 0
        self._prev_kills = 0
        self._prev_health = self._get_character_health()
        self._wave_number = 0

        obs = self._get_observation()
        info = {"step": 0, "elapsed_ticks": 0, "wave": 0}
        return obs, info

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self._step_count += 1

        # Execute action
        invalid = self._execute_action(action)

        # Unpause, let game run for decision_cadence ticks, then pause
        self.instance.unpause()
        # Wait for ticks by sending a sleep command through RCON
        sleep_seconds = self.decision_cadence / 60.0 / self.game_speed
        if sleep_seconds > 0:
            import time
            time.sleep(sleep_seconds)
        self.instance.pause()

        # Check wave spawning
        elapsed_ticks = self.instance.get_elapsed_ticks()
        new_wave = int(elapsed_ticks / self._ticks_per_wave)
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
            reward += self.terminal_penalty
        elif truncated:
            reward += self.terminal_bonus

        info = {
            "step": self._step_count,
            "elapsed_ticks": elapsed_ticks,
            "wave": self._wave_number,
            "invalid_action": invalid,
        }

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute the given action. Returns True if the action was invalid."""
        action_type = int(action.get("action_type", ACTION_NOOP))
        target_x = int(action.get("target_x", 0))
        target_y = int(action.get("target_y", 0))
        source_x = int(action.get("source_x", 0))
        source_y = int(action.get("source_y", 0))
        ammo_amount = int(action.get("ammo_amount", 0))

        # Convert grid coords to world coords
        world_tx = self._center_x - self._radius + target_x * self.cell_size
        world_ty = self._center_y - self._radius + target_y * self.cell_size
        world_sx = self._center_x - self._radius + source_x * self.cell_size
        world_sy = self._center_y - self._radius + source_y * self.cell_size

        ns = self.instance.first_namespace

        try:
            if action_type == ACTION_NOOP:
                return False

            elif action_type == ACTION_PLACE_WALL:
                ns.place_entity(
                    Prototype.Wall,
                    Direction.UP,
                    Position(x=world_tx, y=world_ty),
                )
                return False

            elif action_type == ACTION_PLACE_TURRET:
                ns.place_entity(
                    Prototype.GunTurret,
                    Direction.UP,
                    Position(x=world_tx, y=world_ty),
                )
                return False

            elif action_type == ACTION_MOVE_WALL:
                # Pick up from source and place at target
                entities = ns.get_entities(
                    position=Position(x=world_sx, y=world_sy),
                    radius=1.0,
                )
                for e in entities:
                    if e.name == "stone-wall":
                        ns.pickup_entity(e)
                        ns.place_entity(
                            Prototype.Wall,
                            Direction.UP,
                            Position(x=world_tx, y=world_ty),
                        )
                        return False
                return True  # No wall found at source

            elif action_type == ACTION_MOVE_TURRET:
                entities = ns.get_entities(
                    position=Position(x=world_sx, y=world_sy),
                    radius=1.0,
                )
                for e in entities:
                    if e.name == "gun-turret":
                        ns.pickup_entity(e)
                        ns.place_entity(
                            Prototype.GunTurret,
                            Direction.UP,
                            Position(x=world_tx, y=world_ty),
                        )
                        return False
                return True  # No turret found at source

            elif action_type == ACTION_REFILL_TURRET:
                entities = ns.get_entities(
                    position=Position(x=world_tx, y=world_ty),
                    radius=1.0,
                )
                for e in entities:
                    if e.name == "gun-turret":
                        amount = max(1, ammo_amount)
                        ns.insert_item(
                            Prototype.FirearmMagazine,
                            e,
                            amount,
                        )
                        return False
                return True  # No turret at target

            elif action_type == ACTION_SHOOT:
                ns._shoot(world_tx, world_ty, self.decision_cadence)
                return False

            else:
                return True  # Unknown action type

        except Exception as e:
            logger.debug(f"Action failed: {e}")
            return True

    def _spawn_wave(self):
        """Spawn a wave of enemies using biter_director."""
        try:
            self.instance.first_namespace._biter_director(
                wave_number=self._wave_number,
                center_x=self._center_x,
                center_y=self._center_y,
                spawn_radius=self._radius * 1.5,
                base_count=5,
                escalation_factor=1.5,
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

        # Turrets
        turrets_arr = np.zeros((MAX_TURRETS, 4), dtype=np.float32)
        try:
            entities = ns.get_entities(
                position=Position(x=self._center_x, y=self._center_y),
                radius=self._radius,
            )
            turret_idx = 0
            for e in entities:
                if e.name == "gun-turret" and turret_idx < MAX_TURRETS:
                    turrets_arr[turret_idx] = [
                        e.position.x,
                        e.position.y,
                        getattr(e, "ammo_count", 0),
                        getattr(e, "health", 0),
                    ]
                    turret_idx += 1
        except Exception:
            pass

        # Walls
        walls_arr = np.zeros((MAX_WALLS, 3), dtype=np.float32)
        try:
            wall_idx = 0
            for e in entities:
                if e.name == "stone-wall" and wall_idx < MAX_WALLS:
                    walls_arr[wall_idx] = [
                        e.position.x,
                        e.position.y,
                        getattr(e, "health", 0),
                    ]
                    wall_idx += 1
        except Exception:
            pass

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
            "turrets": turrets_arr,
            "walls": walls_arr,
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
        reward = 0.0

        # Survival bonus
        reward += self.alpha

        # Kill bonus
        kills = self._get_kill_count()
        delta_kills = kills - self._prev_kills
        self._prev_kills = kills
        reward += self.beta * delta_kills

        # Damage penalty
        health = self._get_character_health()
        damage_taken = max(0, self._prev_health - health)
        self._prev_health = health
        reward -= self.delta * damage_taken

        # Invalid action penalty
        if invalid:
            reward -= self.epsilon

        return reward

    def _check_terminated(self, obs: Dict[str, np.ndarray]) -> bool:
        """Check if episode should terminate."""
        # Radar destroyed
        if obs["radar"][2] <= 0 and self._step_count > 1:
            return True
        # Character dead
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
