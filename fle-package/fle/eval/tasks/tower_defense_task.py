"""Tower Defense task definition."""

from typing import Dict, Optional

from fle.env import FactorioInstance, Inventory
from fle.env.entities import Position
from fle.env.game_types import Prototype
from fle.commons.models.task_response import TaskResponse
from fle.eval.tasks.task_abc import TaskABC


class TowerDefenseTask(TaskABC):
    """
    Tower Defense task: defend a radar from waves of biters.

    Setup:
    - Clears existing entities
    - Places a radar at the center
    - Gives the player starting inventory (walls, turrets, ammo)
    - Sets peaceful=False so enemies persist

    Verification:
    - Success if the radar survives for the full trajectory
    - Failure if radar or character is destroyed
    """

    _DEFAULT_INVENTORY = {
        "firearm-magazine": 100,
        "stone-wall": 50,
        "gun-turret": 5,
        "piercing-rounds-magazine": 50,
        "pistol": 1,
    }

    def __init__(
        self,
        trajectory_length: int = 1800,  # 30 minutes of game time at 60 ticks/sec
        task_key: str = "tower_defense",
        goal_description: str = "Defend the radar from biter waves using walls, turrets, and shooting.",
        radar_position: Optional[Position] = None,
        starting_inventory_dict: Optional[Dict] = None,
    ):
        inv_dict = dict(starting_inventory_dict) if starting_inventory_dict else dict(self._DEFAULT_INVENTORY)
        # Ensure pistol is always present for the player to shoot with
        inv_dict.setdefault("pistol", 1)
        starting_inventory = Inventory(**inv_dict)

        super().__init__(
            trajectory_length=trajectory_length,
            starting_inventory=starting_inventory,
            goal_description=goal_description,
            task_key=task_key,
            all_technology_reserached=True,
        )

        self.radar_position = radar_position or Position(x=0, y=0)

    def setup_instance(self, instance: FactorioInstance):
        """Place the radar at the player spawn, moving the player clear first."""
        ns = instance.first_namespace

        # Anchor radar to player spawn position
        try:
            loc = ns.player_location
            self.radar_position = Position(x=float(loc.x), y=float(loc.y))
        except Exception:
            pass

        # The character stands on the spawn tile, but the radar is a 3x3 entity
        # that would collide with the player and fail to place. Move the character
        # clear of the radar footprint before placing it.
        try:
            ns.move_to(
                Position(x=self.radar_position.x + 4.0, y=self.radar_position.y)
            )
        except Exception as e:
            print(f"Warning: Could not move character clear of radar: {e}")

        # Place the radar
        try:
            ns.place_entity(
                Prototype.Radar,
                position=self.radar_position,
            )
        except Exception as e:
            print(f"Error: Could not place radar at {self.radar_position}: {e}")

    def verify(
        self,
        score: float,
        step: int,
        instance: FactorioInstance,
        step_statistics: Dict,
    ) -> TaskResponse:
        """Check if radar is still alive."""
        ns = instance.first_namespace

        # Check radar health
        radar_alive = False
        try:
            entities = ns.get_entities(
                position=self.radar_position,
                radius=2.0,
            )
            for e in entities:
                if e.name == "radar" and getattr(e, "health", 0) > 0:
                    radar_alive = True
                    break
        except Exception:
            pass

        # Check character health
        character_alive = False
        try:
            response = instance.rcon_client.send_command(
                "/sc rcon.print(game.get_player(1).character and game.get_player(1).character.health or 0)"
            )
            character_alive = float(response) > 0
        except Exception:
            pass

        elapsed_ticks = instance.get_elapsed_ticks()
        success = radar_alive and character_alive

        return TaskResponse(
            success=success,
            meta={
                "radar_alive": radar_alive,
                "character_alive": character_alive,
                "elapsed_ticks": elapsed_ticks,
                "step": step,
            },
        )
