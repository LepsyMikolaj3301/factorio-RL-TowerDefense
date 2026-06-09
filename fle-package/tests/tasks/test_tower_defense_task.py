"""Unit tests for TowerDefenseTask."""

import pytest
from unittest.mock import MagicMock

from fle.eval.tasks.tower_defense_task import TowerDefenseTask
from fle.commons.models.task_response import TaskResponse


class TestTowerDefenseTask:
    def test_default_construction(self):
        task = TowerDefenseTask()
        assert task.task_key == "tower_defense"
        assert task.trajectory_length == 1800
        assert task.radar_position.x == 0
        assert task.radar_position.y == 0

    def test_custom_params(self):
        task = TowerDefenseTask(
            trajectory_length=3600,
            starting_ammo=200,
            starting_walls=100,
            starting_turrets=10,
        )
        assert task.trajectory_length == 3600
        assert task.starting_ammo == 200

    def test_starting_inventory_contents(self):
        task = TowerDefenseTask(starting_ammo=50, starting_walls=20, starting_turrets=3)
        inv = task.starting_inventory
        assert inv["firearm-magazine"] == 50
        assert inv["stone-wall"] == 20
        assert inv["gun-turret"] == 3
        assert inv["piercing-rounds-magazine"] == 50
        assert inv["pistol"] == 1

    def test_verify_returns_task_response(self):
        task = TowerDefenseTask()
        instance = MagicMock()
        instance.first_namespace = MagicMock()
        instance.first_namespace.get_entities = MagicMock(return_value=[])
        instance.rcon_client = MagicMock()
        instance.rcon_client.send_command = MagicMock(return_value="100")
        instance.get_elapsed_ticks = MagicMock(return_value=500)

        result = task.verify(
            score=0.0,
            step=10,
            instance=instance,
            step_statistics={},
        )

        assert isinstance(result, TaskResponse)
        assert "radar_alive" in result.meta
        assert "character_alive" in result.meta

    def test_goal_description_set(self):
        task = TowerDefenseTask()
        assert "Defend" in task.goal_description
