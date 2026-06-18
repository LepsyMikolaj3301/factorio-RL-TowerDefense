"""Gymnasium environment registry for Factorio."""

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import gymnasium

from fle.commons.cluster_ips import get_local_container_ips
from fle.env import FactorioInstance
from fle.eval.tasks import TaskFactory

PORT_OFFSET = int(os.environ.get("PORT_OFFSET", 0))


@dataclass
class GymEnvironmentSpec:
    """Specification for a registered gymnasium environment."""

    task_key: str
    task_config_path: str
    description: str
    num_agents: int
    enable_vision: bool = False


class FactorioGymRegistry:
    """Registry for Factorio gymnasium environments."""

    def __init__(self):
        self._environments: Dict[str, GymEnvironmentSpec] = {}
        self._discovered = False

    def discover_tasks(self) -> None:
        """Automatically discover all task definitions and register them as gym environments."""
        if self._discovered:
            return

        from fle.eval.tasks.task_definitions.task_registry import (
            list_all_tasks,
            get_task_info,
        )

        for task_key in list_all_tasks():
            task_info = get_task_info(task_key)
            self.register_environment(
                task_key=task_key,
                task_config_path=task_key,
                description=task_info["goal_description"],
                num_agents=task_info["num_agents"],
            )

        self._discovered = True

    def register_environment(
        self,
        task_key: str,
        task_config_path: str,
        description: str,
        num_agents: int,
        enable_vision: bool = False,
    ) -> None:
        """Register a new gymnasium environment."""
        spec = GymEnvironmentSpec(
            task_key=task_key,
            task_config_path=task_config_path,
            description=description,
            num_agents=num_agents,
            enable_vision=enable_vision,
        )
        self._environments[task_key] = spec

    def list_environments(self) -> List[str]:
        return list(self._environments.keys())

    def get_environment_spec(self, env_id: str) -> Optional[GymEnvironmentSpec]:
        return self._environments.get(env_id)


# Global registry instance
_registry = FactorioGymRegistry()


def _get_factorio_connection(run_idx: int = 0):
    """Get connection details for a Factorio container."""
    address = os.getenv("FACTORIO_SERVER_ADDRESS")
    tcp_port = os.getenv("FACTORIO_SERVER_PORT")

    if not address and not tcp_port:
        try:
            ips, udp_ports, tcp_ports = get_local_container_ips()
        except ValueError:
            raise RuntimeError("No Factorio containers available")

        if len(tcp_ports) == 0:
            raise RuntimeError("No Factorio containers available")

        container_idx = PORT_OFFSET + run_idx
        if container_idx >= len(tcp_ports):
            raise RuntimeError(
                f"Container index {container_idx} exceeds available containers ({len(tcp_ports)})"
            )

        address = ips[container_idx]
        tcp_port = tcp_ports[container_idx]

    return address, int(tcp_port)


def make_td_env(
    run_idx: int = 0,
    save_path: Optional[str] = None,
    config=None,
    **kwargs,
):
    """Create a Tower Defense gymnasium environment.

    Args:
        run_idx: Container index for multi-env setups.
        save_path: Optional path to a prebuilt Factorio save.
        config: TDScenarioConfig instance (defaults to MEDIUM).
        **kwargs: Additional kwargs forwarded to TowerDefenseEnv.
    """
    from fle.env.gym_env.td_config import TDScenarioConfig
    from fle.env.gym_env.td_environment import TowerDefenseEnv
    from fle.eval.tasks.tower_defense_task import TowerDefenseTask

    if config is None:
        config = TDScenarioConfig.MEDIUM

    address, tcp_port = _get_factorio_connection(run_idx)

    instance = FactorioInstance(
        address=address,
        tcp_port=tcp_port,
        num_agents=1,
        fast=True,
        cache_scripts=True,
        inventory={},
        all_technologies_researched=True,
        peaceful=False,
        save_path=save_path,
    )

    task = TowerDefenseTask(starting_inventory_dict=config.starting_inventory)
    task.setup(instance)

    env = TowerDefenseEnv(instance=instance, config=config, **kwargs)
    return env


def register_all_environments() -> None:
    """Register all discovered environments with gymnasium."""
    _registry.discover_tasks()

    # Register the TD environment
    try:
        gymnasium.register(
            id="factorio-td-v0",
            entry_point="fle.env.gym_env.registry:make_td_env",
        )
    except gymnasium.error.Error:
        pass  # Already registered


def list_available_environments() -> List[str]:
    return _registry.list_environments()


def get_environment_info(task_key: str) -> Optional[Dict[str, Any]]:
    spec = _registry.get_environment_spec(task_key)
    if spec is None:
        return None
    return asdict(spec)


# Auto-register environments when module is imported
register_all_environments()
