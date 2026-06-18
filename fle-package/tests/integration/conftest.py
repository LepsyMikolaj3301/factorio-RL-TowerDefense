"""Conftest for integration tests requiring a live Factorio server."""

import logging

import pytest

from fle.commons.cluster_ips import get_local_container_ips

log = logging.getLogger("td_actions")

# Inventory used by the action-level tests: enough turrets, ammo and walls to
# place, pick up, refill, and render entities on the radar grid.
TD_INVENTORY = {
    "gun-turret": 5,
    "firearm-magazine": 100,
    "stone-wall": 20,
    "pistol": 1,
}


@pytest.fixture()
def clean_namespace(request):
    """A fresh FactorioInstance namespace, isolated per test.

    Each test gets its own clean instance (constructor clears entities + resets),
    so there is no cross-test state leakage. Skips automatically when no Factorio
    container is running.
    """
    from fle.env import FactorioInstance

    res = get_local_container_ips()
    if not res or len(res) < 3 or not res[2]:
        pytest.skip(
            "No Factorio containers running — start with: "
            "fle cluster start --num-instances 1 --scenario tower_defense"
        )

    tcp_port = sorted(res[2])[-1]
    log.info("[%s] creating clean instance on tcp_port=%s", request.node.name, tcp_port)
    inst = FactorioInstance(
        address="localhost",
        tcp_port=tcp_port,
        inventory=dict(TD_INVENTORY),
        peaceful=False,
        cache_scripts=True,
        fast=True,
        all_technologies_researched=True,
    )
    inst.set_speed(10.0)
    try:
        yield inst.first_namespace
    finally:
        log.info("[%s] cleaning up instance", request.node.name)
        inst.cleanup()


@pytest.fixture(scope="session")
def td_env():
    """Create a TowerDefenseEnv against a live container. Skips if none available."""
    result = get_local_container_ips()
    if not result or len(result) < 3 or not result[2]:
        pytest.skip(
            "No Factorio containers running — start with: "
            "fle cluster start --num-instances 1 --scenario tower_defense"
        )

    _ips, _udp_ports, tcp_ports = result

    from fle.env.gym_env.registry import make_td_env

    env = make_td_env(run_idx=0)
    yield env
    env.close()


@pytest.fixture(autouse=True)
def _reset_between_tests(request):
    """No-op: integration tests manage their own state."""
    yield
