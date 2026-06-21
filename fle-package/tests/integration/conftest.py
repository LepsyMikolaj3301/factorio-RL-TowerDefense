"""Conftest for integration tests requiring a live Factorio server."""

import logging

import pytest

from fle.commons.cluster_ips import get_local_container_ips

log = logging.getLogger("td_actions")


def pytest_addoption(parser):
    """Custom flags for the anchor-walk test.

    --walk-verbose : print every A* waypoint coordinate (default: quiet, only print
                     when the character reaches each destination anchor).
    --walk-step    : use discrete tile-by-tile teleport stepping instead of the
                     default smooth sub-tile glide.
    """
    group = parser.getgroup("anchor-walk")
    group.addoption(
        "--walk-verbose",
        action="store_true",
        default=False,
        help="Print all A* waypoint positions during the anchor walk.",
    )
    group.addoption(
        "--walk-step",
        action="store_true",
        default=False,
        help="Teleport tile-by-tile instead of the default smooth glide.",
    )

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


@pytest.fixture()
def td_save_instance(request):
    """FactorioInstance connected to a server loaded from the tower_defense save.

    Unlike ``clean_namespace``, this fixture does NOT clear entities on connect,
    so the radar, turrets, walls, and enemy spawners baked into the save are
    preserved exactly as authored. Use this for save-validation tests.
    """
    from fle.env import FactorioInstance

    res = get_local_container_ips()
    if not res or len(res) < 3 or not res[2]:
        pytest.skip(
            "No Factorio containers running — start with: "
            "fle cluster start -n 1  (loads data/saves/tower_defense.zip by default)"
        )

    tcp_port = sorted(res[2])[-1]
    log.info("[%s] creating td_save_instance on tcp_port=%s", request.node.name, tcp_port)
    inst = FactorioInstance(
        address="localhost",
        tcp_port=tcp_port,
        inventory=dict(TD_INVENTORY),
        clear_entities=False,
        peaceful=False,
        cache_scripts=True,
        fast=True,
        all_technologies_researched=True,
    )
    inst.set_speed(10.0)
    # Chart a generous area so radar_view and get_entities work without env.reset()
    inst.rcon_client.send_command(
        "/sc game.forces.player.chart(game.surfaces[1], {{-128,-128},{128,128}})"
    )
    try:
        yield inst
    finally:
        log.info("[%s] cleaning up td_save_instance", request.node.name)
        inst.cleanup()


@pytest.fixture(autouse=True)
def _reset_between_tests(request):
    """No-op: integration tests manage their own state."""
    yield
