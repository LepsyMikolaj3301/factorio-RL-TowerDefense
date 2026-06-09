"""Local conftest for gym_env tests.

Overrides the global autouse `_reset_between_tests` fixture so that unit tests
(test_td_spaces, test_td_environment space checks) can run without a live
Factorio server. Tests that genuinely need an instance will still get one via
the session-scoped `instance` fixture when a server is available.
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_between_tests(request):
    """No-op reset for gym_env unit tests that don't need a live server."""
    if "instance" not in request.fixturenames:
        yield
        return
    # If the test explicitly requested `instance`, delegate to the real fixture.
    instance = request.getfixturevalue("instance")
    if instance is None:
        yield
        return
    if hasattr(instance, "default_initial_inventory"):
        try:
            instance.initial_inventory = dict(instance.default_initial_inventory)
        except Exception:
            instance.initial_inventory = instance.default_initial_inventory
    instance.reset(reset_position=True)
    yield
