"""Regression tests for FLE Lua tick ownership."""

import re
from pathlib import Path


MODS_DIR = Path(__file__).resolve().parents[1] / "fle" / "env" / "mods"
ON_TICK_RE = re.compile(r"script\.on_event\(\s*defines\.events\.on_tick")
ON_ENTITY_DIED_RE = re.compile(r"script\.on_event\(\s*defines\.events\.on_entity_died")


def test_fle_mods_have_single_on_tick_owner():
    direct_owners = []
    for lua_file in MODS_DIR.glob("*.lua"):
        if ON_TICK_RE.search(lua_file.read_text(encoding="utf-8")):
            direct_owners.append(lua_file.name)

    assert direct_owners == ["tick_dispatcher.lua"]


def test_tick_dispatcher_calls_runtime_tick_jobs():
    dispatcher = (MODS_DIR / "tick_dispatcher.lua").read_text(encoding="utf-8")

    assert "storage.elapsed_ticks" in dispatcher
    assert 'call_tick_action("update_crafting_queue", event)' in dispatcher
    assert 'call_tick_action("update_td_walk", event)' in dispatcher
    assert 'call_tick_action("update_alerts", event)' in dispatcher


def test_td_events_registers_runtime_death_counters():
    td_events = (MODS_DIR / "td_events.lua").read_text(encoding="utf-8")

    assert ON_ENTITY_DIED_RE.search(td_events)
    for key in (
        "wall_n",
        "wall_e",
        "wall_s",
        "wall_w",
        "turret_n",
        "turret_e",
        "turret_s",
        "turret_w",
    ):
        assert key in td_events
