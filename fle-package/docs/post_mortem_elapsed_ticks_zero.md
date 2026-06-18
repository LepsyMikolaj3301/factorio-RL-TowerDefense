# Post-Mortem: `test_elapsed_ticks_increment` always fails on default_lab_scenario

**Filed:** 2026-06-18  
**Symptom:** `obs["game"][0]` stays `0.0` after multiple `step()` calls. The smoke test
`TestTowerDefenseSmoke::test_elapsed_ticks_increment` asserts `obs["game"][0] > initial_ticks`
and always fails with `assert np.float32(0.0) > np.float32(0.0)`.

---

## Root cause

`TowerDefenseEnv` reads elapsed ticks from a **Lua-side counter** that only exists in the
`tower_defense` scenario:

```lua
-- fle/cluster/scenarios/tower_defense/control.lua
script.on_event(defines.events.on_tick, function(event)
    storage.elapsed_ticks = (storage.elapsed_ticks or 0) + 1
end)
```

`get_elapsed_ticks()` reads this counter via RCON:

```python
# fle/env/instance.py:105
self.rcon_client.send_command("/sc rcon.print(storage.elapsed_ticks or 0)")
```

`default_lab_scenario/control.lua` is essentially empty — it has no `on_tick` handler, so
`storage.elapsed_ticks` is never set. The RCON call returns `0` on every tick, making
`obs["game"][0]` and `info["elapsed_ticks"]` permanently `0`.

The `fle cluster` CLI's `run-envs.sh` only accepts `open_world` or `default_lab_scenario`
right now (hardcoded validation), so integration tests that need the `tower_defense`
scenario can't start a container with the correct scenario via the CLI.

---

## Evidence

- `tower_defense/control.lua`: has `on_tick` → `storage.elapsed_ticks += 1` ✓
- `default_lab_scenario/control.lua`: empty (just `util = require("util")`) — no counter
- `get_elapsed_ticks()` → `/sc rcon.print(storage.elapsed_ticks or 0)` → always `0` on lab scenario
- Direct RCON probe on a lab container: `storage.elapsed_ticks` is always `nil`
- The test fails only when run against `default_lab_scenario`; it would likely pass
  against `tower_defense` (not verified because the CLI blocks that scenario)

---

## Impact

- `obs["game"][0]` (elapsed ticks) and `obs["game"][1]` (wave number, since
  `new_wave = int(elapsed_ticks / ticks_per_wave)` is always 0) are both stuck at 0
- Wave spawning never triggers (`_spawn_wave()` is never called)
- Termination by `max_ticks` never fires (`truncated = elapsed_ticks >= self.max_ticks`)
- The agent cannot use game time as a signal for anything

---

## Affected code

| File | Location | Note |
|------|----------|------|
| `fle/env/instance.py` | `get_elapsed_ticks()` L105 | Reads `storage.elapsed_ticks or 0` |
| `fle/env/instance.py` | `_reset_elapsed_ticks()` L114 | Writes `storage.elapsed_ticks = 0` |
| `fle/env/gym_env/td_environment.py` | `step()` L299 | Reads `get_elapsed_ticks()` for obs + wave |
| `fle/env/gym_env/td_environment.py` | `_get_observation()` L501 | Encodes ticks into `obs["game"]` |
| `fle/cluster/scenarios/tower_defense/control.lua` | L22-24 | ✓ Has counter |
| `fle/cluster/scenarios/default_lab_scenario/control.lua` | (whole file) | ✗ Missing counter |
| `fle/cluster/run-envs.sh` | scenario validation | Blocks `tower_defense` as a valid value |

---

## Fix options

**Option A (preferred): Add the `on_tick` counter to `default_lab_scenario/control.lua`.**

```lua
-- default_lab_scenario/control.lua
util = require("util")

script.on_init(function()
    storage.elapsed_ticks = 0
end)

script.on_event(defines.events.on_tick, function(event)
    storage.elapsed_ticks = (storage.elapsed_ticks or 0) + 1
end)
```

This makes `get_elapsed_ticks()` work correctly regardless of which scenario is running.
Low risk — the counter is pure bookkeeping and changes no game mechanics.

**Option B: Allow `tower_defense` in `run-envs.sh` and test against the right scenario.**

Remove or relax the scenario validation in `run-envs.sh` so `fle cluster start -n 1
-s tower_defense` works. Integration tests for the tower-defense env should run against the
tower-defense scenario. `default_lab_scenario` can remain the default for non-TD tests.

**Option C: Fall back to `game.tick` in `get_elapsed_ticks()`.**

If `storage.elapsed_ticks` is nil, fall back to the built-in Factorio tick counter:
```python
"/sc rcon.print(storage.elapsed_ticks or game.tick)"
```
This would work universally but the value is not reset between episodes (unlike
`_reset_elapsed_ticks()` which only zeroes the storage counter).

**Recommendation: apply both A and B** — A for correctness on any scenario, B so TD
integration tests can actually target the right scenario.

---

## How to verify the fix

```bash
# After applying Option A:
fle cluster start -n 1 -s default_lab_scenario
pytest tests/integration/test_td_smoke.py::TestTowerDefenseSmoke::test_elapsed_ticks_increment -v
# Expect: PASSED

# After applying Option B (can also verify wave spawning):
fle cluster start -n 1 -s tower_defense
pytest tests/integration/test_td_smoke.py -v
```
