# CLAUDE.md — Factorio Tower Defense RL Project

## What This Project Is

A reinforcement learning agent that learns dynamic perimeter defense in Factorio. The agent has limited resources (turrets, walls, ammo) and must place/move them to survive escalating biter attacks as long as possible. Built on top of the Factorio Learning Environment (FLE), with all LLM/agent layers stripped out and replaced by a gymnasium-compatible RL interface.

## Core Loop

```
Every C×N game ticks:
  1. Game pauses → agent receives observation (radar grid, inventory, turret/wall state)
  2. Agent picks action (place/move turret or wall, refill ammo, shoot, noop)
  3. Action executes → game unpauses for C×N more ticks
  4. Repeat until radar destroyed, character dies, or max ticks reached
```

## Goal

Survive the longest possible time. Terminal conditions: radar HP ≤ 0 or character HP ≤ 0.

## Observation Space (Dict)

- `map`: uint8 symbolic grid `(C, H, W)` from radar — channels: empty, wall, turret, ammo%, biter, spitter, spawner, character. Only radar-charted area is visible (partial observability).
- `inventory`: ammo count, available turrets, available walls.
- `turrets`: padded array of `(x, y, ammo, health)` per placed turret.
- `walls`: padded array of `(x, y, health)` per placed wall.
- `character`: `(x, y, health, weapon_ammo)`.
- `radar`: `(x, y, health)` — the base's heart.
- `game`: `(elapsed_ticks, wave_intensity_proxy)`.

## Action Space (Dict → flatten wrapper available)

- `action_type`: Discrete(7) — noop, place_wall, place_turret, move_wall, move_turret, refill_turret, shoot.
- `target_xy`: MultiDiscrete([W, H]) — where to place/move to.
- `source_xy`: MultiDiscrete([W, H]) — where to move from (for move actions).
- `ammo_amount`: Discrete — how much ammo to insert (for refill).
- Invalid actions → noop + small penalty.

## Reward

`+α·tick_survived + β·Δkills − γ·ammo_used − δ·damage_taken − ε·invalid_action_penalty`
Terminal bonus for long survival, terminal penalty for death.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Training Script (SB3 PPO / CleanRL)            │
├─────────────────────────────────────────────────┤
│  TowerDefenseEnv (gymnasium.Env)                │
│    td_environment.py — step/reset/obs/reward    │
│    td_spaces.py — space definitions             │
│    td_vector.py — AsyncVectorEnv over containers│
├─────────────────────────────────────────────────┤
│  FLE Core (kept from original)                  │
│    FactorioInstance — RCON, reset, game control  │
│    Namespace — tool dispatch (no LLM eval)      │
│    Lua tools — place/move/inspect entities       │
│    Docker/Cluster — container orchestration      │
├─────────────────────────────────────────────────┤
│  New Lua/Python Tools                           │
│    admin/spawn_enemy — create biters            │
│    admin/radar_view — symbolic grid obs          │
│    admin/biter_director — wave escalation        │
│    admin/grant_items — periodic ammo resupply    │
│    agent/shoot — character shooting              │
├─────────────────────────────────────────────────┤
│  Factorio Headless Server (Docker)              │
│    Prebuilt save with terrain + spawners         │
│    Peaceful=false, custom map-settings           │
│    tower_defense/control.lua scenario            │
└─────────────────────────────────────────────────┘
```

## Key Files

### Kept from FLE (no changes needed)
- `fle/env/tools/tool.py`, `controller.py` — tool dispatch infra
- `fle/env/lua_manager.py` — loads server.lua files
- `fle/env/entities.py`, `game_types.py` — entity/prototype models
- `fle/env/utils/rcon.py`, `camera.py`, `achievements.py` — game utilities
- `fle/cluster/` — Docker/compose infra
- `fle/env/a2a_instance.py` — multi-agent (kept for future use)

### Refactored (LLM code stripped)
- `fle/env/instance.py` — deleted eval/LLM methods, kept reset/GameControl
- `fle/env/namespace.py` — deleted ~1000 lines AST/LLM eval, kept tool wiring
- `fle/env/gym_env/observation.py` — removed LLM fields
- `fle/eval/tasks/task_abc.py` — removed LLM text helpers
- `fle/run.py` — deleted eval subcommands, added --save-path

### New (to be created)
- `fle/env/gym_env/td_environment.py` — TowerDefenseEnv
- `fle/env/gym_env/td_spaces.py` — observation/action space helpers
- `fle/env/gym_env/td_vector.py` — vectorized env over Docker containers
- `fle/env/tools/admin/spawn_enemy/` — {client.py, server.lua}
- `fle/env/tools/admin/radar_view/` — {client.py, server.lua}
- `fle/env/tools/admin/biter_director/` — {server.lua, client.py}
- `fle/env/tools/admin/grant_items/` — {client.py, server.lua}
- `fle/env/tools/agent/shoot/` — {client.py, server.lua}
- `fle/eval/tasks/tower_defense_task.py`
- `fle/cluster/scenarios/tower_defense/control.lua`
- `fle/cluster/config/map-gen-settings.td.json`, `map-settings.td.json`
- `examples/rl/train_td_ppo.py`

### Deleted (LLM-only)
- `fle/agents/` — entire directory
- `fle/eval/inspect/`, `fle/eval/algorithms/`, `fle/eval/open/`, `fle/eval/analysis/`, `fle/eval/entrypoints/`
- `fle/mcp_dataloader.py`, `fle/overlay*.py`, `fle/server.py`
- LLM formatters, prompt generators, conversation models
- `data/plans/`, `data/prompts/`

## Implementation Phases

1. **Strip & fix imports** — Delete LLM layers, migrate TaskResponse, fix broken imports, update pyproject.toml deps. ✅ DONE (or in progress)
2. **New Lua/Python tools** — spawn_enemy, radar_view, shoot, biter_director, grant_items. Gate remove_enemies() behind explicit flag.
3. **New gym env + task** — TowerDefenseEnv, TowerDefenseTask, tower_defense scenario, map settings.
4. **Training infra** — AsyncVectorEnv, action-mask wrapper, reference PPO training script.
5. **Verification** — Per-tool tests, env space conformance, scripted policy survival test, PPO smoke test.

## Design Decisions

- **Observation**: symbolic grid from radar (partial observability), NOT pixel rendering. Full-grid available as debug flag.
- **Action encoding**: Dict space (clean, needs multi-input policy). Flat MultiDiscrete wrapper shipped alongside.
- **Build mode**: character-based by default. Construction-robot mode (ghost placement) deferred to Phase 6+.
- **One Docker container per env instance** — matches existing ClusterManager.
- **Prebuilt save files** — no per-episode map switching; server starts with save, env resets within it.
- **Ammo is periodically granted** by the environment (admin/grant_items), not infinite but capped, so agent must manage scarcity.

## Biter Spawning

A naive `biter_director` algorithm controls wave spawning server-side:
- Escalates wave size based on `elapsed_ticks`
- Spawn positions are outside radar range (agent doesn't know origin direction)
- Biters path toward the radar/base naturally via Factorio's AI

## Tech Stack

- **Environment**: Factorio headless server in Docker
- **Interface**: gymnasium (not legacy gym)
- **Communication**: RCON + Lua tools
- **Training**: stable-baselines3 (optional dep), compatible with CleanRL/RLlib
- **Language**: Python + Lua (server-side tools)

## Commands

```bash
# Start container with prebuilt save
mkdir -p ./saves && cp /path/to/my_map.zip ./saves/
SAVE_FILE=/factorio/saves/my_map.zip bash fle/cluster/run-envs.sh

# Run with save
python fle/run.py --env-id my_env --save-path ./saves/my_map.zip

# Smoke test (after Phase 3)
python -c "import gymnasium as gym; import fle.env.gym_env.td_environment; e=gym.make('factorio-td-v0'); e.reset(); e.step(e.action_space.sample())"

# Training smoke test (after Phase 4)
python examples/rl/train_td_ppo.py --total-steps 2000
```