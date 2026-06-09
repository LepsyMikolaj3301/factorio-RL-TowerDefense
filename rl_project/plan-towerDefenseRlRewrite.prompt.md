# Plan: FLE → Tower-Defense RL Rewrite

**TL;DR.** Keep the low-level Factorio plumbing (`FactorioInstance`, RCON, Lua-tool system, entity models, docker cluster) and **replace everything above it**. Delete the LLM/agents/inspect/MCP layers. Replace `FactorioGymEnv` (which exposes "execute arbitrary Python code" as its action) with a gymnasium-compatible `TowerDefenseEnv` that has a small `MultiDiscrete` action space and a compact symbolic-tensor observation derived from radar coverage. Add three new Lua/Python tools (spawn-enemy, radar-view, shoot) and a `TowerDefenseTask` with survival-based reward.

## Phases

### Phase 1 — Strip & fix imports (parallelisable, low risk)

#### 1a. Delete directories & files (LLM / evaluation / MCP layers)
- Directories: [fle/agents/](fle-package/fle/agents/), [fle/eval/inspect/](fle-package/fle/eval/inspect/), [fle/eval/algorithms/](fle-package/fle/eval/algorithms/), [fle/eval/open/](fle-package/fle/eval/open/), [fle/eval/analysis/](fle-package/fle/eval/analysis/), [fle/eval/entrypoints/](fle-package/fle/eval/entrypoints/), [data/plans/](fle-package/data/plans/), [data/prompts/](fle-package/data/prompts/).
- Files: [fle/mcp_dataloader.py](fle-package/fle/mcp_dataloader.py), [fle/overlay.py](fle-package/fle/overlay.py), [fle/overlay_mcp.py](fle-package/fle/overlay_mcp.py), [fle/server.py](fle-package/fle/server.py).
- LLM formatters: `fle/env/gym_env/observation_formatter.py`, `fle/env/gym_env/system_prompt_formatter.py`.
- LLM prompt generator: `fle/env/utils/controller_loader/system_prompt_generator.py`.
- LLM-only commons models: `fle/commons/models/conversation.py`, `fle/commons/models/message.py`, `fle/commons/models/serializable_function.py`.
- LLM tool docs: `fle/env/tools/agent.md` (top-level LLM example patterns file; per-tool `agent.md` files are harmless documentation and can stay).

#### 1b. Migrate `TaskResponse` out of `fle.agents`
`TaskResponse` is imported by 3 files outside `fle/agents/` that must survive:
- [fle/env/gym_env/observation.py](fle-package/fle/env/gym_env/observation.py) (line 8)
- [fle/env/gym_env/environment.py](fle-package/fle/env/gym_env/environment.py) (line 17)
- [fle/eval/tasks/task_abc.py](fle-package/fle/eval/tasks/task_abc.py) (line 4)

**Action**: copy the `TaskResponse` dataclass (and its `REWARD_OVERRIDE_KEY` constant) from `fle/agents/models.py` into `fle/commons/models/task_response.py`; update all three imports to `from fle.commons.models.task_response import TaskResponse`.

#### 1c. Fix broken package-level imports
- [fle/__init__.py](fle-package/fle/__init__.py) — remove `agents` from the `from fle import agents, env, eval, cluster, commons` line and from `__all__`.
- [fle/commons/__init__.py](fle-package/fle/commons/__init__.py) and [fle/commons/models/__init__.py](fle-package/fle/commons/models/__init__.py) — remove exports of `Conversation`, `Message`, `SerializableFunction`.

#### 1d. Strip LLM methods from kept files
- [fle/env/instance.py](fle-package/fle/env/instance.py) — **delete** `eval()`, `eval_with_error()`, `get_system_prompt()` methods; delete import of `SystemPromptGenerator`. Keep `reset()`, `background_step()`, `_reset()`, `GameControl` (pause/unpause/speed).
- [fle/env/namespace.py](fle-package/fle/env/namespace.py) (~1300 lines, ~1000 are LLM-specific) — **delete**: `eval_with_timeout()` (128 lines), all recursive AST execution helpers (`execute_body()`, `execute_node()`, ~650 lines), error-suggestion helpers (`_get_suggestions_from_name_error()`, `_extract_error_lines()`), `_change_print_to_log()`, `get_functions()`, `SerializableFunction` import and all references, builtins injection block, `LoopContext`, `_freeze_protected_names()`. **Keep**: constructor (tool wiring), `reset()`, `log()`, `player_location` cache, tool-dispatch methods (place_entity, get_entities, etc.), `load()` (but strip `unwrap_after_deserialization()` call and `SerializableFunction` filtering from it).
- [fle/eval/tasks/task_abc.py](fle-package/fle/eval/tasks/task_abc.py) — **delete** `enhance_response_with_task_output()` (LLM-specific text appending). Keep `verify()`, `setup_instance()`, `setup()`.
- [fle/env/gym_env/observation.py](fle-package/fle/env/gym_env/observation.py) — **remove** fields: `serialized_functions`, `messages` (LLM conversation), `raw_text` (LLM-oriented string). Keep structured fields: entities, inventory, research, game_info, score, flows, map_image, character_positions, task_verification.
- [fle/commons/models/game_state.py](fle-package/fle/commons/models/game_state.py) — in `filter_serializable_vars()`, remove `SerializableFunction`-aware logic; simplify to standard pickle filter.

#### 1e. Trim CLI and prune tools
- [fle/run.py](fle-package/fle/run.py) — **delete** `fle_eval()` and `fle_inspect_eval()` (~250 lines); remove `--eval` and `--inspect-eval` subcommands from argparse. Keep `fle init` + `fle cluster`.
- Prune unused tools under [fle/env/tools/agent/](fle-package/fle/env/tools/agent/) — **keep**: `place_entity`, `place_entity_next_to`, `pickup_entity`, `shift_entity`, `rotate_entity`, `can_place_entity`, `insert_item`, `extract_item`, `inspect_inventory`, `get_entity`, `get_entities`, `nearest`, `move_to`, `score`, `print`, `sleep`. **Delete or feature-flag**: `craft_item`, `connect_entities`, `get_prototype_recipe`, `set_entity_recipe`, `set_research`, `get_research_progress`, `harvest_resource`, `get_resource_patch`, `get_connection_amount`, `send_message`, `launch_rocket`, `nearest_buildable`.

#### 1f. Update dependencies
- [pyproject.toml](fle-package/pyproject.toml) — **delete**: `openai>=2.0.0`, `anthropic>=0.69.0`, `inspect-ai>=0.3.139`, `jinja2>=3.1.6` (used for LLM prompt templating). Delete optional groups `[agents]` and `[inspect]`. **Add**: `gymnasium`, `numpy`, `stable-baselines3` (optional extras group `[rl]`).

#### 1g. Delete broken tests (import from deleted modules)
The following test files import from modules being deleted and will fail:
- `tests/eval/test_python_parser.py`
- `tests/eval/_test_recursive_formatter_functional.py`
- `tests/eval/test_recursive_formatter.py`
- `tests/eval/test_save_load_python_namespace.py`
- `tests/eval/test_mcts_chunker.py`
- `tests/eval/samplers/test_kld_mean_sampler.py`
- `tests/eval/samplers/test_weighted_reward_sampler.py`
- `tests/eval/samplers/test_python_parser.py`
- `tests/eval/test_conversation_formatter.py`
- `tests/multiagent/test_messages.py` (imports `SimpleFactorioEvaluator` from `fle.eval.algorithms.independent`)
- `tests/gym_env/test_observation_formatter.py` (imports from `fle.agents`)

### Phase 2 — New low-level Lua/Python tools (parallel after Phase 1 §3)
Each follows the existing `(client.py, server.lua, agent.md)` triple under [fle/env/tools/](fle-package/fle/env/tools/).
1. `admin/spawn_enemy` — wraps `surface.create_entity{force='enemy'}`; replaces inline `/silent-command` from [tests/test_character_persistence.py](fle-package/tests/test_character_persistence.py).
2. `admin/radar_view` — **core observation primitive**: returns `uint8[C,H,W]` symbolic grid restricted to `force.is_chunk_charted(...)`. Channels: empty, wall, turret, ammo-loaded%, biter, spitter, spawner, character. Implemented via `surface.find_entities_filtered` + chart query, serialised base64, decoded to NumPy client-side.
3. `agent/shoot` — drives `LuaPlayer.shooting_state` / `character.update_selected_entity`. (No combat tool exists today.)
4. `admin/biter_director` — server-side spawn scheduler escalating wave size with `elapsed_ticks` (on `on_nth_tick`).
5. `admin/grant_items` — periodic ammo top-up (called by env, not agent), capped.
6. Audit [fle/env/mods/utils.lua](fle-package/fle/env/mods/utils.lua) — `remove_enemies()` must not run when peaceful=false; gate behind explicit flag to avoid reset races.

### Phase 3 — New gym env + task (depends on Phase 2)
1. New `fle/env/gym_env/td_environment.py` — `TowerDefenseEnv(gymnasium.Env)`, single-agent, **gymnasium not legacy gym**. Decision cadence: `on_nth_tick(C)` server-side flag; env unpauses, runs `C` ticks, pauses, fetches obs (matches project plan's "C*Nty tick").
2. **Observation space** (`Dict`):
   - `map`: `Box(0,255,(C,H,W),uint8)` from `radar_view`.
   - `inventory`: counts of firearm-magazine, piercing-rounds-magazine, gun-turret, stone-wall.
   - `turrets`: padded `(x,y,ammo,health)` table.
   - `walls`: padded `(x,y,health)`.
   - `character`: `(x,y,health,weapon_ammo)`.
   - `radar`: `(x,y,health)` (terminal if 0).
   - `game`: `(elapsed_ticks, wave_intensity_proxy)`.
3. **Action space** (`Dict`): `action_type ∈ Discrete(7)` {noop, place_wall, place_turret, move_wall, move_turret, refill_turret, shoot}; `target_xy`, `source_xy` `MultiDiscrete([W,H])`; `ammo_amount Discrete`. Invalid → no-op + small penalty.
4. **Reward**: `+α·tick_survived + β·Δkills − γ·ammo_used − δ·damage_taken − ε·invalid`; terminal bonus/penalty.
5. **Termination**: `radar.health<=0 or character.health<=0`; `truncated` at `max_ticks`.
6. New `TowerDefenseTask(TaskABC)` at `fle/eval/tasks/tower_defense_task.py` extending [fle/eval/tasks/task_abc.py](fle-package/fle/eval/tasks/task_abc.py): clears entities, places radar, seeds nest ring, snapshots `GameState` for `reset()` via [fle/commons/models/game_state.py](fle-package/fle/commons/models/game_state.py).
7. New scenario `fle/cluster/scenarios/tower_defense/control.lua` (mirrors [default_lab_scenario](fle-package/fle/cluster/scenarios/default_lab_scenario/) but peaceful=false, registers `on_nth_tick`). New aggressive `map-settings.td.json` / `map-gen-settings.td.json` under [fle/cluster/config/](fle-package/fle/cluster/config/); extend `ComposeGenerator` in [fle/cluster/run_envs.py](fle-package/fle/cluster/run_envs.py) to select scenario via env var.

### Phase 4 — Training infra (depends on Phase 3)
1. `fle/env/gym_env/td_vector.py` — `AsyncVectorEnv` mapping each sub-env to a distinct docker container (existing `ClusterManager` already spawns N replicas on ports `27000+i`).
2. Action-mask wrapper using `can_place_entity` + grid occupancy + inventory counts.
3. Reference `examples/rl/train_td_ppo.py` (SB3 PPO smoke test, not a feature).

### Phase 5 — Verification
1. Per-tool round-trip tests against live container: `tests/actions/test_spawn_enemy.py`, `test_shoot.py`, `test_radar_view.py`.
2. `tests/gym_env/test_td_environment.py` — space conformance, reset determinism, invalid-action penalty, terminal on `radar.die()`.
3. `tests/functional/test_td_scenario.py` — scripted policy survives ≥ 5000 ticks against light wave.
4. `python -c "import gymnasium as gym, fle.env.gym_env.td_environment; e=gym.make('factorio-td-v0'); e.reset(); e.step(e.action_space.sample())"` clean.
5. `examples/rl/train_td_ppo.py --total-steps 2000` runs, reward variance > 0.
6. Dev script: render `obs['map']` channels with matplotlib; biter motion visible only inside charted area.

## Relevant files

### Keep unchanged (no LLM code)
- [fle/env/tools/tool.py](fle-package/fle/env/tools/tool.py), [fle/env/tools/controller.py](fle-package/fle/env/tools/controller.py), [fle/env/tools/init.py](fle-package/fle/env/tools/init.py), [fle/env/tools/__init__.py](fle-package/fle/env/tools/__init__.py) — tool dispatch infra, no LLM patterns.
- [fle/env/lua_manager.py](fle-package/fle/env/lua_manager.py) — loads `server.lua` files only, does NOT load `agent.md`; no LLM code.
- [fle/env/entities.py](fle-package/fle/env/entities.py), [fle/env/game_types.py](fle-package/fle/env/game_types.py) — entity/prototype models, no LLM code.
- [fle/env/utils/achievements.py](fle-package/fle/env/utils/achievements.py), [fle/env/utils/camera.py](fle-package/fle/env/utils/camera.py), [fle/env/utils/rcon.py](fle-package/fle/env/utils/rcon.py) — pure game utilities.
- [fle/env/utils/controller_loader/](fle-package/fle/env/utils/controller_loader/) — keep `code_analyzer.py`, `schema_generator.py`, `type_definition_processor.py`, `module_loader.py`, `call_info.py` (useful for tool registry introspection); **delete only** `system_prompt_generator.py`.
- [fle/cluster/](fle-package/fle/cluster/) — docker/compose infra, no LLM code.
- [fle/commons/models/game_state.py](fle-package/fle/commons/models/game_state.py) (after removing `SerializableFunction` logic), [fle/commons/models/achievements.py](fle-package/fle/commons/models/achievements.py), [fle/commons/models/rendered_image.py](fle-package/fle/commons/models/rendered_image.py), [fle/commons/models/timing_metrics.py](fle-package/fle/commons/models/timing_metrics.py), [fle/commons/models/generation_parameters.py](fle-package/fle/commons/models/generation_parameters.py).
- [fle/env/a2a_instance.py](fle-package/fle/env/a2a_instance.py), [fle/env/a2a_namespace.py](fle-package/fle/env/a2a_namespace.py) — multi-agent comms, agent-type-agnostic; keep for future multi-agent RL.
- [fle/__main__.py](fle-package/fle/__main__.py) — simple entry point, no LLM code.

### Refactor (mixed LLM/core code — see Phase 1d)
- [fle/env/instance.py](fle-package/fle/env/instance.py) — delete `eval()`, `eval_with_error()`, `get_system_prompt()`; keep `reset()`, `background_step()`, `GameControl`.
- [fle/env/namespace.py](fle-package/fle/env/namespace.py) — delete ~1000 lines of AST execution / LLM error handling; keep tool wiring + reset + log.
- [fle/eval/tasks/task_abc.py](fle-package/fle/eval/tasks/task_abc.py) — delete `enhance_response_with_task_output()`; fix `TaskResponse` import.
- [fle/env/gym_env/observation.py](fle-package/fle/env/gym_env/observation.py) — remove `serialized_functions`, `messages`, `raw_text` fields; fix import.
- [fle/commons/models/game_state.py](fle-package/fle/commons/models/game_state.py) — remove `SerializableFunction` filtering.
- [fle/__init__.py](fle-package/fle/__init__.py) — remove `agents` from imports/exports.
- [fle/commons/__init__.py](fle-package/fle/commons/__init__.py), [fle/commons/models/__init__.py](fle-package/fle/commons/models/__init__.py) — remove deleted model exports.
- [fle/run.py](fle-package/fle/run.py) — delete `fle_eval()`, `fle_inspect_eval()`, associated argparse subcommands.
- [fle/eval/tasks/task_definitions/task_registry.py](fle-package/fle/eval/tasks/task_definitions/task_registry.py) — remove `unbounded_production` reference (Inspect-only), add `tower_defense → TowerDefenseTask` mapping.

### Rewrite from scratch
- [fle/env/gym_env/environment.py](fle-package/fle/env/gym_env/environment.py) → replaced by `td_environment.py`.
- [fle/env/gym_env/action.py](fle-package/fle/env/gym_env/action.py) — current `Action(code: str)` is purely LLM; replace with `TDAction(action_type: int, target_xy, source_xy, ammo_amount)`.
- [fle/env/gym_env/registry.py](fle-package/fle/env/gym_env/registry.py) — register only `factorio-td-v0`.

### Audit
- [fle/env/mods/utils.lua](fle-package/fle/env/mods/utils.lua) — `remove_enemies()` gating when `peaceful=false`.

### New files to create
- `fle/commons/models/task_response.py` — migrated `TaskResponse` + `REWARD_OVERRIDE_KEY` from deleted `fle/agents/models.py`.
- `fle/env/tools/admin/spawn_enemy/{client.py,server.lua}`
- `fle/env/tools/admin/radar_view/{client.py,server.lua}`
- `fle/env/tools/admin/biter_director/{server.lua,client.py}`
- `fle/env/tools/admin/grant_items/{client.py,server.lua}`
- `fle/env/tools/agent/shoot/{client.py,server.lua}`
- `fle/env/gym_env/td_environment.py`
- `fle/env/gym_env/td_spaces.py` (observation/action space helpers)
- `fle/env/gym_env/td_vector.py`
- `fle/eval/tasks/tower_defense_task.py`
- `fle/cluster/scenarios/tower_defense/control.lua`
- `fle/cluster/config/map-gen-settings.td.json`, `map-settings.td.json`
- `tests/gym_env/test_td_environment.py`, `tests/actions/test_spawn_enemy.py`, `test_shoot.py`, `test_radar_view.py`, `tests/functional/test_td_scenario.py`
- `examples/rl/train_td_ppo.py`

### Delete (LLM-only, broken after strip)
- **Directories**: `fle/agents/`, `fle/eval/inspect/`, `fle/eval/algorithms/`, `fle/eval/open/`, `fle/eval/analysis/`, `fle/eval/entrypoints/`, `data/plans/`, `data/prompts/`.
- **Files**: `fle/mcp_dataloader.py`, `fle/overlay.py`, `fle/overlay_mcp.py`, `fle/server.py`, `fle/env/gym_env/observation_formatter.py`, `fle/env/gym_env/system_prompt_formatter.py`, `fle/env/utils/controller_loader/system_prompt_generator.py`, `fle/commons/models/conversation.py`, `fle/commons/models/message.py`, `fle/commons/models/serializable_function.py`, `fle/env/tools/agent.md`.
- **Tests (import from deleted modules)**: `tests/eval/test_python_parser.py`, `tests/eval/_test_recursive_formatter_functional.py`, `tests/eval/test_recursive_formatter.py`, `tests/eval/test_save_load_python_namespace.py`, `tests/eval/test_mcts_chunker.py`, `tests/eval/samplers/test_kld_mean_sampler.py`, `tests/eval/samplers/test_weighted_reward_sampler.py`, `tests/eval/samplers/test_python_parser.py`, `tests/eval/test_conversation_formatter.py`, `tests/multiagent/test_messages.py`, `tests/gym_env/test_observation_formatter.py`.

## Decisions / scope
**In**: gymnasium single-agent env, symbolic-grid obs, new tools (spawn_enemy/radar_view/shoot/director/grant_items), one new task + scenario, strip LLM stack.
**Out**: multi-agent (existing [a2a_instance.py](fle-package/fle/env/a2a_instance.py) can be re-enabled later), pixel-vision obs (defer; symbolic first), pure construction-robot/ghost build mode (Phase 6+), curriculum.
**Assumption**: training uses one docker container per env (matches existing `ClusterManager`).

## Further considerations
1. **Action encoding** — A) `Dict` (clean, needs multi-input policy) / B) flat `MultiDiscrete([7,W,H,W,H,A])` (works with any algo). **Recommend A**, ship a flatten wrapper.
2. **Character-vs-bots build mode** — A) character-only / B) ghost-only (project-plan "PS" option using construction robots) / C) toggle. **Recommend C, default A.**
3. **Observation coverage** — A) strictly radar-charted (matches project intent, partial observability) / B) full grid. **Recommend A**, expose B as debug flag.
