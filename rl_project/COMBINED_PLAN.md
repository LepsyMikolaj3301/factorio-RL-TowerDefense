# Combined Plan: Tower-Defense RL Rewrite + Prebuilt Save Support

## Codebase findings (filled in from actual code)

- `fle/commons/__init__.py` does NOT exist — no removals needed there
- Registry and gym env use `gym` (legacy OpenAI Gym), not `gymnasium` — migration needed
- `ComposeGenerator` already supports `save_file=` parameter and `_copy_save()` — save plan is simpler
- `REWARD_OVERRIDE_KEY` lives in `fle/commons/constants.py`, NOT in `fle/agents/models.py`
- `fle/run.py` imports `from fle.agents.data.sprites.download import download_sprites_from_hf, generate_sprites` at top — must fix
- `TaskResponse` is imported via `from fle.agents import TaskResponse` in 5 surviving files
- `fle/commons/models/game_state.py` has NO SerializableFunction references — no change needed there
- `filter_serializable_vars()` in game_state.py filters by pickle-ability (no SF logic)
- `fle/env/namespace.py` uses `SerializableFunction` directly (import + eval_with_timeout + load + get_functions + wrap/unwrap helpers)
- `fle/eval/tasks/unbounded_throughput_task.py`, `default_task.py`, `throughput_task.py` also import `TaskResponse` from `fle.agents`

---

## Phase 1 — Strip & fix imports

### 1a. Delete directories & files
**Directories to delete:**
- `fle/agents/`
- `fle/eval/inspect/`
- `fle/eval/algorithms/`
- `fle/eval/open/`
- `fle/eval/analysis/`
- `fle/eval/entrypoints/`
- `data/plans/`
- `data/prompts/`

**Files to delete:**
- `fle/mcp_dataloader.py`
- `fle/overlay.py`
- `fle/overlay_mcp.py`
- `fle/server.py`
- `fle/env/gym_env/observation_formatter.py`
- `fle/env/gym_env/system_prompt_formatter.py`
- `fle/env/utils/controller_loader/system_prompt_generator.py`
- `fle/commons/models/conversation.py`
- `fle/commons/models/message.py`
- `fle/commons/models/serializable_function.py`
- `fle/env/tools/agent.md`

### 1b. Migrate TaskResponse out of fle.agents
Create `fle/commons/models/task_response.py` containing `TaskResponse` (from `fle/agents/models.py`).

Update imports in these surviving files:
- `fle/eval/tasks/task_abc.py` (line 4)
- `fle/eval/tasks/throughput_task.py` (line 7)
- `fle/eval/tasks/default_task.py` (line 4)
- `fle/eval/tasks/unbounded_throughput_task.py` (line 7)
- `fle/env/gym_env/observation.py` (line 8)
- `fle/env/gym_env/environment.py` (line 17) — this file gets rewritten anyway

### 1c. Fix package-level imports
- `fle/__init__.py`: remove `agents` from imports and `__all__`
- `fle/commons/models/__init__.py`: remove exports of `Conversation`, `Message`, `SerializableFunction`; add `TaskResponse`

### 1d. Strip LLM methods from kept files
- `fle/env/instance.py`: delete `eval()`, `eval_with_error()`, `__eval_with_error()`, `get_system_prompt()`, `_extract_partial_output()`; delete `SystemPromptGenerator` import; delete `ThreadPoolExecutor` import and `self._executor` usage
- `fle/env/namespace.py`: delete `eval_with_timeout()`, `execute_body()`, `execute_node()`, `_get_suggestions_from_name_error()`, `_extract_error_lines()`, `_change_print_to_log()`, `get_functions()`, `_check_protected()`, `_freeze_protected_names()`, `LoopContext` class, `SerializableFunction` import, `wrap_for_serialization()`, `unwrap_after_deserialization()`. In `load()`, remove `unwrap_after_deserialization()` call. In constructor, remove `LoopContext` init. In `reset()`, remove `loop_context` reset. Remove `self.capture_whole_output` and `self.execution_trace` (only used by eval_with_timeout). Remove builtins injection block. Keep: constructor (tool wiring), `reset()`, `log()`, `_static_members`, entity/direction/math/type-hint setup, `__getitem__`, `__setitem__`, `_assign_target`, `get_messages`, `load_messages`.
- `fle/eval/tasks/task_abc.py`: delete `enhance_response_with_task_output()`. Fix `TaskResponse` import path.
- `fle/env/gym_env/observation.py`: remove `raw_text`, `serialized_functions`, `messages` fields. Fix `TaskResponse` import. (Keep `from_dict`/`to_dict` but strip those fields.)
- `fle/run.py`: delete `fle_eval()`, `fle_inspect_eval()`, `fle_sandbox()`, `fle_sprites()` and all associated argparse subcommands. Remove `from fle.agents.data.sprites.download` import. Keep `fle init` + `fle cluster`. Add `--save-path` option (from prebuilt-save plan).

### 1e. Prune agent tools
**Keep**: `place_entity`, `place_entity_next_to`, `pickup_entity`, `shift_entity`, `rotate_entity`, `can_place_entity`, `insert_item`, `extract_item`, `inspect_inventory`, `get_entity`, `get_entities`, `nearest`, `move_to`, `score`, `print`, `sleep`
**Delete**: `craft_item`, `connect_entities`, `get_prototype_recipe`, `set_entity_recipe`, `set_research`, `get_research_progress`, `harvest_resource`, `get_resource_patch`, `get_connection_amount`, `send_message`, `launch_rocket`, `nearest_buildable`

### 1f. Update dependencies
- `pyproject.toml`: remove `openai`, `anthropic`, `inspect-ai`, `jinja2`, `a2a-sdk`, `mcp`, `hf`, `psycopg2-binary`, `fastapi`, `uvicorn`, `huggingface-hub`. Remove optional groups `agents`, `inspect`, `mcp`, `env`. Replace `gym` with `gymnasium`. Add `stable-baselines3` to new optional `[rl]` group.

### 1g. Delete broken tests
- `tests/eval/test_python_parser.py`
- `tests/eval/_test_recursive_formatter_functional.py`
- `tests/eval/test_recursive_formatter.py`
- `tests/eval/test_save_load_python_namespace.py`
- `tests/eval/test_mcts_chunker.py`
- `tests/eval/test_conversation_formatter.py`
- `tests/eval/samplers/test_kld_mean_sampler.py`
- `tests/eval/samplers/test_weighted_reward_sampler.py`
- `tests/eval/samplers/test_python_parser.py`
- `tests/multiagent/test_messages.py`
- `tests/gym_env/test_observation_formatter.py`

---

## Phase 2 — New Lua/Python tools (after Phase 1)

Each follows `(client.py, server.lua)` triple under `fle/env/tools/`.

1. **`admin/spawn_enemy`** — wraps `surface.create_entity{force='enemy'}` for spawning biters/spitters/spawners at coordinates.
2. **`admin/radar_view`** — returns `uint8[C,H,W]` symbolic grid. Channels: empty, wall, turret, ammo-loaded%, biter, spitter, spawner, character. Uses `surface.find_entities_filtered` + `force.is_chunk_charted()`. Base64 encoded, decoded to NumPy.
3. **`agent/shoot`** — drives `LuaPlayer.shooting_state` for a given target position.
4. **`admin/biter_director`** — server-side spawn scheduler; escalating wave size with `elapsed_ticks` via `on_nth_tick`.
5. **`admin/grant_items`** — periodic ammo/resource top-up, capped.
6. **Audit `fle/env/mods/utils.lua`** — `remove_enemies()` should be gated: only run when `storage.peaceful == true`.

---

## Phase 3 — New gym env + task + save support (depends on Phase 2)

### 3a. New TowerDefenseEnv
- `fle/env/gym_env/td_environment.py` — `TowerDefenseEnv(gymnasium.Env)`, single-agent.
- Decision cadence: `on_nth_tick(C)` server-side; env unpauses, runs C ticks, pauses, fetches obs.

### 3b. Observation space (Dict)
- `map`: `Box(0,255,(C,H,W),uint8)` from radar_view
- `inventory`: counts of firearm-magazine, piercing-rounds-magazine, gun-turret, stone-wall
- `turrets`: padded `(x,y,ammo,health)` table
- `walls`: padded `(x,y,health)`
- `character`: `(x,y,health,weapon_ammo)`
- `radar`: `(x,y,health)` (terminal if 0)
- `game`: `(elapsed_ticks, wave_intensity_proxy)`

### 3c. Action space (Dict)
- `action_type ∈ Discrete(7)`: noop, place_wall, place_turret, move_wall, move_turret, refill_turret, shoot
- `target_xy`, `source_xy`: `MultiDiscrete([W,H])`
- `ammo_amount`: `Discrete`
- Invalid → no-op + small penalty

### 3d. Reward
- `+α·tick_survived + β·Δkills − γ·ammo_used − δ·damage_taken − ε·invalid`
- Terminal bonus/penalty

### 3e. Termination
- `radar.health<=0` or `character.health<=0`
- `truncated` at `max_ticks`

### 3f. TowerDefenseTask
- `fle/eval/tasks/tower_defense_task.py` extending TaskABC
- Clears entities, places radar, seeds nest ring, snapshots GameState for reset()

### 3g. Tower defense scenario
- `fle/cluster/scenarios/tower_defense/control.lua` — peaceful=false, registers `on_nth_tick`
- New `map-gen-settings.td.json` / `map-settings.td.json` under `fle/cluster/config/`
- Extend `ComposeGenerator` in `fle/cluster/run_envs.py` to select config per scenario

### 3h. Save file support (from prebuilt-save plan)
- `--save-path` CLI option in `fle/run.py`
- `save_path` field in `FactorioInstance.__init__()` / `initialise()`
- Pass `save_path` through registry → instance creation
- Existing `ComposeGenerator.save_file` already handles container mounting

### 3i. Registry
- `fle/env/gym_env/td_registry.py` — register only `factorio-td-v0`
- Rewrite `fle/env/gym_env/registry.py` to use `gymnasium` and support both legacy tasks + TD

---

## Phase 4 — Training infra + tests (depends on Phase 3)

### 4a. Vector env
- `fle/env/gym_env/td_vector.py` — `AsyncVectorEnv` mapping sub-envs to docker containers

### 4b. Action mask wrapper
- Using `can_place_entity` + grid occupancy + inventory counts

### 4c. Reference training script
- `examples/rl/train_td_ppo.py` (SB3 PPO smoke test)

### 4d. Unit tests (no integration tests)
- `tests/tools/test_spawn_enemy.py` — test tool client construction and param validation
- `tests/tools/test_radar_view.py` — test decode logic (base64 → numpy) with mock RCON
- `tests/tools/test_shoot.py` — test param validation
- `tests/gym_env/test_td_environment.py` — space conformance, action encoding, reward calc
- `tests/gym_env/test_td_spaces.py` — obs/action space shape validation
- `tests/tasks/test_tower_defense_task.py` — task config, inventory, verify method
- `tests/test_save_path.py` — FactorioInstance records save_path correctly
- `tests/test_task_response_migration.py` — TaskResponse importable from new location
