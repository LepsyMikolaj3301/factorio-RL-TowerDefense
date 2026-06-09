## Plan: Support prebuilt Factorio save for RL experiments

TL;DR
Implement repository-level support to run RL experiments on a prebuilt Factorio save file (.zip). Because you do not need to switch maps per episode, prefer a minimal orchestration approach: start the headless Factorio server with the chosen save (via run scripts / container env var), then connect the existing Python env as-is. This provides full fidelity (terrain, resources, enemies) with small code changes and clear docs.

**Steps**
1. Add a `--save-path` CLI/config option so runs can declare a save to use. *(depends: none — small CLI change)*
2. Ensure the container/run scripts accept and forward the save file. Update `fle/cluster/run-envs.sh` to document/ensure `SAVE_FILE` is accepted and mounted into the container. Provide example `docker run` and `run-envs.sh` usage. *(parallel with step 1)*
3. Add a small convenience field in `FactorioInstance` to record the expected `save_path` and validate on startup (non-blocking warning). Update `fle/env/instance.py` to accept `save_path` in the constructor or `initialise()` and store it (no automatic server restart). *(depends on step 1)*
4. Minimal Python wiring: allow passing `save_path` from `fle/run.py` / `registry` into `FactorioInstance` so metadata is recorded. No runtime map switching required. *(depends on step 1 & 3)*
5. Docs & examples: add a README subsection and example config showing how to create a save, place it under `saves/`, and run experiments pointing to it. Include commands for mounting `saves/` into the container and launching the environment. *(parallel)*
6. Tests & validation: add an integration test that runs `FactorioInstance.initialise()` against a container started with a save and asserts successful RCON connection and expected top-level entities exist. Add a small unit test asserting `FactorioInstance` records `save_path` correctly.
7. Optional follow-ups (future work): implement in-repo automated server restart to switch saves at runtime, or implement lightweight in-game validation markers inside the save for stronger validation.

**Relevant files**
- `fle/cluster/run-envs.sh` — container startup / save mounting
- `fle/run.py` — add `--save-path` CLI flag and pass it into the environment creation path.
- `fle/env/gym_env/registry.py` — propagate `save_path` from run config / CLI into instance creation.
- `fle/env/instance.py` — add `save_path` storage and a small startup validation in `initialise()`; `FactorioInstance.reset()` and `initialise()` are here (see current reset at `fle/env/instance.py#L274`).
- `fle/env/gym_env/environment.py` — no functional change required for runtime resets if server already started with save, but optionally record `save_path` on env creation (`FactorioGymEnv` constructors at `fle/env/gym_env/environment.py#L223`, `reset()` at `fle/env/gym_env/environment.py#L533`).
- `fle/env/gym_env/registry.py` — propagate save path when creating instances ([fle/env/gym_env/registry.py](fle/env/gym_env/registry.py#L45))  

**Concrete patch notes (what to change)**
- `fle/run.py`: add `parser.add_argument('--save-path', help='Path to Factorio save zip to use for this run')`. When starting runs, store `save_path` in the run config passed to the registry.
- `fle/cluster/run-envs.sh`: surface `SAVE_FILE` env var in the startup path and mount a local `saves/` dir into the container. Document usage:

```bash
mkdir -p ./saves
cp /path/to/my_map.zip ./saves/my_map.zip
# start container with save mounted and env var pointing to it
export SAVE_FILE=/factorio/saves/my_map.zip
bash fle/cluster/run-envs.sh
```

- `fle/env/gym_env/registry.py`: when creating `FactorioInstance`, pass `save_path` from the run config (store it in `FactorioInstance` constructor or call `instance.set_save_path(save_path)`).
- `fle/env/instance.py`: add `self.save_path = save_path` (constructor) and, inside `initialise()`, after connecting via RCON, log `Connected to Factorio server using save: <path>` if provided. Optionally run a small Lua snippet to check existence of a known entity or flag if the repo can guarantee it.

**Verification**
1. Manual run (example):

```bash
# put your save into repo/saves
mkdir -p ./saves && cp /path/to/my_map.zip ./saves/
# start container (example using run-envs.sh which reads SAVE_FILE)
SAVE_FILE=/factorio/saves/my_map.zip bash fle/cluster/run-envs.sh
# start the python runner with save argument
python fle/run.py --env-id my_env --save-path ./saves/my_map.zip
```

2. Integration test: start container with the save and run a test that instantiates `FactorioInstance` and calls `initialise()`. Assert RCON responds and `initialise()` completes without error.
3. Confirm `env.reset()` returns the expected observation and that the RL loop can step through at least one episode tick.

**Decisions / rationale**
- You requested full-game resets with terrain/resources; the save-file approach gives exact fidelity and is simplest given no per-episode map switching requirements.
- Minimal code changes keep the system stable and rely on existing container orchestration to choose the map at server start.

**Further considerations**
- If later you need to run many different maps in one job (per-episode switching), implement blueprint-based dynamic loading or add server-restart orchestration from Python.
- If you want stronger validation, include a small mod or marker inside each save (e.g., set a global variable or place a uniquely-named entity) so `initialise()` can confirm the correct save is running.
