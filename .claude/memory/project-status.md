---
name: project-status
description: RL rewrite phase completion status, known issues, and what runs end-to-end
metadata:
  type: project
---

## What works end-to-end (as of 2026-06-05)

- `import fle` — clean, no broken imports
- `test_td_spaces` — 8/8 unit tests pass (no server needed)
- Factorio container starts with `tower_defense` scenario via docker run (image: `factoriotools/factorio:2.0.73`)
- RCON connects at port 27000; all admin/agent Lua tools load successfully
- `TowerDefenseEnv.reset()` + `.step()` work against live server
- Short PPO smoke test (512 steps) completes; model saved to `./td_ppo_model`

## How to start the server manually

```bash
CLUSTER_DIR=/path/to/fle-package/fle/cluster
docker run -d --name factorio_td_0 \
  -p 34197:34197/udp -p 27000:27015/tcp \
  -v "${CLUSTER_DIR}/scenarios:/opt/factorio/scenarios:ro" \
  -v "${CLUSTER_DIR}/config:/opt/factorio/config:ro" \
  -v "${CLUSTER_DIR}/mods:/opt/factorio/mods" \
  -v "/home/miki2/.fle/data/_screenshots:/opt/factorio/script-output" \
  --user root --entrypoint /bin/sh factoriotools/factorio:2.0.73 \
  -c 'rm -rf /opt/factorio/data/elevated-rails /opt/factorio/data/quality /opt/factorio/data/space-age && exec /opt/factorio/bin/x64/factorio --start-server-load-scenario tower_defense --port 34197 --server-settings /opt/factorio/config/server-settings.json --map-gen-settings /opt/factorio/config/map-gen-settings.td.json --map-settings /opt/factorio/config/map-settings.td.json --server-banlist /opt/factorio/config/server-banlist.json --rcon-port 27015 --rcon-password "factorio" --server-whitelist /opt/factorio/config/server-whitelist.json --use-server-whitelist --server-adminlist /opt/factorio/config/server-adminlist.json --mod-directory /opt/factorio/mods --map-gen-seed 44340'
```

## How to run PPO training

```bash
cd fle-package
FACTORIO_SERVER_ADDRESS=localhost FACTORIO_SERVER_PORT=27000 \
  python3 examples/rl/train_td_ppo.py --total-timesteps 512 --n-steps 64 --batch-size 32
```

## Known issues / observations

- Radar placement warning on first `setup_instance()` call ("empty" at 0,0) — radar is also placed in `reset()` so this is benign
- `ep_len_mean=2` — episodes terminate after 2 steps because radar=0 health (no radar placed before first obs). Needs proper radar placement before first obs.
- ~2 FPS — each step takes ~0.5s (RCON round-trip + game tick wait). Expected for single container.
- `_reset_elapsed_ticks()` returns elapsed_ticks=0 after reset, but the game timer still runs; needs server-side Lua to actually reset.

## Phases complete

1. Strip & fix imports ✅ (fixed residual broken imports in this session)
2. New Lua/Python tools ✅
3. New gym env + task ✅
4. Training infra (train_td_ppo.py + FlatTDActionWrapper) ✅
5. Verification — unit tests pass; integration tests require live server

**Why:** RL project to train agents to defend Factorio radar from biter waves, no LLM involvement.
**How to apply:** When suggesting next steps, focus on improving episode length (radar placement fix) and training throughput (game speed, vectorized envs).
