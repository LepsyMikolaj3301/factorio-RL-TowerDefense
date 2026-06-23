# Tower Defense RL — Environment, Models & Rewards

A detailed reference for the reinforcement-learning stack that lives in
`fle-package/fle`. It covers three things:

1. **How the Gym environment works** — the `TowerDefenseEnv`, its observation
   and action spaces, the slot/anchor model, the threat model, action masking,
   and the reset/step lifecycle.
2. **The two policy architectures** — the feedforward **MaskablePPO** model
   (v1) and the recurrent **LSTM + Maskable PPO** model (v2), both in detail.
3. **The training objective and reward function** — standard PPO/GAE training,
   no regret term, and every environment reward term with the actual weights.

Source files referenced throughout:

| Concern | File |
| --- | --- |
| Environment | [td_environment.py](../fle/env/gym_env/td_environment.py) |
| Spaces | [td_spaces.py](../fle/env/gym_env/td_spaces.py) |
| Scenario config / reward weights | [td_config.py](../fle/env/gym_env/td_config.py) |
| Action masking wrapper | [action_mask.py](../fle/env/gym_env/action_mask.py) |
| v1 extractor + pointer head | [policy.py](../fle/rl/policy.py) |
| v1 MaskablePPO training | [train.py](../fle/rl/train.py) |
| v2 recurrent masked model | [lstm_policy.py](../fle/rl/lstm_policy.py) |
| v2 recurrent config | [lstm_config.py](../fle/rl/lstm_config.py) |
| v2 hand-rolled PPO loop | [train_lstm.py](../fle/rl/train_lstm.py) |

---

## 1. The Gym Environment

`TowerDefenseEnv` ([td_environment.py:62](../fle/env/gym_env/td_environment.py#L62))
is a standard `gymnasium.Env`. The agent defends a **radar** (the "heart" of the
base) and its surrounding **boilers** from waves of biters by managing a fixed
set of **gun-turret slots**, refilling their ammo, and moving its character
between **anchor tiles**.

### 1.1 Core loop & cadence

Each `step()` does the following ([td_environment.py:1126](../fle/env/gym_env/td_environment.py#L1126)):

```
1. _execute_action(action)          # place/pick/refill a turret, or start a walk
2. instance.unpause()               # let the game run...
   sleep(decision_cadence / 60 / game_speed)
   instance.pause()                 # ...for `decision_cadence` game ticks
3. _read_walk_state()               # advance async character movement
4. _read_events()                   # pull server-side kill/loss counters
5. _get_observation()               # build the obs dict
6. _compute_reward(...)             # score the window
7. _check_terminated() / truncated  # radar dead? char dead? boilers lost? time up?
```

Key timing knobs ([td_config.py:38](../fle/env/gym_env/td_config.py#L38)):

- `decision_cadence = 60` ticks between agent decisions (1 in-game second @ 60 tps).
- `game_speed = 10.0` — the server runs 10× wall-clock speed, so a 60-tick
  window only blocks the Python process for `60/60/10 = 0.1 s`.
- `max_ticks = 108_000` (30 in-game minutes) → episode **truncation**.

Because the game runs *between* decisions, biters keep attacking while the
character walks and while the agent "thinks". Movement is genuinely costly.
The current predefined-save workflow does **not** spawn Python-driven waves at
runtime; attacks come from the baked-in Factorio spawners/nests in the loaded map.

### 1.2 The slot / anchor model

The environment does **not** allow free-coordinate placement. Instead:

- **Turret slots** are read once from the map (every pre-placed `gun-turret`
  center becomes a canonical slot, deduped within `0.6` tiles). Stored in
  `_turret_slots`, shape `(N, 2)`, capped at `MAX_SLOTS = 64`
  ([td_spaces.py:37](../fle/env/gym_env/td_spaces.py#L37)).
- **Anchor tiles** are read once from the map by scanning for
  `hazard-concrete-left` tiles. These are the **only** positions the character
  may stand on. Stored in `_anchor_slots`, capped at `MAX_ANCHORS = 32`.
- A static **reach matrix** `(MAX_ANCHORS, MAX_SLOTS)` is precomputed: entry
  `[i, j] = 1` iff anchor `i` is within `REACH_DISTANCE = 10.0` tiles of slot
  `j` ([td_environment.py:851](../fle/env/gym_env/td_environment.py#L851)). A
  turret action is only legal when the character's current anchor can reach the
  target slot.

At the start of every episode, `_delete_turret_subset()` destroys a random
fraction (`turret_deletion_percentage`, default `0.5`) of slot turrets and hands
those turrets to the agent's inventory — so the agent must re-place exactly the
turrets that were removed ([td_environment.py:1011](../fle/env/gym_env/td_environment.py#L1011)).
All slot positions remain known; only their occupancy changes.

### 1.3 Observation space

A `spaces.Dict` ([td_spaces.py:94](../fle/env/gym_env/td_spaces.py#L94)). All
positions are emitted **relative to the radar center** and normalized by the
grid radius so the network sees well-conditioned inputs.

| Key | Shape | Meaning |
| --- | --- | --- |
| `map` | `(8, 64, 64)` uint8 | Symbolic radar grid. Channels: empty, wall, turret, ammo%, biter, spitter, spawner, character. Partial observability — only charted area. |
| `inventory` | `(4,)` int32 | Counts of `firearm-magazine`, `piercing-rounds-magazine`, `gun-turret`, `stone-wall`. |
| `turret_slots` | `(64, 5)` | Per slot: `[x, y, occupied, ammo, health]`. `occupied = -1` for padding rows. |
| `slot_valid_mask` | `(64,)` | Real (non-padding) slots. |
| `place_slot_mask` / `refill_slot_mask` / `pick_slot_mask` | `(64,)` | Real slots valid for each action *type* (empty / occupied). |
| `place_reach_mask` / `refill_reach_mask` / `pick_reach_mask` | `(64,)` | The above **AND** `reach_slot_mask` (zeroed entirely while moving). |
| `reach_slot_mask` | `(64,)` | Slots within `REACH_DISTANCE` of the current anchor. |
| `anchors` | `(32, 2)` | Anchor tile centers (relative, normalized). |
| `anchor_valid_mask` | `(32,)` | Real anchors. Doubles as the move-target mask. |
| `biter_groups` | `(12, 10)` | Live enemy clusters. Per group: `[rel_cx, rel_cy, count_norm, spread_norm, head_dx, head_dy, dist_radar_norm, dist_turret_norm, eta_norm, is_swarm]`. |
| `group_valid_mask` | `(12,)` | Real groups. |
| `nests` | `(8, 6)` | Biter spawners: `[rel_cx, rel_cy, health_norm, dist_radar_norm, dir_dx, dir_dy]`. |
| `nest_valid_mask` | `(8,)` | Real nests. |
| `boilers` | `(32, 4)` | Critical power structures: `[rel_x, rel_y, alive, health_norm]`. |
| `boiler_valid_mask` | `(32,)` | Real boilers. |
| `movement` | `(5,)` | `[is_moving, target_anchor_norm, remaining_dist_norm, head_dx, head_dy]`. |
| `recent_losses` | `(8,)` | Walls then turrets lost N/E/S/W in the last window, clipped/normalized. |
| `character` | `(4,)` | `[x, y, health, weapon_ammo]`. |
| `radar` | `(3,)` | `[x, y, health]`. |
| `game` | `(2,)` | `[elapsed_ticks_norm, wave_intensity_proxy]`. |

Normalization constants live at [td_spaces.py:82](../fle/env/gym_env/td_spaces.py#L82)
(`COUNT_NORM=100`, `ETA_NORM=600`, `CHAR_HP_NORM=250`, `TURRET_HP_NORM=400`, …).

### 1.4 Action space

A `spaces.Dict` ([td_spaces.py:207](../fle/env/gym_env/td_spaces.py#L207)) with
four factors. A `FlatTDActionWrapper` flattens it to a `MultiDiscrete` for
algorithms that need a non-Dict space:

```python
spaces.Dict({
    "action_type":  Discrete(6),    # see below
    "slot_index":   Discrete(64),   # which turret slot
    "ammo_amount":  Discrete(51),   # 0..50 magazines (refill/take-ammo only)
    "anchor_index": Discrete(32),   # which anchor tile (move only)
})
# flat MultiDiscrete: [6, 64, 51, 32]  → 153 total logits
```

The six action types ([td_spaces.py:16](../fle/env/gym_env/td_spaces.py#L16)):

| ID | Name | Effect |
| --- | --- | --- |
| 0 | `NOOP` | Do nothing. Always valid. |
| 1 | `PICK_TURRET` | Mine the turret in `slot_index` back into inventory. |
| 2 | `PLACE_TURRET` | Place an inventory turret into the empty `slot_index`. |
| 3 | `REFILL_TURRET` | Insert `max(1, ammo_amount)` magazines into the turret in `slot_index`. |
| 4 | `MOVE_ANCHOR` | Start an async A\*-planned walk to `anchor_index`. |
| 5 | `TAKE_AMMO` | Remove `max(1, ammo_amount)` configured magazines from the turret in `slot_index` without mining it. |

`_execute_action()` ([td_environment.py:1315](../fle/env/gym_env/td_environment.py#L1315))
returns `True` for an **invalid** action (out-of-range slot, occupied slot for
PLACE, empty slot for PICK/REFILL/TAKE_AMMO, a turret with no removable configured
ammo for TAKE_AMMO, or any turret action issued while in transit), which feeds
the invalid-action penalty. Walls and direct shooting exist in the
engine but are intentionally **not** exposed to the policy.

#### Asynchronous movement

`MOVE_ANCHOR` does not teleport. `_move_to_anchor()`
([td_environment.py:1405](../fle/env/gym_env/td_environment.py#L1405)) plans a
path (Factorio A\* via `request_path`, straight-line fallback) and hands it to a
server-side walker that glides the character `walk_speed = 0.2` tiles/tick during
the unpaused window. The agent may re-route mid-walk, but **cannot service
turrets while moving** — the reach masks are all-zero in transit.

### 1.5 Action masking

The policy never sees an illegal action thanks to `ActionMaskWrapper`
([action_mask.py:24](../fle/env/gym_env/action_mask.py#L24)). It augments the obs
with `action_type_mask`, `slot_mask`, `anchor_mask`, and exposes
`action_masks()` returning the **full flat 153-length mask**
`[type(6) | slot(64) | ammo(51) | anchor(32)]` consumed by both training paths.

Masking rules ([action_mask.py:56](../fle/env/gym_env/action_mask.py#L56)):

- `NOOP` always valid; `MOVE_ANCHOR` valid whenever an anchor exists.
- **While moving:** only `NOOP` and `MOVE_ANCHOR` are offered.
- `PLACE` valid only with a turret in inventory **and** a reachable empty slot.
- `REFILL` valid only with ammo **and** a reachable occupied slot.
- `PICK` valid only with a reachable occupied slot.
- `TAKE_AMMO` valid only with a reachable occupied turret that has removable
  configured ammo.
- The `ammo` factor is unconstrained; slot/anchor factors fall back to index 0
  when nothing is constrained (so each factor always has ≥1 valid option, as
  MaskablePPO requires).

### 1.6 Reset & save-state lifecycle

The env uses a **predefined Factorio save** rather than per-episode map
generation ([td_environment.py:203](../fle/env/gym_env/td_environment.py#L203)):

- **First reset:** configure speed, pause, read the radar/boilers/nests/turret
  slots/anchor tiles, build the reach matrix, print a one-time save-validation
  report, then capture a `GameState` snapshot with all turrets present.
- **Subsequent resets:** restore from the snapshot, then explicitly re-wipe
  TD-specific Lua `storage` (walk state, event counters, threat cache — these
  are **not** captured by `GameState`), re-apply difficulty knobs (evolution
  factor, group size), restore starting nests / infinity-chest filters, and
  re-roll which slots start empty.

`step()` is wrapped to survive RCON drops (a human spectator joining, or a
server crash+restart): it reconnects, pauses, and ends the episode cleanly with
a death-equivalent penalty rather than killing the whole training run
([td_environment.py:1083](../fle/env/gym_env/td_environment.py#L1083)).

### 1.7 Termination

`_check_terminated()` ([td_environment.py:2047](../fle/env/gym_env/td_environment.py#L2047)):

- Radar HP ≤ 0 (or a `radar_lost` event), **or**
- ≥ `boiler_loss_fraction` (default ⅓) of the initial boilers destroyed, **or**
- character HP ≤ 0 (or a `char_died` event).

**Truncation** happens when `elapsed_ticks >= max_ticks`.

---

## 2. The Policy Architectures

There are **two** models. Both share the same feature extractor (`TDExtractor`)
and the same flat 153-logit masked action space; the difference is purely the
core that sits between the extractor and the heads.

| | v1 — MaskablePPO | v2 — LSTM Maskable PPO |
| --- | --- | --- |
| File | [policy.py](../fle/rl/policy.py) | [lstm_policy.py](../fle/rl/lstm_policy.py) |
| Core | feedforward MLP | recurrent LSTM |
| Library | sb3-contrib `MaskablePPO` | hand-rolled CleanRL-style loop |
| Memory | Markov (current obs only) | hidden state across timesteps |
| Trainer | [train.py](../fle/rl/train.py) | [train_lstm.py](../fle/rl/train_lstm.py) |

### 2.1 Shared feature extractor — `TDExtractor`

[policy.py:94](../fle/rl/policy.py#L94). A multi-input extractor that fuses
heterogeneous observation components into a single latent of `features_dim = 256`:

```
map (8×64×64) ─► CoordConv CNN ─► Linear ──┐ (cnn_dim=128)
turret_slots  ─► DeepSets MLP ─► pool ──────┤
anchors       ─► DeepSets MLP ─► pool ──────┤
biter_groups  ─► DeepSets MLP ─► pool ──────┼─► concat ─► Linear+ReLU ─► latent(256)
nests         ─► DeepSets MLP ─► pool ──────┤
boilers       ─► DeepSets MLP ─► pool ──────┤
scalars       ─► MLP ──────────────────────┘ (inventory, character, radar,
                                               game, movement, recent_losses)
```

Design details:

- **CoordConv CNN** — two extra channels of normalized x/y coordinates are
  concatenated to the 8-channel map before three stride-2 conv layers, giving the
  CNN explicit spatial awareness ([policy.py:153](../fle/rl/policy.py#L153)).
- **DeepSets pooling** (`masked_mean_max_pool`,
  [policy.py:48](../fle/rl/policy.py#L48)) — every set input (slots, anchors,
  groups, nests, boilers) is embedded element-wise then pooled by concatenating
  a **masked mean** and a **masked max**. Padding rows are excluded; all-padding
  sets pool to zeros. This makes the model **permutation-invariant** and able to
  generalize across maps with different element counts.
- **PointerHead** ([policy.py:66](../fle/rl/policy.py#L66)) — an attention-style
  scorer (`logits = (Wk·emb · Wq·query)·scale`) that produces one logit per set
  element. It's the building block for permutation-aware slot/anchor selection;
  the extractor caches `last_slot_emb` / `last_anchor_emb` so a pointer policy
  can reuse them.

### 2.2 v1 — Feedforward MaskablePPO

`make_td_policy_kwargs()` ([policy.py:187](../fle/rl/policy.py#L187)) plugs
`TDExtractor` into sb3-contrib's `MaskablePPO` with a `MultiInputPolicy`:

```python
MaskablePPO(
    "MultiInputPolicy", venv,
    policy_kwargs=dict(
        features_extractor_class=TDExtractor,
        features_extractor_kwargs=dict(features_dim=256, cnn_dim=128, set_dim=64),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),   # separate actor/critic MLPs
    ),
    n_steps=1024, batch_size=256, n_epochs=…, gamma=…, clip_range=…,
)
```

- The fused 256-d latent feeds a standard actor MLP `[256, 128]` and critic MLP
  `[256, 128]`.
- **Masking** is handled by `MaskablePPO`: at every forward pass it calls the
  env's `action_masks()` and sets logits of invalid actions to `-∞` **per
  discrete factor independently** (type, slot, ammo, anchor masked separately).
- Because factors are masked independently, MaskablePPO cannot condition the
  slot mask on the chosen action type. Residual cross-factor illegality (e.g.
  REFILL on an empty slot) is caught at execution time via the invalid-action
  penalty.
- Training is the standard SB3 PPO update: clipped surrogate objective, clipped
  value loss, GAE, entropy bonus — all driven by `model.learn()` in
  [train.py:249](../fle/rl/train.py#L249). Vectorization is over Docker
  containers via `SubprocVecEnv` + `VecMonitor`.
- On resume, the trainer checks that the checkpoint's first `MultiDiscrete`
  factor matches the current `NUM_ACTION_TYPES = 6`; older 5-action checkpoints
  must be retrained or loaded with compatible code.

**Limitation that motivates v2:** this model is **Markov** — it only sees the
current observation. It cannot remember which direction the last swarm came
from, how long ago it refilled a turret, or that it is mid-plan.

### 2.3 v2 — Recurrent (LSTM) Maskable PPO

The recurrent model adds memory. The catch (documented in the module header,
[lstm_policy.py:1](../fle/rl/lstm_policy.py#L1)):

> sb3-contrib's `MaskablePPO` has **no recurrence**, and SB3's `RecurrentPPO`
> has **no masking**. No off-the-shelf class does both.

So v2 is a hand-rolled module + a CleanRL-style PPO loop that fuses the two.

#### Network: `RecurrentMaskableActorCritic`

[lstm_policy.py:47](../fle/rl/lstm_policy.py#L47):

```
obs ─► TDExtractor (reused verbatim) ─► features(256)
                                          │
                                          ▼
                              nn.LSTM(256 → 256, 1 layer)
                                          │ hidden(256)
                        ┌─────────────────┼──────────────────┐
                        ▼                 ▼                  ▼
              4 masked actor heads                     critic head
        Linear(256→6)  Linear(256→64)              Linear(256→1)
        Linear(256→51) Linear(256→32)
```

- `TDExtractor` is **imported and never edited** — the *only* difference from
  the v1 baseline is the LSTM core, making this a clean ablation.
- One `nn.Linear` actor head per action factor: `[6, 64, 51, 32]` logits = 153
  total ([lstm_policy.py:35](../fle/rl/lstm_policy.py#L35)). Heads use small init
  std (0.01) so early logits are near-uniform; critic uses std 1.0.
- LSTM weights are orthogonally initialized; biases zeroed.

#### The single fused function: `get_action_and_value`

[lstm_policy.py:124](../fle/rl/lstm_policy.py#L124). This **one** function is
used **identically** at rollout (`T == 1`) and at the PPO update (full sequence),
which keeps the three masked-recurrent fusion hazards inspectable in one place:

1. **Masks are applied on every call** — each head's logits are
   `masked_fill(mask == 0, -1e8)` before building the `Categorical`. So the
   log-prob ratio and the entropy are always computed over the **masked**
   distribution, at both rollout and update.
2. **Mask is reapplied per timestep** inside the sequence replay — `get_states`
   loops over `T`, and the head loop re-masks each row.
3. **Only the action-validity mask exists** — because each env's full rollout is
   replayed as one whole sequence (no padding), there is no sequence-padding mask
   that could be conflated with the action mask.

At rollout it samples (or argmaxes if `deterministic`); at update it **scores
stored actions** (passes `action=…`) to recompute new log-probs/entropy under the
current parameters.

#### Recurrence & hidden-state reset: `get_states`

[lstm_policy.py:95](../fle/rl/lstm_policy.py#L95). The model operates on a
flattened `(T*B, …)` batch where the flat index is **T-major**
(`index = t*B + b`). `get_states` reshapes back to `(T, B)` and rolls the LSTM
through time, zeroing the hidden/cell state wherever an episode starts:

```python
for f_t, d_t in zip(feat, done):          # iterate over T timesteps
    keep = (1.0 - d_t).view(1, -1, 1)     # 0 where a new episode begins
    lstm_state = (keep * h, keep * c)     # reset hidden on episode boundary
    out, lstm_state = self.lstm(f_t.unsqueeze(0), lstm_state)
```

This **recurrent invariant** — the hidden state is reset on `done`
(episode_start) — is what keeps memory from leaking across episode boundaries.

#### The PPO loop: `train_lstm.py`

[train_lstm.py:126](../fle/rl/train_lstm.py#L126). A CleanRL-style loop (no
CleanRL dependency). Per update:

1. **Rollout** `T = n_steps` (default 128) steps across `N` envs, storing into
   `(T, N, …)` buffers: obs (dict), the 4-factor action, the **flat 153 action
   mask**, value, logprob, reward, and per-step `done`. The action mask is stored
   so the update can reconstruct the *identical* masked distribution.
2. **GAE** advantages computed backward through the rollout
   ([train_lstm.py:238](../fle/rl/train_lstm.py#L238)).
3. **PPO update**, minibatched **by env**: each env's whole length-`T` sequence
   is replayed intact through the LSTM (so there is no sequence padding, hence no
   padding/action-mask conflation). Standard clipped surrogate loss + clipped
   value loss + entropy bonus, gradient-clipped to `max_grad_norm`
   ([train_lstm.py:255](../fle/rl/train_lstm.py#L255)).

Recurrent-specific config defaults ([lstm_config.py:18](../fle/rl/lstm_config.py#L18)):

| Knob | Value | Note |
| --- | --- | --- |
| `n_steps` | 128 | rollout / BPTT sequence length per env |
| `num_minibatches` | 2 | env-wise (sequences kept whole) |
| `update_epochs` | 4 | recurrent overfits faster than feedforward's 10 |
| `gamma` | 0.997 | long survival horizon |
| `gae_lambda` | 0.95 | |
| `clip_coef` | 0.2 | |
| `ent_coef` | 0.01 | entropy is over *valid* actions only (masking) |
| `lstm_hidden` | 256 | sized for capacity, not a VRAM budget |
| `learning_rate` | 2.5e-4 | linearly annealed to 0 |

Checkpoints are plain `torch.save` dicts (`model_state`, `model_kwargs`,
`global_step`, `config`) written to a `*_lstm` run dir so they can never clobber
a feedforward run.

---

## 3. PPO Objective & Reward Function

The current PPO code does **not** apply a regret criterion. There is no oracle,
expert-policy comparator, best-response baseline, or `regret` term in the loss or
reward. Both training paths optimize environment returns with standard PPO:

- v1 delegates to sb3-contrib `MaskablePPO`, using GAE, the clipped policy
  surrogate, value loss, entropy bonus, and gradient clipping.
- v2 computes GAE manually, then minimizes the standard clipped PPO policy loss
  plus clipped value loss minus entropy bonus in `train_lstm.py`.

If a regret metric is needed, add it as evaluation/logging first by comparing
episode return against a fixed scripted baseline or oracle; do not assume it is
currently part of training.

### 3.1 Environment reward

Computed in `_compute_reward()`
([td_environment.py:1983](../fle/env/gym_env/td_environment.py#L1983)) each step,
with terminal bonus/penalty added in `step()`
([td_environment.py:1225](../fle/env/gym_env/td_environment.py#L1225)). All weights
come from `TDScenarioConfig`
([td_config.py:122](../fle/env/gym_env/td_config.py#L122)).

### 3.2 Full per-step reward

$$
\begin{aligned}
r_t = \;& \alpha_{\text{survive}} \\
      &+ \beta_{\text{kills}}\cdot \Delta\text{kills} \\
      &- p_{\text{wall}}\cdot \text{walls\_lost} - p_{\text{turret}}\cdot \text{turrets\_lost} - p_{\text{building}}\cdot \text{buildings\_lost} \\
      &- \delta_{\text{damage}}\cdot \Delta\text{char\_HP}^{-} - p_{\text{radar}}\cdot \Delta\text{radar\_HP}^{-} \\
      &- \epsilon_{\text{invalid}}\cdot \mathbb{1}[\text{invalid}] \\
      &- p_{\text{move\_cmd}}\cdot \mathbb{1}[\text{move issued}] - p_{\text{transit}}\cdot \mathbb{1}[\text{in transit}] \\
      &+ \mathbb{1}[\text{early valid PLACE}]\cdot w_{\text{early\_place}} \\
      &+ \mathbb{1}[\text{early valid REFILL}]\cdot w_{\text{early\_refill}} \\
      &+ \text{decay}\cdot\big(w_{\text{cov}}\Delta\text{cov}^{+} + w_{\text{ammo}}\Delta\text{ammo}^{+} + w_{\text{thr}}\Delta\text{readiness}^{+} - w_{\text{threat}}\cdot\text{closeness}\big)
\end{aligned}
$$

where $\Delta(\cdot)^{+}=\max(0,\cdot)$ (improvement only) and
$\Delta(\cdot)^{-}=\max(0, \text{prev}-\text{cur})$ (damage only). The early
PLACE/REFILL bonuses only apply to valid actions during the first
`early_turret_setup_steps = 200` environment steps.

### 3.3 Term-by-term

| Term | Weight (default) | Source |
| --- | --- | --- |
| **Survival** | `alpha_survive = 0.01` per step | always added |
| **Kills** | `beta_kills = 1.0` per enemy killed | server event counter |
| **Wall destroyed** | `p_wall_destroyed = 12.0` each | strong penalty |
| **Turret destroyed** | `p_turret_destroyed = 20.0` each | very strong (turret permanently lost) |
| **Building destroyed** | `p_building_destroyed = 20.0` each | very strong |
| **Character damage** | `delta_damage = 0.5` per HP lost | from stashed `_cur_char_hp` |
| **Radar damage** | `p_radar_damage = 0.2` per HP the radar loses | the "heart" |
| **Invalid action** | `epsilon_invalid = 0.5` | when `_execute_action` returns invalid |
| **Move command** | `p_move_command = 0.25` per valid `MOVE_ANCHOR` | discourages thrashing |
| **In transit** | `p_move_transit = 0.05` per step spent walking | makes long walks costly |
| **Early valid PLACE** | `w_early_place_turret = 0.10` | first 200 steps only |
| **Early valid REFILL** | `w_early_refill_turret = 0.20` | first 200 steps only |

### 3.4 Dense shaping (annealed)

To bootstrap early learning, four shaping terms are added, then **annealed
linearly to zero** over `shaping_decay_steps = 200_000` lifetime steps
([td_environment.py:2027](../fle/env/gym_env/td_environment.py#L2027)):

```python
decay = max(0.0, 1.0 - total_steps / shaping_decay_steps)
```

| Shaping term | Weight | Fires when |
| --- | --- | --- |
| `w_coverage_delta` | 0.5 | filled-slot coverage **improves** vs last step |
| `w_ammo_delta` | 0.2 | loaded-turret fraction **improves** |
| `w_threatened_turret_delta` | 0.75 | armed-turret coverage improves **on the side threats are coming from** |
| `w_threat` | 0.05 | penalty scaling with **closeness** of the nearest swarm |

Only *positive* deltas are rewarded (so the agent isn't farmed by oscillating a
metric up and down). The "threatened turret readiness" signal
([td_environment.py:1892](../fle/env/gym_env/td_environment.py#L1892)) is the most
nuanced: it projects each turret slot onto the direction of each live biter
group (or, before contact, visible nests), weighting occupied + ammo-loaded
slots facing the threat. Placement gives partial credit
(`0.4 + 0.6 * ammo_ready`); ammo makes it a real defense. Closer / larger /
swarm-flagged groups carry more weight.

Two legacy absolute-coverage knobs (`w_coverage`, `w_ammo`) are kept at `0.0`.

### 3.5 Terminal reward

Added in `step()` after the per-step reward:

| Event | Reward | Weight |
| --- | --- | --- |
| Survived to `max_ticks` (truncated) | `+ terminal_bonus` | `+100.0` |
| Radar destroyed / character died / boilers lost (terminated) | `+ terminal_penalty` | `-200.0` |

An RCON-drop mid-step is treated as a death and also returns `terminal_penalty`,
since the game state was lost
([td_environment.py:1119](../fle/env/gym_env/td_environment.py#L1119)).

### 3.6 Difficulty presets

Three presets adjust turret scarcity and starting ammo
([td_config.py:161](../fle/env/gym_env/td_config.py#L161)):

| Preset | `turret_deletion_percentage` | Starting ammo | Runtime wave cadence knob |
| --- | --- | --- | --- |
| `EASY` | 0.3 (most turrets stay) | 1000 mags | 90 s |
| `MEDIUM` (default) | 0.5 | 500 mags | 60 s |
| `HARD` | 0.7 (many empty slots) | 200 mags | 40 s |

Reward weights are identical across presets. The wave-cadence/base-count fields
remain in the config for a future runtime-spawn mode, but current training leaves
`spawn_waves_at_runtime = False`, so the loaded map's own enemy AI controls attack
timing. Per-instance overrides are also possible without building a config:
`TowerDefenseEnv(instance, turret_deletion_percentage=0.3)`.

---

## Quick reference — running the two pipelines

```bash
# Start N Factorio containers (loads the prebuilt TD save):
fle cluster start -n 4

# v1 — feedforward MaskablePPO:
python -m fle.rl.train       --num-envs 4 --total-timesteps 1000000 --evolution-factor 0.5

# v2 — recurrent (LSTM) Maskable PPO:
python -m fle.rl.train_lstm  --num-envs 4 --total-timesteps 1000000 --evolution-factor 0.5

# Evaluate a recurrent checkpoint (one inference episode, no learning):
python -m fle.rl.eval_lstm   --model-path td_runs/<run>_lstm/final_model.pt
```
