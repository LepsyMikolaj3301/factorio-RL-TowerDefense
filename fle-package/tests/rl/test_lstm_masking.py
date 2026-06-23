"""Unit tests for the masked PPO-LSTM model (fle.rl.lstm_policy).

These target the classic masked-recurrent fusion bugs without needing a Factorio
container — they exercise the policy module directly:

  * masked actions are never selected, and score to ~zero probability;
  * the importance ratio is exactly 1 under a frozen policy (the killer test:
    catches any rollout<->update mask/feature/hidden misalignment);
  * the LSTM hidden state resets on episode boundaries (done flag);
  * entropy is computed over valid actions only (a 1-valid factor contributes 0).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")  # TDExtractor's BaseFeaturesExtractor

from fle.env.gym_env.td_spaces import make_observation_space
from fle.rl.lstm_policy import (
    ACTION_FACTORS,
    ACTION_MASK_DIM,
    RecurrentMaskableActorCritic,
)


def _make_agent(seed: int = 0) -> RecurrentMaskableActorCritic:
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_space = make_observation_space()
    # Tiny dims keep the CNN/LSTM cheap on CPU.
    return RecurrentMaskableActorCritic(
        obs_space, features_dim=32, cnn_dim=16, set_dim=8, lstm_hidden=16
    ).eval()


def _sample_obs(obs_space, n: int) -> dict:
    rows = {k: [] for k in obs_space.spaces}
    for _ in range(n):
        s = obs_space.sample()
        for k, v in s.items():
            rows[k].append(v)
    return {k: torch.as_tensor(np.stack(v)) for k, v in rows.items()}


def _random_masks(n: int, seed: int = 1) -> torch.Tensor:
    """Masks with >= 1 valid option per factor, so each factor is samplable."""
    rng = np.random.default_rng(seed)
    m = (rng.random((n, ACTION_MASK_DIM)) > 0.5).astype(np.float32)
    off = 0
    for width in ACTION_FACTORS:
        m[:, off] = 1.0  # guarantee index 0 of every factor slice is valid
        off += width
    return torch.as_tensor(m)


def _factor_slices(mask_row: torch.Tensor):
    out, off = [], 0
    for width in ACTION_FACTORS:
        out.append(mask_row[off:off + width])
        off += width
    return out


def test_factor_widths_match_mask_dim():
    assert sum(ACTION_FACTORS) == ACTION_MASK_DIM == 153


@pytest.mark.parametrize("deterministic", [True, False])
def test_masked_action_never_selected(deterministic):
    agent = _make_agent()
    obs_space = make_observation_space()
    n = 6
    obs = _sample_obs(obs_space, n)
    masks = _random_masks(n)
    state = agent.initial_state(n, torch.device("cpu"))
    done = torch.zeros(n)

    with torch.no_grad():
        actions, logprob, _, _, _ = agent.get_action_and_value(
            obs, state, done, masks, deterministic=deterministic
        )

    # Every chosen factor index must be a *valid* index per its mask slice.
    for b in range(n):
        slices = _factor_slices(masks[b])
        for f, sl in enumerate(slices):
            assert sl[int(actions[b, f])] == 1.0
    assert torch.isfinite(logprob).all()


def test_frozen_policy_ratio_is_one():
    """Score rollout actions again with identical weights/state/mask -> ratio == 1."""
    agent = _make_agent(seed=3)
    obs_space = make_observation_space()
    T, B = 4, 2
    obs = _sample_obs(obs_space, T * B)              # T-major flatten
    masks = _random_masks(T * B, seed=7)
    state = agent.initial_state(B, torch.device("cpu"))
    done = torch.zeros(T * B)

    with torch.no_grad():
        actions, logp_old, _, _, _ = agent.get_action_and_value(obs, state, done, masks)
        # Replay: same obs/state/done/mask, scoring the *stored* actions.
        _, logp_new, _, _, _ = agent.get_action_and_value(
            obs, state, done, masks, action=actions
        )

    ratio = (logp_new - logp_old).exp()
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5), ratio


def test_hidden_state_resets_on_done():
    """A done at t=1 must make the post-done rollout independent of pre-done history."""
    agent = _make_agent(seed=5)
    obs_space = make_observation_space()
    B = 1
    full = _sample_obs(obs_space, 3 * B)             # steps t=0,1,2

    state0 = agent.initial_state(B, torch.device("cpu"))
    done_full = torch.tensor([0.0, 1.0, 0.0])        # reset at t=1
    with torch.no_grad():
        _, full_state = agent.get_states(full, state0, done_full)

        # Fresh run over steps t=1,2 only, with a clean initial state.
        sub = {k: v[1 * B:3 * B] for k, v in full.items()}
        done_sub = torch.tensor([0.0, 0.0])
        _, sub_state = agent.get_states(sub, agent.initial_state(B, torch.device("cpu")), done_sub)

    assert torch.allclose(full_state[0], sub_state[0], atol=1e-5)
    assert torch.allclose(full_state[1], sub_state[1], atol=1e-5)


def test_entropy_over_valid_actions_only():
    """If every factor has exactly one valid option, total entropy is ~0."""
    agent = _make_agent(seed=9)
    obs_space = make_observation_space()
    n = 3
    obs = _sample_obs(obs_space, n)
    state = agent.initial_state(n, torch.device("cpu"))
    done = torch.zeros(n)

    one_hot = torch.zeros(n, ACTION_MASK_DIM)
    off = 0
    for width in ACTION_FACTORS:
        one_hot[:, off] = 1.0                        # single valid index per factor
        off += width

    with torch.no_grad():
        _, _, entropy, _, _ = agent.get_action_and_value(obs, state, done, one_hot)

    assert torch.allclose(entropy, torch.zeros_like(entropy), atol=1e-3), entropy
