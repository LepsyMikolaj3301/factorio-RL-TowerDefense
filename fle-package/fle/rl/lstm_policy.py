"""Masked PPO-LSTM model for Tower Defense — v2.

This is the recurrent sibling of the v1 feedforward model in :mod:`fle.rl.policy`.
It **reuses** ``TDExtractor`` verbatim (imported, never edited) and adds an LSTM
core plus four masked categorical heads over the flat MultiDiscrete action
factors ``[action_type(6), slot(64), ammo(51), anchor(32)]`` (= 153 logits).

sb3-contrib's MaskablePPO has no recurrence and RecurrentPPO has no action
masking, so this module combines the two pieces directly. The same
:meth:`get_action_and_value` function is used at rollout and update time:

  1. masks are applied **every** call (rollout *and* update);
  2. the masked distribution is reconstructed **per timestep** inside the
     sequence replay (``get_states`` loops over T; the head loop re-applies the
     mask per row);
  3. only the **action-validity** mask exists — the full-sequence-per-env scheme
     needs no padding, so there is no sequence-padding mask to conflate with it.

Recurrent invariant: the LSTM hidden state is reset on episode boundaries via the
``done`` (episode_start) gate in :meth:`get_states`.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from fle.rl.policy import TDExtractor
from fle.env.gym_env.td_spaces import (
    AMMO_AMOUNT_LEVELS,
    MAX_ANCHORS,
    MAX_SLOTS,
    NUM_ACTION_TYPES,
)

# Flat MultiDiscrete action factors, matching td_spaces.flatten_action_space and
# action_mask.ActionMaskWrapper._flat_action_mask: [type, slot, ammo, anchor].
ACTION_FACTORS: List[int] = [
    NUM_ACTION_TYPES,
    MAX_SLOTS,
    AMMO_AMOUNT_LEVELS,
    MAX_ANCHORS,
]
ACTION_MASK_DIM: int = sum(ACTION_FACTORS)  # 153

LstmState = Tuple[torch.Tensor, torch.Tensor]  # (h, c), each (num_layers, B, H)


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class RecurrentMaskableActorCritic(nn.Module):
    """TDExtractor -> LSTM -> {4 masked actor heads, critic}.

    The model operates on a flattened ``(T*B, ...)`` batch where the flat index is
    **T-major** (``index = t * B + b``); ``get_states`` reshapes back to ``(T, B)``
    to roll the LSTM through time. Rollout uses ``T == 1``; the update uses the
    full sequence length per env.
    """

    FACTORS = ACTION_FACTORS

    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        cnn_dim: int = 128,
        set_dim: int = 64,
        lstm_hidden: int = 256,
    ):
        super().__init__()
        self.features_dim = features_dim
        self.lstm_hidden = lstm_hidden

        self.extractor = TDExtractor(
            observation_space, features_dim=features_dim, cnn_dim=cnn_dim, set_dim=set_dim
        )

        self.lstm = nn.LSTM(features_dim, lstm_hidden, num_layers=1)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # One head per action factor; small std so early logits are near-uniform.
        self.actor_heads = nn.ModuleList(
            [_layer_init(nn.Linear(lstm_hidden, n), std=0.01) for n in self.FACTORS]
        )
        self.critic = _layer_init(nn.Linear(lstm_hidden, 1), std=1.0)

    # -- hidden state helpers --------------------------------------------------
    def initial_state(self, num_envs: int, device) -> LstmState:
        h = torch.zeros(self.lstm.num_layers, num_envs, self.lstm_hidden, device=device)
        c = torch.zeros(self.lstm.num_layers, num_envs, self.lstm_hidden, device=device)
        return (h, c)

    def get_states(
        self, obs: Dict[str, torch.Tensor], lstm_state: LstmState, done: torch.Tensor
    ) -> Tuple[torch.Tensor, LstmState]:
        """Roll the LSTM over a ``(T*B, ...)`` batch, resetting hidden on ``done``.

        :param obs:        dict of ``(T*B, *shape)`` tensors (T-major flatten).
        :param lstm_state: ``(h, c)`` each ``(num_layers, B, H)`` — state at t=0.
        :param done:       ``(T*B,)`` episode_start flags (1 = reset hidden here).
        :return:           ``((T*B, H) features, new (h, c))``.
        """
        feat = self.extractor(obs)                       # (T*B, features_dim)
        batch = lstm_state[0].shape[1]                   # B sequences
        feat = feat.reshape(-1, batch, self.features_dim)  # (T, B, features_dim)
        done = done.reshape(-1, batch).to(feat.dtype)    # (T, B)
        outs = []
        for f_t, d_t in zip(feat, done):                 # iterate over T
            keep = (1.0 - d_t).view(1, -1, 1)            # zero hidden where episode starts
            lstm_state = (keep * lstm_state[0], keep * lstm_state[1])
            out, lstm_state = self.lstm(f_t.unsqueeze(0), lstm_state)  # (1, B, H)
            outs.append(out)
        new_hidden = torch.cat(outs).reshape(-1, self.lstm_hidden)  # (T*B, H)
        return new_hidden, lstm_state

    def get_value(
        self, obs: Dict[str, torch.Tensor], lstm_state: LstmState, done: torch.Tensor
    ) -> torch.Tensor:
        hidden, _ = self.get_states(obs, lstm_state, done)
        return self.critic(hidden).flatten()

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        lstm_state: LstmState,
        done: torch.Tensor,
        action_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        """The one function used identically at rollout and update.

        :param action_mask: ``(T*B, 153)`` flat factor mask (1 = valid).
        :param action:      ``(T*B, 4)`` to *score* stored actions (update), or
                            ``None`` to sample/argmax fresh actions (rollout).
        :return: ``(action (T*B,4), logprob (T*B,), entropy (T*B,),
                   value (T*B,), new lstm_state)``.
        """
        hidden, lstm_state = self.get_states(obs, lstm_state, done)
        value = self.critic(hidden).flatten()

        masks = torch.split(action_mask, self.FACTORS, dim=-1)  # 4 slices
        logprob = torch.zeros(hidden.shape[0], device=hidden.device)
        entropy = torch.zeros_like(logprob)
        chosen = []
        for i, head in enumerate(self.actor_heads):
            logits = head(hidden).masked_fill(masks[i] == 0, -1e8)
            dist = torch.distributions.Categorical(logits=logits)
            if action is not None:
                a_i = action[:, i]
            elif deterministic:
                a_i = torch.argmax(logits, dim=-1)
            else:
                a_i = dist.sample()
            logprob = logprob + dist.log_prob(a_i)      # ratio computed vs masked dist
            entropy = entropy + dist.entropy()          # entropy over valid actions only
            chosen.append(a_i)
        actions = torch.stack(chosen, dim=-1)           # (T*B, 4)
        return actions, logprob, entropy, value, lstm_state


def build_agent(observation_space, model_kwargs: dict) -> RecurrentMaskableActorCritic:
    """Construct the agent from a saved ``model_kwargs`` dict (used by eval)."""
    return RecurrentMaskableActorCritic(observation_space, **model_kwargs)
