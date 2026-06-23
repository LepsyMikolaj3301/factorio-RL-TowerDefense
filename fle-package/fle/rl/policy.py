"""Tower Defense RL model — v1.

The v1 model is a custom multi-input feature extractor (`TDExtractor`) plus a
pointer scoring head (`PointerHead`), trained with MaskablePPO (sb3-contrib).

Architecture (see the design plan, Part D):

    map (8x64x64)  -> CoordConv CNN ----\
    turret_slots   -> DeepSets ----------+
    anchors        -> DeepSets ----------+--> fuse -> shared latent -> actor + critic
    biter_groups   -> DeepSets ----------+
    nests          -> DeepSets ----------+
    scalars        -> MLP --------------/

The extractor returns a single fused latent (SB3's `features`), consumed by the
standard MaskableActorCriticPolicy actor/critic heads over the flat MultiDiscrete
action space, with reach/transit-aware per-factor masks (see ActionMaskWrapper).

`PointerHead` is the building block for permutation-aware slot/anchor selection.
`make_td_policy_kwargs()` returns the policy_kwargs to plug `TDExtractor` into
MaskablePPO. Everything here is plain ``torch`` / SB3 ``BaseFeaturesExtractor`` so
it is device-agnostic — SB3 moves it to the configured device automatically and
the CoordConv channels are created on the input tensor's device.
"""

import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

try:  # SB3 is an optional dep; only needed when actually training.
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except Exception:  # pragma: no cover - allows importing PointerHead without SB3
    BaseFeaturesExtractor = nn.Module  # type: ignore


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
        nn.ReLU(),
    )


def masked_mean_max_pool(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """DeepSets pool: concat masked mean and masked max over the set dim.

    :param emb:  (B, N, D) per-element embeddings.
    :param mask: (B, N) 1 for real elements, 0 for padding.
    :return:     (B, 2D). Rows with no valid elements pool to zeros.
    """
    m = mask.unsqueeze(-1).to(emb.dtype)            # (B, N, 1)
    valid_any = (m.sum(dim=1) > 0).to(emb.dtype)    # (B, 1)
    count = m.sum(dim=1).clamp(min=1.0)             # (B, 1)
    mean = (emb * m).sum(dim=1) / count             # (B, D)
    neg = torch.finfo(emb.dtype).min
    masked = emb.masked_fill(m == 0, neg)
    mx = masked.max(dim=1).values                   # (B, D)
    # Zero out pooled values for all-padding sets (also kills the finfo.min).
    return torch.cat([mean * valid_any, mx * valid_any], dim=-1)


class PointerHead(nn.Module):
    """Score a set of element embeddings against a query vector (pointer net).

    Produces one logit per set element, so the slot/anchor action factor is
    permutation-aware and generalizes across maps with different element counts.
    """

    def __init__(self, emb_dim: int, query_dim: int, hidden: int = 64):
        super().__init__()
        self.q = nn.Linear(query_dim, hidden)
        self.k = nn.Linear(emb_dim, hidden)
        self._scale = 1.0 / math.sqrt(hidden)

    def forward(
        self,
        emb: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """:param emb: (B, N, E); :param query: (B, Q) -> logits (B, N)."""
        q = self.q(query).unsqueeze(1)              # (B, 1, H)
        k = self.k(emb)                             # (B, N, H)
        logits = (k * q).sum(dim=-1) * self._scale  # (B, N)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        return logits


class TDExtractor(BaseFeaturesExtractor):
    """Multi-input feature extractor for the Tower Defense observation.

    Returns a single fused latent of size ``features_dim``. Also exposes the
    per-slot / per-anchor embeddings from the most recent forward pass on
    ``self.last_slot_emb`` / ``self.last_anchor_emb`` so a pointer-head policy can
    reuse them without recomputation.
    """

    # Scalar obs keys concatenated into the scalar MLP.
    SCALAR_KEYS = (
        "inventory",
        "character",
        "radar",
        "game",
        "movement",
        "recent_losses",
    )

    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        cnn_dim: int = 128,
        set_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim)
        spaces = observation_space.spaces

        # --- Map CNN (+2 CoordConv channels) ---
        c, h, w = spaces["map"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c + 2, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, c + 2, h, w)).shape[1]
        self.cnn_head = nn.Sequential(nn.Linear(cnn_out, cnn_dim), nn.ReLU())

        # --- Set encoders (DeepSets) ---
        self.slot_mlp = _mlp(spaces["turret_slots"].shape[1], set_dim)
        self.anchor_mlp = _mlp(spaces["anchors"].shape[1], set_dim)
        self.group_mlp = _mlp(spaces["biter_groups"].shape[1], set_dim)
        self.nest_mlp = _mlp(spaces["nests"].shape[1], set_dim)
        self.boiler_mlp = _mlp(spaces["boilers"].shape[1], set_dim)

        # --- Scalar MLP ---
        scalar_dim = sum(int(np.prod(spaces[k].shape)) for k in self.SCALAR_KEYS)
        self.scalar_mlp = _mlp(scalar_dim, 2 * set_dim)

        fused_in = cnn_dim + 5 * (2 * set_dim) + (2 * set_dim)
        self.fuse = nn.Sequential(nn.Linear(fused_in, features_dim), nn.ReLU())

        # Cached per-element embeddings from the last forward (for pointer heads).
        self.last_slot_emb: torch.Tensor = None
        self.last_anchor_emb: torch.Tensor = None

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        map_ = obs["map"].float() / 255.0
        b, _, h, w = map_.shape
        ys = torch.linspace(-1.0, 1.0, h, device=map_.device).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1.0, 1.0, w, device=map_.device).view(1, 1, 1, w).expand(b, 1, h, w)
        cnn = self.cnn_head(self.cnn(torch.cat([map_, xs, ys], dim=1)))

        slot_emb = self.slot_mlp(obs["turret_slots"].float())
        anchor_emb = self.anchor_mlp(obs["anchors"].float())
        self.last_slot_emb = slot_emb
        self.last_anchor_emb = anchor_emb

        slot_pool = masked_mean_max_pool(slot_emb, obs["slot_valid_mask"].float())
        anchor_pool = masked_mean_max_pool(anchor_emb, obs["anchor_valid_mask"].float())
        group_pool = masked_mean_max_pool(
            self.group_mlp(obs["biter_groups"].float()), obs["group_valid_mask"].float()
        )
        nest_pool = masked_mean_max_pool(
            self.nest_mlp(obs["nests"].float()), obs["nest_valid_mask"].float()
        )
        boiler_pool = masked_mean_max_pool(
            self.boiler_mlp(obs["boilers"].float()), obs["boiler_valid_mask"].float()
        )

        scalars = torch.cat([obs[k].float().flatten(1) for k in self.SCALAR_KEYS], dim=-1)
        scal = self.scalar_mlp(scalars)

        fused = torch.cat(
            [cnn, slot_pool, anchor_pool, group_pool, nest_pool, boiler_pool, scal],
            dim=-1,
        )
        return self.fuse(fused)


def make_td_policy_kwargs(
    features_dim: int = 256,
    cnn_dim: int = 128,
    set_dim: int = 64,
    net_arch=None,
) -> dict:
    """policy_kwargs to plug TDExtractor into (Maskable)PPO's MultiInputPolicy."""
    if net_arch is None:
        net_arch = dict(pi=[256, 128], vf=[256, 128])
    return dict(
        features_extractor_class=TDExtractor,
        features_extractor_kwargs=dict(
            features_dim=features_dim, cnn_dim=cnn_dim, set_dim=set_dim
        ),
        net_arch=net_arch,
    )
