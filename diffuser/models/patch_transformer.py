"""Diffusion transformer over patch tokens, with factorized space-time attention.

A drop-in replacement for models.TemporalUnet / models.transformer.DiffusionTransformer.
The external contract is unchanged -- forward(x, cond, time) with
x : [batch, horizon, transition_dim] -- so GaussianDiffusion, apply_conditioning, the
loss weighting, Trainer and the training script are all untouched. The token grid is
recovered internally and reflattened on the way out.

Why this exists. models.transformer.DiffusionTransformer projects a whole timestep --
all 256x384 values -- through a single Linear(98312, dim), so it never attends among
patch tokens and spends 93% of its parameters on two projections. That is a reasonable
flat-token baseline but it is not comparable to LAGO, whose Predictor keeps
[B, 256, 384] and attends over the 256 tokens of one timestep.

Attention is factorized rather than joint because joint attention over
horizon x n_tokens (20 x 256 = 5120) positions would need ~27 GB of attention matrices
at batch 64. Factorized costs ~1.4 GB:

    spatial   [B*H, N+1, dim]  -- attends within a timestep; this is the block whose
                                 context (256 tokens) matches LAGO's Predictor.
    temporal  [B*N', H,   dim] -- attends across timesteps; the part LAGO has no
                                 analogue for, being a one-step world model.

Actions ride as one extra token per timestep, so a timestep is N+1 positions. When
jump_action is "none" the action slice is empty and the token count is just N.

Timestep conditioning is adaLN-zero, as in DiT: every block starts as the identity, so
training begins from a well-behaved point.
"""

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .helpers import SinusoidalPosEmb


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Attention(nn.Module):
    """Multi-head self-attention, optionally with QK normalization.

    LAGO's ViTBlock uses plain nn.MultiheadAttention with no QK norm, and does not need
    it: its attention is spatial only, over 256 genuinely distinct patch tokens. Our
    temporal attention runs over consecutive frames that are near-duplicates -- at K=4
    the agent moves ~5 px between them -- and near-identical keys with unbounded logits
    are the classic setup for attention-entropy collapse. Observed empirically: runs
    diverged sooner the shorter the temporal sequence (horizon 5 -> step 3500,
    9 -> 4700, 10 -> 32700, 20 and 40 -> never).

    Normalizing q and k bounds the logits. It guards a component LAGO does not have, so
    the spatial block -- the one that mirrors LAGO -- stays equivalent to theirs.
    """

    def __init__(self, dim, n_heads, dropout=0.0, qk_norm=True):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} not divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = self.q_norm(qkv[0]), self.k_norm(qkv[1]), qkv[2]
        ## SDPA keeps the L x L matrix out of memory (flash/mem-efficient backends)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.proj(out)


class SpaceTimeBlock(nn.Module):
    """adaLN-zero block: spatial attention, then temporal, then MLP."""

    def __init__(self, dim, n_heads, mlp_ratio, dropout=0.0, qk_norm=True):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_s = Attention(dim, n_heads, dropout, qk_norm)
        self.norm_t = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_t = Attention(dim, n_heads, dropout, qk_norm)
        self.norm_m = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"),
            nn.Dropout(dropout), nn.Linear(hidden, dim),
        )
        ## 3 sublayers x (shift, scale, gate)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, x, t):
        """x : [B, H, N, dim];  t : [B, dim]"""
        B, H, N, D = x.shape
        (ss, sc, sg, ts, tc, tg, ms, mc, mg) = self.modulation(t).chunk(9, dim=-1)

        def spread(v):  ## [B, dim] -> broadcast over horizon and tokens
            return v[:, None, None, :]

        h = modulate(self.norm_s(x), spread(ss), spread(sc))
        h = self.attn_s(h.reshape(B * H, N, D)).reshape(B, H, N, D)
        x = x + spread(sg) * h

        h = modulate(self.norm_t(x), spread(ts), spread(tc))
        h = einops.rearrange(h, "b h n d -> (b n) h d")
        h = einops.rearrange(self.attn_t(h), "(b n) h d -> b h n d", b=B)
        x = x + spread(tg) * h

        h = modulate(self.norm_m(x), spread(ms), spread(mc))
        return x + spread(mg) * self.mlp(h)


class PatchDiffusionTransformer(nn.Module):
    """Denoiser that treats a timestep as N patch tokens rather than one flat vector."""

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        n_tokens=256,
        token_dim=384,
        dim=128,
        depth=6,
        n_heads=4,
        mlp_ratio=4.0,
        ## 0.1 matches LAGO's LatentPlanner default, which reaches every ViTBlock
        dropout=0.1,
        qk_norm=True,
        grad_checkpoint=True,
        amp_dtype="bfloat16",
        ## accepted for drop-in compatibility with the training script; unused here.
        dim_mults=None,
        kernel_size=None,
        upsample_k=None,
        downsample_k=None,
        attention=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.action_dim = transition_dim - n_tokens * token_dim
        if self.action_dim < 0:
            raise ValueError(
                f"transition_dim {transition_dim} is smaller than "
                f"n_tokens*token_dim = {n_tokens * token_dim}"
            )
        self.grad_checkpoint = grad_checkpoint
        ## bf16 inside the denoiser halves step time on this card. Kept local to
        ## forward, with the output cast back to fp32, so GaussianDiffusion's loss
        ## and the training loop stay in full precision and Trainer is untouched.
        self.amp_dtype = getattr(torch, amp_dtype) if amp_dtype else None

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.token_proj = nn.Linear(token_dim, dim)
        ## actions occupy one extra token per timestep
        self.n_slots = n_tokens + (1 if self.action_dim else 0)
        if self.action_dim:
            self.action_proj = nn.Linear(self.action_dim, dim)
            self.action_head = nn.Linear(dim, self.action_dim)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, self.n_slots, dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, horizon, 1, dim) * 0.02)

        self.blocks = nn.ModuleList(
            [SpaceTimeBlock(dim, n_heads, mlp_ratio, dropout, qk_norm)
             for _ in range(depth)])

        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mod_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.token_head = nn.Linear(dim, token_dim)
        ## zero-init the output path so the model starts as an identity-like map
        nn.init.zeros_(self.mod_out[1].weight)
        nn.init.zeros_(self.mod_out[1].bias)
        nn.init.zeros_(self.token_head.weight)
        nn.init.zeros_(self.token_head.bias)
        if self.action_dim:
            nn.init.zeros_(self.action_head.weight)
            nn.init.zeros_(self.action_head.bias)

    def forward(self, x, cond, time):
        """x : [batch, horizon, transition_dim] -> same shape.

        `cond` is ignored, exactly as in TemporalUnet: conditioning is applied
        externally by apply_conditioning.
        """
        if self.amp_dtype is not None and x.is_cuda:
            with torch.autocast("cuda", dtype=self.amp_dtype):
                return self._forward(x, time).float()
        return self._forward(x, time)

    def _forward(self, x, time):
        B, H, _ = x.shape
        t = self.time_mlp(time)

        actions, obs = x[..., :self.action_dim], x[..., self.action_dim:]
        tokens = obs.reshape(B, H, self.n_tokens, self.token_dim)
        h = self.token_proj(tokens)
        if self.action_dim:
            h = torch.cat([self.action_proj(actions).unsqueeze(2), h], dim=2)
        h = h + self.spatial_pos + self.temporal_pos

        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                h = checkpoint(block, h, t, use_reentrant=False)
            else:
                h = block(h, t)

        shift, scale = self.mod_out(t).chunk(2, dim=-1)
        h = modulate(self.norm_out(h), shift[:, None, None, :], scale[:, None, None, :])

        if self.action_dim:
            a_out = self.action_head(h[:, :, 0])
            tok_out = self.token_head(h[:, :, 1:])
            return torch.cat(
                [a_out, tok_out.reshape(B, H, -1)], dim=-1)
        return self.token_head(h).reshape(B, H, -1)
