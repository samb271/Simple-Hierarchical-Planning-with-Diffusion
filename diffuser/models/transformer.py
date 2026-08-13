import torch
import torch.nn as nn

from .helpers import SinusoidalPosEmb


# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#


def modulate(x, shift, scale):
    """
    x : [ batch x seq x dim ], shift / scale : [ batch x dim ]
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerBlock(nn.Module):
    """
    pre-norm self-attention + feedforward, with the diffusion timestep injected
    through adaLN-zero: shift / scale / gate are regressed from the timestep
    embedding, and the gates are zero-initialized so the block starts as identity.

    this plays the same role as ResidualTemporalBlock in models/temporal.py, which
    injects the timestep as an additive FiLM bias between its two convolutions.
    """

    def __init__(self, dim, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        self.adaLN = nn.Sequential(nn.Mish(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, t):
        """
        x : [ batch x seq x dim ]
        t : [ batch x dim ]
        """
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN(t).chunk(6, dim=-1)

        h = modulate(self.norm1(x), shift_attn, scale_attn)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_attn.unsqueeze(1) * h

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    """
    adaLN-zero output head. the projection is zero-initialized so the model starts
    by predicting all-zeros, matching the DiT recipe.
    """

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.Mish(), nn.Linear(dim, 2 * dim))
        self.proj = nn.Linear(dim, out_dim)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, t):
        shift, scale = self.adaLN(t).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))


# -----------------------------------------------------------------------------#
# ----------------------------------- model -----------------------------------#
# -----------------------------------------------------------------------------#


class DiffusionTransformer(nn.Module):
    """
    transformer denoiser -- a drop-in replacement for models.TemporalUnet.

    identical external contract:
        forward(x, cond, time), x : [ batch x horizon x transition_dim ]
                                -> [ batch x horizon x transition_dim ]

    one token per timestep: each transition vector is projected to `dim` and the
    horizon becomes the sequence axis, so self-attention replaces the conv-unet's
    temporal convolutions and up/downsampling. `cond` is ignored here exactly as in
    TemporalUnet -- conditioning is applied externally by apply_conditioning.

    note the high-level maze2d config sets jump_action "none", which makes
    action_dim 0 (scripts/train.py), so transition_dim is then just observation_dim
    and the model diffuses states only. nothing here depends on that split.

    nothing in the original codebase is modified; this class is selected purely by
    setting "model": "models.transformer.DiffusionTransformer" in a config, and it
    is constructed by the unmodified scripts/train.py.
    """

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=128,
        depth=6,
        n_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        ## accepted for drop-in compatibility with scripts/train.py, unused here.
        ## these configure the conv-unet's kernels and resolution ladder, neither of
        ## which a transformer has; listed explicitly rather than swallowed by
        ## **kwargs so that a genuine typo in a config still raises.
        dim_mults=None,
        kernel_size=None,
        upsample_k=None,
        downsample_k=None,
    ):
        super().__init__()

        self.horizon = horizon
        self.transition_dim = transition_dim

        ## same timestep embedding as TemporalUnet (models/temporal.py)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.input_proj = nn.Linear(transition_dim, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, horizon, dim) * 0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.final = FinalLayer(dim, transition_dim)

        print(
            f"[ models/transformer ] dim {dim}, depth {depth}, heads {n_heads}, "
            f"horizon {horizon}, transition_dim {transition_dim}"
        )

    def forward(self, x, cond, time):
        """
        x : [ batch x horizon x transition ]
        """
        t = self.time_mlp(time)

        x = self.input_proj(x) + self.pos_emb

        for block in self.blocks:
            x = block(x, t)

        return self.final(x, t)
