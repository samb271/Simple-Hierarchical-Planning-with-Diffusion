"""
maze2d high-level planner, DiffusionTransformer denoiser.

imports `base` from config/maze2d_hl.py and overrides ONLY the denoiser, its width,
and the log prefix, so every other setting -- horizon, jump, diffusion steps, dataset,
normalizer, loss, optimizer, seed -- is identical to the U-Net baseline by construction
rather than by copy-paste.

dim=176 is chosen to match the baseline's parameter count at this shape:
    HL umaze (H=120//15=8,  transition_dim=4): unet 3.699M vs transformer 3.675M
    HL large (H=390//15=26, transition_dim=4): unet 3.665M vs transformer 3.678M

note jump_action "none" makes action_dim 0 (scripts/train.py), so the high-level model
diffuses states only and transition_dim == observation_dim.
"""

import copy

from config.maze2d_hl import base as _base
from config.maze2d_hl import maze2d_umaze_v1 as _umaze
from config.maze2d_hl import maze2d_large_v1 as _large

# ------------------------ base ------------------------#

base = copy.deepcopy(_base)

base["diffusion"].update(
    {
        "model": "models.transformer.DiffusionTransformer",
        "dim": 176,
        "prefix": "diffusion_transformer/",
    }
)

base["plan"].update(
    {
        "dim": 176,
        "prefix": "plans/release_transformer",
        "diffusion_loadpath": "f:diffusion_transformer/H{horizon}_T{n_diffusion_steps}_J{jump}",
        ## step-matched with the U-Net baseline: state_320000 holds exactly step
        ## 359000 in all four runs, whereas "latest" (state_360000) holds 362000 for
        ## hl_unet vs 378000 here. see config/maze2d_hl_unet.py.
        "diffusion_epoch": 320000,
    }
)


# ------------------------ overrides ------------------------#

maze2d_umaze_v1 = copy.deepcopy(_umaze)
maze2d_large_v1 = copy.deepcopy(_large)
