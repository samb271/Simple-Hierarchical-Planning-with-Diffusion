"""
maze2d low-level planner, DiffusionTransformer denoiser.

imports `base` from config/maze2d_ll.py and overrides ONLY the denoiser, its width,
and the log prefix -- plus one inherited-bug fix, see below.

dim=176 matches the baseline's parameter count at this shape:
    LL (H=16, transition_dim=6): unet 3.700M vs transformer 3.677M

INHERITED BUG: config/maze2d_ll.py sets "logbase" twice inside its "plan" dict (lines
94 and 96). python keeps the last, so plan.logbase becomes the original author's
cluster path, "/common/users/cc1547/projects/diffuser/logs". hd_plan_maze2d.py passes
ll_args.logbase straight to load_diffusion, so low-level checkpoints would be looked up
there; and Parser.add_extras is commented out (utils/setup.py:61) so it cannot be
overridden from the command line. we restore the module-level `logbase` here rather
than editing the original config. the U-Net baseline needs the same fix at planning
time -- that is not needed for training, so it is deferred.
"""

import copy

from config.maze2d_ll import base as _base
from config.maze2d_ll import logbase as _logbase

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
        "logbase": _logbase,
        "prefix": "plans/release_transformer",
        "diffusion_loadpath": "f:diffusion_transformer/H{horizon}_T{n_diffusion_steps}_J{jump}",
        ## step-matched with the U-Net baseline; see config/maze2d_hl_unet.py
        "diffusion_epoch": 320000,
    }
)
