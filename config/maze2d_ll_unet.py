"""
maze2d low-level planner, original TemporalUnet denoiser, with the checkpoint pinned
so the architecture comparison is step-matched.

imports `base` from config/maze2d_ll.py unchanged and overrides only diffusion_epoch.
this exists purely so the U-Net baseline and the transformer are evaluated at the same
number of training steps; see config/maze2d_hl_unet.py for the reasoning.

    state_360000 ("latest") holds step 377000 for ll_unet but 379000 for ll_transformer
    state_320000 holds exactly step 359000 in all four runs

the plan.logbase bug that used to live in config/maze2d_ll.py is fixed in that file
itself now, so nothing to do about it here.
"""

import copy

from config.maze2d_ll import base as _base

# ------------------------ base ------------------------#

base = copy.deepcopy(_base)

base["plan"]["diffusion_epoch"] = 320000
base["plan"]["prefix"] = "plans/release_unet"
