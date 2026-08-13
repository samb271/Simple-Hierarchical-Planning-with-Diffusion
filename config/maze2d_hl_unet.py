"""
maze2d high-level planner, original TemporalUnet denoiser -- with the maze2d-large
upsample kernels corrected so the model can actually train.

imports `base` from config/maze2d_hl.py unchanged. the ONLY difference is upsample_k
in the maze2d_large_v1 override.

INHERITED BUG
-------------
config/maze2d_hl.py sets, for maze2d_large_v1:

    "upsample_k": (3, 3, 4),  "downsample_k": (4, 3, 3)

with dim_mults (1, 4, 8) there are 3 resolutions but only 2 up/downsample layers, and
TemporalUnet indexes them positionally as upsample_k[ind] / downsample_k[ind] for
ind = 0, 1. so the third entry is never read. the high-level model horizon is
390 // 15 = 26, and the encoder path is

    26 --Conv1d(k=4,s=2,p=1)--> 13 --Conv1d(k=3,s=2,p=1)--> 7

which the decoder must mirror in reverse order, i.e. k=3 then k=4:

    7 --ConvT(k=3)--> 13 --ConvT(k=4)--> 26

the published tuple gives k=3 then k=3, producing 25 instead of 26. GoalDataset
conditions on index `horizon // jump - 1` = 25, so training dies immediately with
    IndexError: index 25 is out of bounds for dimension 1 with size 25
in apply_conditioning. moving the 4 into the slot the loop actually reads fixes it.

verified: (3,3,4) -> in H=26, out H=25;  (3,4,4) -> in H=26, out H=26.
the maze2d_umaze_v1 override is unaffected -- (4,4,4)/(4,4,4) round-trips H=8 -> 8 --
so it is re-exported unchanged.
"""

import copy

from config.maze2d_hl import base as _base
from config.maze2d_hl import maze2d_umaze_v1 as _umaze
from config.maze2d_hl import maze2d_large_v1 as _large

# ------------------------ base ------------------------#

base = copy.deepcopy(_base)

## pin the checkpoint so the architecture comparison is step-matched.
## checkpoint labels are step // (n_train_steps // n_saves) * ..., i.e. multiples of
## 40000, and each label file is overwritten every save_freq=1000 steps until the next
## label is reached. the runs were stopped at slightly different steps, so "latest"
## (state_360000) holds 362000 steps for hl_unet but 378000 for hl_transformer.
## state_320000 holds exactly step 359000 in all four runs -- verified by reading the
## "step" field of each checkpoint -- so it is the best-trained matched comparison.
base["plan"]["diffusion_epoch"] = 320000
## keep results separate from a run made with the published config, which writes to
## plans/release (exp_name does not encode diffusion_epoch, so they would collide)
base["plan"]["prefix"] = "plans/release_unet"


# ------------------------ overrides ------------------------#

maze2d_umaze_v1 = copy.deepcopy(_umaze)

maze2d_large_v1 = copy.deepcopy(_large)
maze2d_large_v1["diffusion"]["upsample_k"] = (3, 4, 4)
