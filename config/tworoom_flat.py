"""Flat / low-level-only variant: no high-level subgoal planning.

K=1 is not a separate implementation -- the paper's own ablation labels it
"K1 (Diffuser default)" -- so the flat baseline is the same code path with jump 1 over
the full horizon, and needs no new model.
"""

from config.tworoom_base import HORIZON, make

base = make(prefix="diffusion/tworoom_flat", horizon=HORIZON, jump=1, jump_action=True,
            grad_checkpoint=True)

## This is the only config that reads all 80 frames per sample: 80 x 256 x 384 x 4 B =
## 31.5 MB, so at batch 32 the stock 12 workers x prefetch 4 hold 48 GB of tensors in
## /dev/shm, which is 31 GB. The workers are killed and the loader reports the dead
## socket as ConnectionResetError. Four workers hold 16 GB and still outrun the step,
## which at horizon 80 is compute-bound at ~2 s.
base["diffusion"]["n_workers"] = 4
