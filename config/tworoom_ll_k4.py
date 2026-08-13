"""Low level, K=4. Horizon K+1 = 5 environment steps, jump 1.

Trained on the trajectory segments between adjacent subgoals, sampled from anywhere in a
full-length episode -- which is why the dataset holds episodes to a fixed
EPISODE_LENGTH independent of this short horizon.
"""

from config.tworoom_base import make

base = make(prefix="diffusion/tworoom_ll_k4", horizon=5, jump=1, jump_action=True, grad_checkpoint=False)
