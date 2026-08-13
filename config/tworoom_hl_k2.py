"""High level (jumpy subgoals), K=2. Model horizon 40 = H/K.

Dense actions (`jump_action: True`) rather than the maze2d states-only convention, so
this single model serves two roles: it supplies subgoals to the low level in the
complete Hierarchical Diffuser, and it acts standalone as the high-level-only variant
by emitting the first action a0. Sharing the weights makes that ablation exact -- the
two differ only in whether low-level refinement is applied. This is the paper's
dense-action formulation (HD-DA).
"""

from config.tworoom_base import HORIZON, make

base = make(prefix="diffusion/tworoom_hl_k2", horizon=HORIZON, jump=2,
            jump_action=True,
            grad_checkpoint=True)
