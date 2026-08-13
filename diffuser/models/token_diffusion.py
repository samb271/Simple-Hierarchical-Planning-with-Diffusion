"""GaussianDiffusion with a loss weighting that survives token-scale observations.

The stock `get_loss_weights` builds `dim_weights = torch.ones(transition_dim)`, so every
dimension of the transition vector contributes equally. On maze2d that is deliberate and
harmless -- a transition is [a_x, a_y, x, y, xd, yd], so the 2 action dimensions are a
third of the loss. Over DINOv2 patch tokens the same line makes a transition 98,306
dimensions of which 2 are actions, i.e. 0.002% of the training signal, and the model
stops learning to act.

Measured on tworoom_ll_k8 at 15k steps, trained under the stock weighting:

    s_loss 0.035          states are learned; sampled high-level subgoals beat linear
                          interpolation between the conditioned endpoints by 2.4x
    a_loss 0.096          against an action variance of ~0.33, i.e. R^2 ~ 71%
    planning              conditioned on two real frames 8 steps apart, the integrated
                          actions land 23-52 px from the true endpoint, on a segment
                          whose true travel is 40 px

So the state pathway works and only state -> action is broken. Reweighting fixes it at
the source, keeping Hierarchical Diffuser's joint [action, state] formulation rather than
bolting on a separate inverse-dynamics model.

`action_loss_fraction` is the share of the summed dimension weights given to actions.
The default 1/3 reproduces the ratio maze2d gets for free (2 action dims out of 6), so
the choice is inherited from the original method rather than tuned here. Solving

    a*w / (a*w + o) = f    ->    w = f*o / ((1-f)*a)

gives w = 24576 for a low level (a=2, o=98304) and w = 3072 for a K=8 high level (a=16),
i.e. the weight adapts to how many actions a level packs per jumpy timestep.

Nothing else changes: the noise schedule, the sampler and the losses are inherited, so
runs stay comparable with the maze2d baselines in every other respect.
"""

import torch

from .diffusion import GaussianDiffusion


class TokenWeightedDiffusion(GaussianDiffusion):
    def __init__(self, *args, action_loss_fraction=1.0 / 3.0, **kwargs):
        ## set before super().__init__, which calls get_loss_weights during construction
        object.__setattr__(self, "action_loss_fraction", float(action_loss_fraction))
        super().__init__(*args, **kwargs)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        """As the parent, but action dimensions carry weight at *every* timestep."""
        weights = super().get_loss_weights(action_weight, discount, weights_dict)

        a, o = self.action_dim, self.observation_dim
        f = self.action_loss_fraction
        if a == 0 or not 0.0 < f < 1.0:
            return weights

        w = f * o / ((1.0 - f) * a)
        weights[:, :a] = w
        ## the parent pins timestep 0 to `action_weight` absolutely, which would now
        ## *demote* the executed action below every later one. Keep it relative instead.
        weights[0, :a] = w * action_weight
        print(f"[ TokenWeightedDiffusion ] action dims {a}, observation dims {o}: "
              f"action weight {w:.0f} (a0 x{action_weight}), "
              f"giving actions {100 * f:.0f}% of the summed dimension weight")
        return weights


def action_weight_for(action_dim, observation_dim, action_loss_fraction=1.0 / 3.0):
    """The per-dimension action weight this class would apply. For tests and reporting."""
    f = action_loss_fraction
    return f * observation_dim / ((1.0 - f) * action_dim)
