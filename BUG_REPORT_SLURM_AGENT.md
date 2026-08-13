# Bug: logged `a_loss` / `a0_loss` collapse to 0.0000 under the action reweighting

**Status:** fixed in `diffuser/models/token_diffusion.py`, verified on all seven TwoRoom
configs. Found 13 August 2026 while smoke-testing the SLURM rerun.

**Severity:** reporting only. No checkpoint, gradient or result is affected — but it
silently blanked the one diagnostic the rerun exists to move, and
`SLURM_RERUN_CONTEXT.md` §4 currently documents the opposite behaviour.

---

## 1. Symptom

Every config trained with `TokenWeightedDiffusion` logged exactly zero action loss, at
every step, while the state loss behaved normally:

```
0:     3.7947 | a0_loss:   0.0000 | a_loss:   0.0000 | s_loss:   3.7386 | t:   1.0254
100:   1.1644 | a0_loss:   0.0000 | a_loss:   0.0000 | s_loss:   1.0821 | t:   5.6388
```

(`config.tworoom_ll_k4`, 200-step smoke run on an H100.)

Read naively this says the actions are perfectly predicted from step 0, which is not
plausible — §3 measured `a_loss` 0.096 against an action variance of ~0.33 after 15k
steps of training. The number is wrong, not the model.

## 2. Root cause

`diffuser/models/helpers.py`, `WeightedLoss.forward` (lines 114-135):

```python
loss = self._loss(pred, targ)                    # 119: RAW elementwise error
weighted_loss = (loss * self.weights).mean()     # 120: the weighting lives here
if self.action_dim != 0:
    a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
    a_loss  = (loss[:, :, :self.action_dim] / self.weights[:, :self.action_dim]).mean()
    s_loss  = (loss[:, :, self.action_dim:] / self.weights[:, self.action_dim:]).mean()
```

The division on lines 122-130 looks like it is undoing the weighting. It is not. The
weighting is applied on line 120, to build `weighted_loss`, and never touches `loss`
itself. So `loss` is *already* unweighted, and dividing it by `self.weights` scales the
reported number **down by the weight** rather than restoring it.

    reported_a_loss = true_a_loss / w

### Why it was harmless before, and is not now

The bug is latent whenever the action weight is 1, which is exactly the regime the line
was written in.

| | action weight `w` | reported `a_loss` |
|---|---|---|
| stock `GaussianDiffusion` on TwoRoom | 1 (`dim_weights = ones`, `action_weight: 1`) | correct — division by 1 |
| `TokenWeightedDiffusion`, low level | 24576 | true / 24576 |
| `TokenWeightedDiffusion`, HL K=2 / K=4 / K=8 | 12288 / 6144 / 3072 | true / w |

At `%8.4f`, a true `a_loss` of 0.1 divided by 24576 prints as `0.0000`. This is why §3's
pre-fix diagnostics were trustworthy and the post-fix logs were not: the fix that made
actions trainable is the same thing that made them invisible.

`s_loss` is unaffected. Observation weights are `discount**t` normalised to mean 1, and
every TwoRoom config sets `loss_discount: 1`, so the observation block of `self.weights`
is all ones and its division is a genuine no-op.

### Pre-existing scope beyond this project

The same line misreports `a0_loss` on maze2d, independently of any of our changes:
`config/maze2d_hl.py` and `config/maze2d_ll.py` set `action_weight: 10`, and the parent
`get_loss_weights` does `loss_weights[0, :action_dim] = action_weight`, so maze2d's
`a0_loss` has always been reported 10x too small. Upstream bug, not introduced here, and
not worth chasing — noted only so the behaviour is not mistaken for something we caused.

## 3. What the documentation says

`SLURM_RERUN_CONTEXT.md` §4 states:

> Note when comparing logs: `a0_loss` / `a_loss` / `s_loss` in `helpers.py` are divided by
> their own weights before reporting, so they stay **unweighted** and remain directly
> comparable with the pre-fix runs.

The first clause is literally true and the conclusion drawn from it is false — dividing
by the weights is precisely what breaks comparability, because the values were never
multiplied by them. **If this claim reached the dissertation text it needs correcting.**

## 4. Fix

In `diffuser/models/token_diffusion.py`, `TokenWeightedDiffusion` now overrides
`p_losses` and multiplies the two action entries of the returned `info` dict back onto
their true scale. The factor is computed in `get_loss_weights`, where the weights are
built.

Three deliberate choices about where the fix lives:

- **Not in `helpers.py`.** §2 of the rerun context puts the original modules off limits so
  the maze2d baselines stay valid, and `WeightedLoss` is shared with them.
- **In `p_losses`, not by wrapping `self.loss_fn`.** Wrapping the loss module would rename
  its registered `weights` buffer (`loss_fn.weights` -> `loss_fn.inner.weights`) in the
  checkpoint's `state_dict`, which the LAGO evaluator loads. `p_losses` leaves the module
  tree untouched.
- **Guarded on uniformity.** A single scalar can only invert `mean(raw / w)` when `w` is
  constant across timesteps. It is, whenever `action_weight` is 1 — as all seven TwoRoom
  configs set it. If someone sets `action_weight != 1`, row 0 of the weights differs from
  the rest, no scalar recovers the mean, and the code leaves the report alone and prints a
  warning rather than emitting a plausible wrong number.

## 5. Verification

The arithmetic, in isolation:

```
reported       : 0.00002183
rescaled       : 0.536504
true unweighted: 0.536504     # rescale exact
```

And live, `config.tworoom_hl_k4` under the fix — non-zero, moving, and on the same scale
as §3's pre-fix `a_loss` of 0.096:

```
0:     5.6500 | a0_loss:   0.4727 | a_loss:   0.1344 | s_loss:   5.5832 | t:   1.7516
100:   1.5219 | a0_loss:   0.3028 | a_loss:   0.1173 | s_loss:   1.4633 | t:  19.9329
```

## 6. Impact on results

None on any trained artefact. `weighted_loss` — the value returned from
`WeightedLoss.forward`, backpropagated by the trainer — is built on line 120 and was
always correct; the misreported values are read from a separate branch and only ever land
in the log line and TensorBoard. The action reweighting described in §4 of the rerun
context was working as designed the whole time.

What was lost is observability. Any run started before this fix cannot be checked for
whether its actions are learning by reading its log, because the recorded values are
`0.0000` and the true value is unrecoverable at four decimals. The 21-job sweep submitted
on 13 August carries the fix and logs usable numbers.
