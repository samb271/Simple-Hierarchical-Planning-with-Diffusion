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

All of it lives in `diffuser/models/token_diffusion.py`, inside the existing
`TokenWeightedDiffusion` subclass. Nothing else is touched. The whole fix is two edits:
compute a correction factor where the weights are built, and apply it where the loss
report leaves the model.

### 4.1 The correction

`WeightedLoss.forward` reports `mean(raw / w)` over the action block, where `raw` is the
unweighted elementwise error and `w` is the corresponding slice of the weights. What we
want reported is `mean(raw)`. If — and only if — `w` is the same value at every position
of that slice, it factors out of the mean:

    mean(raw / w) = mean(raw) / w        =>        mean(raw) = w * mean(raw / w)

So the true value is recovered by multiplying the reported one by that single `w`. If `w`
varies across the slice it does not factor out, and no scalar recovers the mean — hence
the guard below.

The action slice is `weights[:, :action_dim]`. `TokenWeightedDiffusion.get_loss_weights`
sets it to `w` everywhere and then overwrites row 0 with `w * action_weight`, so the slice
is uniform exactly when `action_weight == 1`. Every TwoRoom config sets `action_weight: 1`.

### 4.2 Edit one — compute the factor in `get_loss_weights`

At the end of the existing override, after `weights[0, :a] = w * action_weight` and before
`return weights`:

```python
col = weights[:, :a]
uniform = float(col.min()) == float(col.max())
object.__setattr__(self, "_action_report_scale",
                   float(col.max()) if uniform else None)

if not uniform:
    print("[ TokenWeightedDiffusion ] WARNING: action_weight != 1 makes the "
          "action weights non-uniform over time; logged a_loss/a0_loss stay on "
          "the weighted scale and are NOT comparable with the pre-fix runs.")
```

Two details that matter:

- **`object.__setattr__`, not plain assignment.** `get_loss_weights` is called *from*
  `GaussianDiffusion.__init__`, so this runs mid-construction. Bypassing
  `nn.Module.__setattr__` makes the write safe regardless of how much of the module
  machinery is initialised, and keeps a plain float out of the parameter/buffer registries
  where it would otherwise risk reaching the `state_dict`. It matches how the class already
  sets `action_loss_fraction`.
- **The early-return path is left alone.** The override returns the parent's weights
  unmodified when `action_dim == 0` or `action_loss_fraction` is outside `(0, 1)`. On that
  path `_action_report_scale` is never set, which is correct: no reweighting was applied,
  so no correction is due.

### 4.3 Edit two — apply the factor in `p_losses`

Add to the same class:

```python
def p_losses(self, *args, **kwargs):
    out = super().p_losses(*args, **kwargs)
    loss, info = out[0], out[1]
    scale = getattr(self, "_action_report_scale", None)
    if scale is not None:
        info = dict(info)
        for key in ("a_loss", "a0_loss"):
            if key in info:
                info[key] = info[key] * scale
    return (loss, info, *out[2:])
```

Details that matter:

- **`getattr(..., None)`**, not `self._action_report_scale`. The attribute is absent on the
  early-return path of §4.2, and a `scale` of `None` means "leave the report alone" — both
  the not-set and the not-uniform cases collapse to the same safe behaviour.
- **Preserve the return arity.** `GaussianDiffusion.p_losses` returns `(loss, info)`
  normally but `(loss, info, x_recon)` when called with `return_rec=True`. Splatting
  `*out[2:]` passes the third element through untouched instead of dropping it.
- **Copy the dict** before mutating it, so the caller's object is not modified in place.
- **`s_loss` is deliberately not corrected.** Its weights are `discount**t` normalised to
  mean 1, and every TwoRoom config sets `loss_discount: 1`, so that slice is all ones and
  its division is already a no-op. Scaling it would introduce the very error being fixed.
- **`loss` is passed through untouched.** It is the value the trainer backpropagates and it
  was never wrong. Touching it would turn a reporting bug into a training bug.

### 4.4 Why the fix lives here rather than anywhere else

- **Not in `helpers.py`.** §2 of the rerun context puts the original modules off limits so
  the maze2d baselines stay valid, and `WeightedLoss` is shared with them. Correcting the
  division at source would also change every maze2d log.
- **In `p_losses`, not by wrapping `self.loss_fn`.** Wrapping the loss module is the
  tempting one-liner, but `WeightedLoss` holds `weights` as a registered buffer, so
  wrapping renames it `loss_fn.weights` -> `loss_fn.inner.weights` in the checkpoint's
  `state_dict` — which the LAGO evaluator loads. `p_losses` leaves the module tree, and
  therefore every saved checkpoint, byte-identical in structure.
- **Guarded rather than assumed.** The uniformity test costs nothing and turns a silent
  wrong number under `action_weight != 1` into a loud warning plus an uncorrected value.

## 5. Verification

The identity from §4.1, checked in isolation — this runs standalone and needs nothing
from the repository:

```python
import torch

torch.manual_seed(0)
raw = torch.rand(8, 5, 2)              # batch 8, horizon 5, 2 action dims
w = torch.full((5, 2), 24576.0)        # the low-level action weight

reported = (raw / w).mean()            # what WeightedLoss.forward prints
assert torch.allclose(reported * w.max(), raw.mean(), rtol=1e-5)
print(f"reported {reported:.8f} -> rescaled {reported * w.max():.6f} "
      f"vs true {raw.mean():.6f}")
```

```
reported 0.00002012 -> rescaled 0.494372 vs true 0.494372
```

Note the reported value, 2.0e-5, is what prints as `0.0000` in the log.

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
