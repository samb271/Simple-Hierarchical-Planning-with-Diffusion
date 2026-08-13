# TwoRoom rerun on SLURM — context for the agent picking this up

You are continuing work that ran on a single RTX 3080 Ti. Everything below happened on
that box; the reruns move to a cluster so the seven configurations train in parallel
instead of in a 12-hour queue.

**Deadline: 13 August 2026, 16:00 (Eastern).** The dissertation is due then, so prefer a
complete-but-shorter run over a partial-but-longer one.

## 1. What this project is

Comparing **Hierarchical Diffuser** (HD, Chen et al., ICLR 2024 — this repository) against
**LAGO**, a test-time-search planner, on the **TwoRoom** environment. A point agent
crosses two chambers joined by one doorway; agent, target and door positions are all
randomized per episode.

Observations are **DINOv2 ViT-S/14 patch tokens** — `x_norm_patchtokens`, 256 tokens of
width 384, so 98,304 numbers per frame. These are the same features LAGO consumes, which
is what makes the comparison fair.

The assignment (`DGA1032`) requires:

- at least three values of the temporal stride **K** → K = 2, 4, 8
- the **complete** hierarchical diffuser → `hl_kK` + `ll_kK`
- a **high-level-only** variant, no low-level refinement → the same `hl_kK` weights acting
  on their own dense actions; no separate training run
- a **flat / low-level-only** variant, no subgoal planning → `tworoom_flat` (K=1, H=80)

plus success rate, subgoal-achievement rate, training cost, planning latency, memory,
over multiple seeds; a fair-comparison argument against LAGO; and an analysis separating
hierarchy from parameter count.

## 2. Design decisions already made — do not relitigate

- **No guidance function.** HD is described with a return-guided sampler, but the released
  maze implementation has no value network and the paper reports guidance scales only for
  locomotion. We plan by **inpainting**: at every denoising step the first and last states
  of the trajectory are overwritten with the current observation and the goal image. This
  is goal-agnostic, which is what lets one model serve randomized goals.
- **Expert data only.** Guidance is what would have justified mixing in random rollouts;
  without it, random trajectories teach an inpainting planner nothing useful. LAGO does
  additionally train on ~50k random transitions. Disclose this asymmetry — it favours
  LAGO, so it does not inflate our numbers.
- **The original code is not modified.** New behaviour goes in new modules that subclass
  or wrap the originals, so the maze2d baseline results stay valid. Follow this.
- `clip_denoised: False` — tokens are LayerNorm'd but not unit scale (range ~[-18, 28]);
  clipping predicted `x0` to [-1, 1] would erase the signal.
- Tokens are consumed exactly as DINOv2 emits them. **No rescaling, no normalization.**

## 3. The problem we hit

The first full sweep trained cleanly — losses fell smoothly and plateaued — but **the
plans were unexecutable**. Evaluated under LAGO's own harness, 0/5 episodes succeeded and
the agent moved *away* from the goal.

The cause is a dimensional accident in `GaussianDiffusion.get_loss_weights`:

```python
dim_weights = torch.ones(self.transition_dim)   # every dimension weighted equally
```

A transition is `[action | observation]`. On maze2d that is `[a_x, a_y, x, y, xd, yd]` —
2 action dims out of 6, so actions are a third of the loss and the line is harmless. Over
patch tokens it is 2 action dims out of **98,306**, so actions receive **0.002%** of the
gradient signal. `action_weight` cannot fix this: it only touches timestep 0.
`loss_weights` (the `weights_dict`) cannot either: it indexes `dim_weights[action_dim +
ind]`, i.e. observations only.

Diagnostics on `tworoom_ll_k8` at 15k steps, all on in-distribution data:

| measurement | result | reading |
|---|---|---|
| `s_loss` | 0.035 | states learned |
| `a_loss` | 0.096 vs action variance ~0.33 | R² ≈ 71%, weakly learned |
| HL sampled subgoals, relative token error | **0.122** vs 0.297 linear interpolation, 0.520 copy-start | high level beats interpolation **2.4×** — it predicts the curved route through the doorway |
| LL sampled states | 0.163 vs 0.156 interpolation | no better than a straight line, but over 8 steps / 40 px that is near-optimal anyway |
| LL actions, conditioned on two *real* frames 8 steps apart | integrated endpoint **23–52 px** off, true travel 40 px | unusable |

So the state pathway works and only state → action is broken.

### Things that were suspected and cleared — do not re-investigate

- **Action normalization.** Data range is exactly [-1, 1] per dimension, so
  `LimitsNormalizer` is the identity. Not a factor.
- **Observation pipeline.** The evaluator's live DINOv2 encode versus the cached tokens
  used in training: relative difference 0.0002, **cosine 1.00000**. Identical.
- **The checkpoints.** `s_loss 0.0353` through the real dataloader. The models are fine.
- **Training stability.** Solved earlier; see §5.

Two false alarms came from broken diagnostics, noted so you do not repeat them:
`np.memmap` on the token cache reads the **128-byte `.npy` header as data**, shifting
every frame — use `np.load(path, mmap_mode="r")`. And `tworoom_expert_pixels.h5` is an
older, unrelated generated dataset; the real one is `tworoom_expert_spwm.h5`.

## 4. The fix being applied in this rerun

`diffuser/models/token_diffusion.py` → **`TokenWeightedDiffusion`**, a `GaussianDiffusion`
subclass that overrides `get_loss_weights` so action dimensions carry weight at *every*
timestep, not just timestep 0.

`action_loss_fraction` is the share of summed dimension weight given to actions, default
**1/3** — chosen to reproduce the ratio maze2d gets for free (2 action dims out of 6), so
the balance is inherited from the original method rather than tuned. The per-dimension
weight follows from `w = f·o / ((1−f)·a)` and therefore adapts to each level:

| level | action dims | weight | action share, before → after |
|---|---|---|---|
| low level (any K) | 2 | 24576 | 0.00002 → 0.333 |
| high level, K=4 | 8 | 6144 | 0.00008 → 0.333 |
| high level, K=8 | 16 | 3072 | 0.00016 → 0.333 |

We chose this over adding an inverse-dynamics model (Decision Diffuser style) **to stay
faithful to HD's joint `[action, state]` formulation**. Reweighting fixes the cause; an
IDM would work around it with a component the original method does not have.

Already wired up — `config/tworoom_base.py` selects the class and sets the fraction, and
`scripts/train_tworoom.py` forwards it. Nothing further to do to enable it.

Note when comparing logs: `a0_loss` / `a_loss` / `s_loss` in `helpers.py` are divided by
their own weights before reporting, so they stay **unweighted** and remain directly
comparable with the pre-fix runs. The total `loss` will look different; that is expected.

## 5. Training settings that are load-bearing

The patch denoiser diverged under the settings inherited from maze2d — three runs went
from loss 0.05 to >1.0 in a single step, at steps 3500, 4700 and 32700, and never
recovered. Onset was monotone in the *temporal* sequence length (model horizon 5 → step
3500, 9 → 4700, 10 → 32700, 20 and 40 → never), implicating attention over consecutive
frames that are near-duplicates. Three changes fixed it and **must stay on**:

- **QK normalization** in `patch_transformer.Attention` (largest effect)
- **gradient clipping** at norm 1.0, via `ClippingAdam` in `utils/token_training.py`
- **dropout 0.1**, matching LAGO's `ViTBlock` default

After the fix, excluding the first 1000 warmup steps, **zero** logged points exceed 0.5,
against 35–94% before.

LAGO does not use QK-norm. That is not an inconsistency: its attention is spatial only,
over 256 genuinely distinct patch tokens, and never hits the failure. The guard protects
the temporal stage, which LAGO has no analogue for; the spatial stage still mirrors theirs.

## 6. What to run

Seven independent jobs, one per config — they share nothing and should run in parallel:

```
config.tworoom_hl_k2   config.tworoom_ll_k2
config.tworoom_hl_k4   config.tworoom_ll_k4
config.tworoom_hl_k8   config.tworoom_ll_k8
config.tworoom_flat
```

```bash
python scripts/train_tworoom.py --dataset tworoom-expert-v0 --config config.tworoom_hl_k4
```

Confirm on the first job that the log line

```
[ TokenWeightedDiffusion ] action dims N, observation dims 98304: action weight W ...
```

appears. If it does not, the config is still on stock `GaussianDiffusion` and the run is
wasted.

### Sizing

Wall-clock on one 3080 Ti at 20k steps, as a scaling reference:
`ll_k2` ~35 min, `ll_k4` 47 min, `ll_k8` ~71 min, `hl_k8` 78 min, `hl_k4` ~2 h,
`hl_k2` ~3.5 h, `flat` ~6 h (`flat` denoises the full 80-step horizon and dominates).

Two traps in the step budget:

- `n_epochs = int(n_train_steps // n_steps_per_epoch)` with `n_steps_per_epoch = 10000`,
  so **`n_train_steps` must be a multiple of 10,000** or the remainder is silently
  dropped. `2.5e4` trains for 20,000 steps, not 25,000.
- Checkpoints are written when `step % save_freq == 0` over steps `0..n-1`, so a 20,000
  step run's **last checkpoint is step 15,000**. Either raise the budget or save at the
  end; do not report a budget the checkpoints do not reflect.

Memory: `grad_checkpoint=True` is set for `hl_k2`, `hl_k4` and `flat`, which otherwise
exceed 12 GB. On a larger card it can be turned off for speed — it changes memory and
throughput only, never the gradients, so mixing it across runs is safe.

### Data

`tworoom_expert_spwm.h5` (~1.1 GB) and `tworoom_expert_spwm.dinov2_vits14.f16.npy`
(~45 GB, being transferred, do **not** regenerate). Both are resolved from, in order:
`TWOROOM_H5` → `TWOROOM_DATA_DIR` → **the repository root** → `<repo>/datasets/tworoom`.
Dropping them at the root of the clone needs no configuration. The cache is a memmap read
at high rate by 12 dataloader workers, so put it on the fastest filesystem available and
set `TWOROOM_TOKEN_CACHE` if that is not beside the `.h5`.

10,000 episodes, 230,604 frames, mean length 23.1, max 71. Episodes are held to
`EPISODE_LENGTH = 80` by clamping the frame index to the episode's last frame
("arrive and hold") — valid because a TwoRoom action is a displacement, so a zero action
is genuinely stationary.

## 7. After training

Evaluation lives in the **LAGO** repository, not this one:
`scripts/solve/tworoom/tworoom_hd.py`. It imports LAGO's own episode construction,
difficulty schedule, success criterion and CSV writers, and replaces only the planner, so
the two methods are scored by identical code. A `sys.modules` shim lets this py3.9
repository's inference path load under LAGO's py3.12 — it needs exactly three symbols
(`Progress`, `Silent`, `to_np`).

```bash
python -m scripts.solve.tworoom.tworoom_hd \
  --config scripts/solve/tworoom/configs/tworoom_hd.yaml \
  --hl_run <SHD>/logs/tworoom-expert-v0/diffusion/tworoom_hl_k4_H80_T64_J4 \
  --ll_run <SHD>/logs/tworoom-expert-v0/diffusion/tworoom_ll_k4_H5_T64_J1
```

Planning latency and peak VRAM are already instrumented and land in `hd_metrics.csv`.
Success rate comes from LAGO's summaries. Five difficulties (35, 75, 115, 155, 200 px),
150 step budget, seed 10^6, success radius 16 px, agent speed 5 px/step.

Still open:

- **Subgoal-achievement rate** is required by the assignment and not implemented. It needs
  subgoals — 98,304-vectors — related back to positions. Nearest-neighbour decoding
  against the training frames does **not** work: token distance is dominated by the scene
  layout, which is randomized per episode, giving 64 px median error and only 3.5% of
  held-out frames within the 16 px radius. Within a *fixed* scene the correlation between
  token distance and pixel distance is 0.90, so a per-episode calibration against the
  executed rollout is the promising direction. It was descoped under time pressure.
- **Multiple seeds.** Every config trains at `seed: 0`. Parallel jobs make training seeds
  affordable now; otherwise vary evaluation seeds via `repeat_per_difficulty` and disclose
  the single training seed.
- **Capacity matching.** The patch denoiser is 2.78M parameters against LAGO's 14.78M —
  the assignment asks for matched capacity. LAGO's `transformer_dim` is pinned at
  `image_emb_dim + condition_dim` = 448, so `depth` is the only free knob: depth 1 gives
  2.707M (matches one HD model, 2.5% off) and depth 2 gives 5.122M (matches the deployed
  two-model hierarchy, 8% off). Retraining LAGO small is one job; scaling HD up is seven.
- **`leakage_data_path`** points at a missing `envs/tworoom/data`, so the train/solve
  leakage check currently skips with a printed warning.
