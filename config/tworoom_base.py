"""Shared settings for every TwoRoom run, over DINOv2 patch tokens.

Per-variant configs import `make` from here and override only what distinguishes them,
so K and the level are the only moving parts across the study. See EVALUATION_PLAN.md.

    H = 80 environment steps (covers 100% of spwm's episodes, max length 71)
    high level : horizon H,   jump K  -> model horizon H/K
    low level  : horizon K+1, jump 1  -> model horizon K+1
    flat       : horizon H,   jump 1  -> model horizon H
"""

from diffuser.utils import watch

HORIZON = 80
EPISODE_LENGTH = 80
logbase = "logs"

diffusion_args_to_watch = [
    ("prefix", ""),
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ("jump", "J"),
]


def make(prefix, horizon, jump, jump_action, dim=128, n_train_steps=2.5e4,
         grad_checkpoint=False):
    """Build a `base` dict for one run.

    jump_action:
      "none" -- states only (the maze2d high-level convention)
      True   -- dense actions, K per jumpy step; required for a level to be able to act
                on its own, which is what the high-level-only variant needs.
    """
    return {
        "diffusion": {
            ## model -- keeps the [B, H/K, N, D] token grid and attends over the 256
            ## patches of a timestep, matching LAGO's Predictor context, then over
            ## time. The flat DiffusionTransformer collapses the grid to one vector
            ## per timestep and is kept only as an ablation.
            "model": "models.patch_transformer.PatchDiffusionTransformer",
            ## GaussianDiffusion with the action dimensions reweighted. Under the stock
            ## weighting actions are 2 dims out of 98,306, get 0.002% of the gradient
            ## and are never learned -- states came out fine but the plans were
            ## unexecutable. See models/token_diffusion.py.
            "diffusion": "models.token_diffusion.TokenWeightedDiffusion",
            "horizon": horizon,
            "jump": jump,
            "jump_action": jump_action,
            "condition": True,
            "n_diffusion_steps": 64,
            "action_weight": 1,
            ## 1/3 reproduces maze2d's own action share (2 action dims out of 6), so the
            ## balance is inherited from the original method rather than tuned here.
            "action_loss_fraction": 1.0 / 3.0,
            "loss_weights": None,
            "loss_discount": 1,
            "predict_epsilon": False,
            "dim": dim,
            ## accepted and ignored by DiffusionTransformer; present so the shared
            ## training script can pass them unconditionally.
            "dim_mults": (1, 4, 8),
            "upsample_k": (3, 3, 3),
            "downsample_k": (3, 3, 3),
            "kernel_size": 5,
            "renderer": "datasets.tworoom.NullRenderer",
            ## dataset
            "loader": "datasets.tworoom.TwoRoomTokenDataset",
            "termination_penalty": None,
            "normalizer": "LimitsNormalizer",
            "preprocess_fns": [],
            ## DINOv2 x_norm_patchtokens are LayerNorm'd but not unit-scale: measured
            ## range ~[-18, 28], std ~2.5. clip_denoised clamps predicted x0 to [-1, 1]
            ## every denoising step, which would erase the signal, and no rescaling is
            ## applied to the encoder output. So clipping must be off.
            "clip_denoised": False,
            ## episodes are held out to EPISODE_LENGTH by the dataset; make_indices
            ## needs max_path_length > horizon or it silently emits no samples.
            "use_padding": True,
            "max_path_length": EPISODE_LENGTH + 1,
            ## serialization
            "logbase": logbase,
            "prefix": prefix,
            "exp_name": watch(diffusion_args_to_watch),
            ## training
            "n_steps_per_epoch": 10000,
            "loss_type": "l2",
            "n_train_steps": n_train_steps,
            ## batch 32 with no accumulation, matching spwm's worldplanner exactly.
            ## Batch size is a scientific parameter here, not a throughput knob.
            "batch_size": 32,
            "learning_rate": 2e-4,
            "gradient_accumulate_every": 1,
            "ema_decay": 0.995,
            ## checkpoint often: the queue is long, and a matched earlier step must be
            ## evaluable if a later run has not finished.
            "save_freq": 5000,
            ## 0 disables sample rendering, so NullRenderer is never invoked
            "sample_freq": 0,
            "n_saves": 10,
            "save_parallel": False,
            "n_reference": 8,
            "n_samples": 4,
            "bucket": None,
            "device": "cuda",
            "seed": 0,
            ## the stock Trainer fixes num_workers at 1, which serialises multi-GB
            ## memmap reads. Consumed by scripts/train_tworoom.py.
            "n_workers": 12,
            ## three overnight runs diverged in a single step without this;
            ## the stock Trainer has no clipping.
            "max_grad_norm": 1.0,
            ## the token grid, forwarded to the patch denoiser
            "n_tokens": 256,
            "token_dim": 384,
            ## off wherever activations fit in 12 GB; on only for the two long
            ## horizons that would otherwise OOM. Affects speed and memory only,
            ## never the gradients, so mixing it across runs is safe.
            "grad_checkpoint": grad_checkpoint,
        },
    }
