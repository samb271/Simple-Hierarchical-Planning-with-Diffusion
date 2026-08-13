"""Train Hierarchical Diffuser on TwoRoom over DINOv2 patch tokens.

Identical to scripts/train.py except for two things, both forced by the scale of a token
observation rather than by any change of method:

  * TokenTrainer replaces Trainer, so the dataloader worker count is configurable
    (the stock one is fixed at 1, which serializes multi-GB memmap reads);
  * the default config points at TwoRoom.

Everything else -- dataset/model/diffusion construction, the loss, EMA, checkpointing --
is the original code path, so results stay comparable with the maze2d baselines.

    cd /home/samuel/local/projects/SHD-maze2d
    PYTHONPATH=. python scripts/train_tworoom.py \
        --dataset tworoom-expert-v0 --config config.tworoom_hl
"""
import json
import numpy as np
from os.path import join
import pdb
from diffuser.guides.policies import Policy
import diffuser.utils as utils
import diffuser.datasets as datasets
from diffuser.utils import token_training
from torch.utils.tensorboard import SummaryWriter


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class Parser(utils.Parser):
    dataset: str = "maze2d-large-v1"
    config: str = "config.tworoom_hl"


args = Parser().parse_args("diffusion")

# -----------------------------------------------------------------------------#
# ---------------------------------- dataset ----------------------------------#
# -----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    jump=args.jump,
    jump_action=args.jump_action,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, "render_config.pkl"),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim * args.jump
if args.jump_action == "none":
    action_dim = 0


# -----------------------------------------------------------------------------#
# ------------------------------ model & trainer ------------------------------#
# -----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, "model_config.pkl"),
    horizon=args.horizon // args.jump,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim=args.dim,
    dim_mults=args.dim_mults,
    kernel_size=args.kernel_size,
    device=args.device,
    upsample_k=args.upsample_k,
    downsample_k=args.downsample_k,
    ## patch-denoiser kwargs; ignored by the flat transformer
    n_tokens=args.n_tokens,
    token_dim=args.token_dim,
    grad_checkpoint=args.grad_checkpoint,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, "diffusion_config.pkl"),
    horizon=args.horizon // args.jump,
    condition=args.condition,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    ## share of the summed dimension weights given to actions. Without it they are
    ## 2 dims out of 98,306 and are never learned; see models/token_diffusion.py.
    action_loss_fraction=args.action_loss_fraction,
    device=args.device,
)

trainer_config = utils.Config(
    token_training.TokenTrainer,
    savepath=(args.savepath, "trainer_config.pkl"),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
    n_workers=args.n_workers,
    max_grad_norm=args.max_grad_norm,
)

# -----------------------------------------------------------------------------#
# -------------------------------- instantiate --------------------------------#
# -----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)


# -----------------------------------------------------------------------------#
# ------------------------ test forward & backward pass -----------------------#
# -----------------------------------------------------------------------------#

utils.report_parameters(model)

print("Testing forward...", end=" ", flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print("✓")


# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
eval_sample_n = 3

train_writer = SummaryWriter(log_dir=args.savepath + "-train")
for i in range(n_epochs):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}")

    trainer.train(n_train_steps=args.n_steps_per_epoch, writer=train_writer)
