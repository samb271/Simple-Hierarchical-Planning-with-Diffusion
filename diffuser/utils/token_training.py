"""Trainer variant with a configurable dataloader, for token-scale observations.

The stock Trainer hard-codes `num_workers=1`. That is fine for maze2d, where an
observation is 4 floats, but a TwoRoom window is 16 frames x 256 tokens x 384 dims read
from a multi-GB memmap, and a single worker serializes those reads: measured 172 ms per
training step against 36 ms when the reads are issued concurrently.

Only the dataloader is replaced. The training loop, EMA schedule, checkpointing and
logging are inherited unchanged, so runs remain comparable with the maze2d baselines.
"""

import torch

from .training import Trainer, cycle


class ClippingAdam(torch.optim.Adam):
    """Adam that clips the global gradient norm before each step.

    The stock Trainer.train() has no clipping, which the maze2d U-Net tolerated for
    360k steps but the patch denoiser does not: three overnight runs diverged in a
    single step (loss 0.05 -> >1.0 at steps 3500, 4700 and 32700) and never recovered.
    Putting the clip inside the optimizer means the original training loop stays
    untouched -- only the optimizer object is swapped.
    """

    def __init__(self, params, max_grad_norm=1.0, **kwargs):
        super().__init__(params, **kwargs)
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"], self.max_grad_norm)
        return super().step(closure)


class TokenTrainer(Trainer):
    def __init__(self, *args, n_workers=8, prefetch_factor=4,
                 max_grad_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_workers = n_workers
        self.max_grad_norm = max_grad_norm
        ## swap in the clipping optimizer over exactly the parameters Trainer selected
        self.optimizer = ClippingAdam(
            [p for g in self.optimizer.param_groups for p in g["params"]],
            max_grad_norm=max_grad_norm,
            lr=self.optimizer.param_groups[0]["lr"])
        self.dataloader = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=n_workers,
                shuffle=True,
                pin_memory=True,
                ## workers are reused across epochs; respawning them every epoch would
                ## re-open the memmap and discard the page cache warmth each time.
                persistent_workers=n_workers > 0,
                prefetch_factor=prefetch_factor if n_workers > 0 else None,
            )
        )
