"""Find the batch size that keeps the GPU busy for a given TwoRoom config.

Separates the two costs that set step time -- dataloading and the denoiser step -- and
reports GPU memory, so the queue can be configured to saturate the card instead of
idling on 32-trajectory batches.

    PYTHONPATH=. python scripts/bench_tworoom_batch.py --config config.tworoom_hl_k4
"""

import argparse
import importlib
import time

import torch

import diffuser.utils as utils


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.tworoom_hl_k4")
    p.add_argument("--dataset", default="tworoom-expert-v0")
    p.add_argument("--batches", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--iters", type=int, default=12)
    args = p.parse_args()

    cfg = importlib.import_module(args.config).base["diffusion"]

    dataset = utils.Config(
        cfg["loader"], env=args.dataset, horizon=cfg["horizon"],
        normalizer=cfg["normalizer"], preprocess_fns=cfg["preprocess_fns"],
        use_padding=cfg["use_padding"], max_path_length=cfg["max_path_length"],
        jump=cfg["jump"], jump_action=cfg["jump_action"], verbose=False)()

    obs_dim = dataset.observation_dim
    act_dim = 0 if cfg["jump_action"] == "none" else dataset.action_dim * cfg["jump"]
    horizon = cfg["horizon"] // cfg["jump"]

    model = utils.Config(
        cfg["model"], horizon=horizon, transition_dim=obs_dim + act_dim,
        cond_dim=obs_dim, dim=cfg["dim"], dim_mults=cfg["dim_mults"],
        kernel_size=cfg["kernel_size"], upsample_k=cfg["upsample_k"],
        downsample_k=cfg["downsample_k"], device="cuda", verbose=False)()
    diffusion = utils.Config(
        cfg["diffusion"], horizon=horizon, condition=cfg["condition"],
        observation_dim=obs_dim, action_dim=act_dim,
        n_timesteps=cfg["n_diffusion_steps"], loss_type=cfg["loss_type"],
        clip_denoised=cfg["clip_denoised"], predict_epsilon=cfg["predict_epsilon"],
        action_weight=cfg["action_weight"], loss_weights=cfg["loss_weights"],
        loss_discount=cfg["loss_discount"], device="cuda", verbose=False)(model)
    opt = torch.optim.Adam(diffusion.parameters(), lr=cfg["learning_rate"])

    print(f"\n{args.config}: model horizon {horizon}, transition_dim "
          f"{obs_dim + act_dim}, params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"\n{'batch':>6} {'compute ms':>11} {'loader ms':>10} {'total ms':>9} "
          f"{'GB':>6} {'traj/s':>8}")

    for bs in args.batches:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=bs, num_workers=cfg["n_workers"], shuffle=True,
            pin_memory=True, persistent_workers=True, prefetch_factor=4)
        it = iter(loader)
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(3):  # warmup
                batch = utils.batch_to_device(next(it))
                loss, _ = diffusion.loss(*batch)
                loss.backward(); opt.step(); opt.zero_grad()
            torch.cuda.synchronize()

            t_load = t_comp = 0.0
            for _ in range(args.iters):
                t0 = time.perf_counter()
                batch = utils.batch_to_device(next(it))
                torch.cuda.synchronize(); t1 = time.perf_counter()
                loss, _ = diffusion.loss(*batch)
                loss.backward(); opt.step(); opt.zero_grad()
                torch.cuda.synchronize(); t2 = time.perf_counter()
                t_load += t1 - t0; t_comp += t2 - t1

            n = args.iters
            tot = (t_load + t_comp) / n
            print(f"{bs:>6} {t_comp / n * 1000:>11.1f} {t_load / n * 1000:>10.1f} "
                  f"{tot * 1000:>9.1f} {torch.cuda.max_memory_allocated()/1e9:>6.2f} "
                  f"{bs / tot:>8.0f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6}  OOM")
            torch.cuda.empty_cache()
            break
        finally:
            del loader, it


if __name__ == "__main__":
    main()
