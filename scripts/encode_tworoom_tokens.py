"""Encode TwoRoom frames to DINOv2 patch tokens, once, into an fp16 memmap.

Must run in the `swmgen` env (py3.10), NOT in `hd`: DINOv2's hub code uses PEP-604
`X | None` annotations, which py3.9 cannot parse, and `hd` is pinned to py3.9 by
diffuser's use of collections.Mapping.

    cd /home/samuel/local/projects/SHD-maze2d
    PYTHONPATH=. /home/samuel/miniconda3/envs/swmgen/bin/python \
        scripts/encode_tworoom_tokens.py \
        --h5 /home/samuel/local/datasets/tworoom/tworoom_expert_pixels.h5

Training then memmaps the result with no encoder in the loop. Encoding the whole dataset
costs ~100 s at measured throughput; re-encoding on the fly would repeat each frame
roughly 160x over a run, and could not run in the training env at all.
"""

import argparse
import os
import time

import h5py
import hdf5plugin  # noqa: F401  -- registers the Blosc filter used for `pixels`
import numpy as np
import torch

## deliberately self-contained: importing diffuser would pull in gym/d4rl, which are not
## installed in `swmgen` (and need not be -- encoding touches no diffuser code).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def encode_tokens(h5_path, cache_path, model_name="dinov2_vits14",
                  batch_size=256, device="cuda"):
    """Encode every frame in `h5_path` to patch tokens, saving an fp16 memmap.

    Uses deterministic ImageNet normalization and no augmentation, so the cache is
    exactly what an on-the-fly encoder would have produced.
    """
    model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    with h5py.File(h5_path, "r") as f:
        n_frames = f["pixels"].shape[0]
        with torch.no_grad():
            probe = model.forward_features(
                torch.zeros(1, 3, 224, 224, device=device))["x_norm_patchtokens"]
        n_tokens, token_dim = int(probe.shape[1]), int(probe.shape[2])
        print(f"{n_frames:,} frames -> {n_tokens} tokens x {token_dim} dims")

        tmp = cache_path + ".partial"
        out = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.float16, shape=(n_frames, n_tokens, token_dim))
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

        for i in range(0, n_frames, batch_size):
            pix = np.asarray(f["pixels"][i:i + batch_size])
            x = torch.from_numpy(pix).to(device).permute(0, 3, 1, 2).float().div_(255)
            x = (x - mean) / std
            with torch.no_grad():
                tok = model.forward_features(x)["x_norm_patchtokens"]
            out[i:i + len(pix)] = tok.half().cpu().numpy()
            if (i // batch_size) % 100 == 0:
                print(f"  encoded {i + len(pix):,}/{n_frames:,}", flush=True)

    out.flush()
    del out
    os.replace(tmp, cache_path)
    return n_tokens, token_dim


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", required=True)
    p.add_argument("--out", default=None,
                   help="defaults to <h5 stem>.dinov2_vits14.f16.npy")
    p.add_argument("--model", default="dinov2_vits14")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = args.out or (os.path.splitext(args.h5)[0] + f".{args.model}.f16.npy")
    if os.path.exists(out):
        print(f"{out} already exists; delete it to re-encode.")
        return

    t0 = time.time()
    n_tokens, token_dim = encode_tokens(args.h5, out, model_name=args.model,
                                        batch_size=args.batch_size, device=args.device)
    dt = time.time() - t0
    size = os.path.getsize(out)
    print(f"\nwrote {out}")
    print(f"  {n_tokens} tokens x {token_dim} dims, fp16, {size / 1e9:.1f} GB, "
          f"{dt:.0f}s")


if __name__ == "__main__":
    main()
