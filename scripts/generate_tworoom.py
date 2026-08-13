"""Generate a TwoRoom expert dataset in D4RL layout for Hierarchical Diffuser.

Runs in the `swmgen` conda env (py3.10, stable-worldmodel==0.0.6), NOT in `hd`:

    cd /home/samuel/local/projects/SHD-maze2d
    PYTHONPATH=. /home/samuel/miniconda3/envs/swmgen/bin/python \
        scripts/generate_tworoom.py --n_keep 5000 --out <path.h5>

Episodes are rolled out with the built-in TwoRoom ExpertPolicy under Gaussian action
noise, then filtered and padded so that every episode is exactly `horizon` frames long:

  * filter  -- drop episodes that fail to reach the goal, that take longer than
               `horizon` steps, or whose start-target distance is below `min_d0`
               (some resets spawn the agent on top of the target).
  * pad     -- "arrive and hold": repeat the final observation with a zero action out
               to `horizon`. TwoRoom's action is a displacement (pos += a * speed), so
               a zero action is genuinely stationary and the padding is a valid,
               executable trajectory rather than a fabricated one.

Padding matters because diffuser's SequenceDataset only emits a training index where
`start + horizon <= path_length`; unpadded episodes shorter than the horizon would
contribute no samples at all, silently.

See TWOROOM_PLAN.md sections 3 and 5 for the reasoning.
"""

import argparse
import os

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_keep", type=int, default=5000,
                   help="number of episodes to keep after filtering")
    p.add_argument("--horizon", type=int, default=64,
                   help="fixed episode length after padding (plan span H)")
    p.add_argument("--min_d0", type=float, default=0.0,
                   help="minimum start-target distance in pixels; 0 keeps everything. "
                        "Randomized start/target/door already produce a natural spread "
                        "of difficulties, including paths through the door.")
    p.add_argument("--action_noise", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_attempts_factor", type=float, default=3.0,
                   help="give up after n_keep * this many rollouts")
    p.add_argument("--pixels", action="store_true",
                   help="also record 224x224 RGB frames (large; needed for tokens)")
    p.add_argument("--out", type=str,
                   default="/home/samuel/local/datasets/tworoom/tworoom_expert_state.h5")
    return p.parse_args()


def rollout(env, policy, base, seed, horizon, min_d0, want_pixels):
    """One expert episode. Returns None if it fails the filter."""
    obs, _ = env.reset(seed=seed, options={"variation": [
        "agent.position", "target.position", "door.position"]})
    target = base.target_position.detach().cpu().numpy().copy()
    start = base.agent_position.detach().cpu().numpy().copy()
    if float(np.linalg.norm(start - target)) < min_d0:
        return None

    observations = [np.asarray(obs, dtype=np.float32)]
    pixels = [np.asarray(env.render(), dtype=np.uint8)] if want_pixels else None
    actions = []

    terminated = truncated = False
    while not (terminated or truncated):
        state = np.asarray(obs, dtype=np.float32)
        action = policy.get_action({"state": state[:2], "goal_state": target})
        obs, _, terminated, truncated, _ = env.step(action)
        actions.append(np.asarray(action, dtype=np.float32))
        observations.append(np.asarray(obs, dtype=np.float32))
        if want_pixels:
            pixels.append(np.asarray(env.render(), dtype=np.uint8))
        ## reaching the goal or exceeding the horizon both end the rollout; only the
        ## former is kept, since a goal-conditioned model needs the final frame to be
        ## the goal.
        if len(actions) >= horizon:
            break

    if not terminated or len(observations) > horizon:
        return None

    obs_arr = np.stack(observations)                       # (T+1, 10)
    act_arr = np.stack(actions) if actions else np.zeros((0, 2), np.float32)
    reached = len(obs_arr)                                 # frames of real motion

    ## arrive-and-hold padding out to the fixed horizon
    pad = horizon - reached
    if pad > 0:
        obs_arr = np.concatenate([obs_arr, np.repeat(obs_arr[-1:], pad, axis=0)])
        act_arr = np.concatenate([act_arr, np.zeros((pad, 2), np.float32)])
    act_arr = act_arr[:horizon]
    if len(act_arr) < horizon:                             # reached == horizon case
        act_arr = np.concatenate(
            [act_arr, np.zeros((horizon - len(act_arr), 2), np.float32)])

    ## sparse reward at the arrival frame, mirroring maze2d's sparse goal reward.
    rewards = np.zeros(horizon, np.float32)
    rewards[reached - 1:] = 1.0

    ## fixed-length episodes end by timeout, never by terminal: ReplayBuffer.add_path
    ## asserts that a terminal episode has no timeouts, and the padded tail means the
    ## episode genuinely continues past arrival.
    terminals = np.zeros(horizon, bool)
    timeouts = np.zeros(horizon, bool)
    timeouts[-1] = True

    out = dict(observations=obs_arr[:horizon], actions=act_arr, rewards=rewards,
               terminals=terminals, timeouts=timeouts,
               reached=reached, target=target, start=start)
    if want_pixels:
        pix = np.stack(pixels)
        if pad > 0:
            pix = np.concatenate([pix, np.repeat(pix[-1:], pad, axis=0)])
        out["pixels"] = pix[:horizon]
    return out


def main():
    args = parse_args()
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers swm/TwoRoom-v1)
    from stable_worldmodel.envs.two_room import ExpertPolicy

    env = gym.make("swm/TwoRoom-v1", max_episode_steps=args.horizon,
                   render_mode="rgb_array" if args.pixels else None)
    base = env.unwrapped
    policy = ExpertPolicy(action_noise=args.action_noise, seed=args.seed)
    policy.set_env(base)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ## episodes are streamed straight to disk: at 5k x 64 frames the pixel array is
    ## ~48 GB, far too large to accumulate in memory before writing.
    import hdf5plugin

    H = args.horizon
    flat_specs = {
        "observations": (np.float32, (10,)),
        "actions": (np.float32, (2,)),
        "rewards": (np.float32, ()),
        "terminals": (bool, ()),
        "timeouts": (bool, ()),
    }
    per_ep_specs = {"steps_to_goal": (np.int64, ()),
                    "target": (np.float32, (2,)),
                    "start": (np.float32, (2,))}

    kept = attempts = 0
    reached_all = []
    max_attempts = int(args.n_keep * args.max_attempts_factor)

    with h5py.File(args.out, "w") as f:
        for key, (dt, shape) in flat_specs.items():
            f.create_dataset(key, shape=(0, *shape), maxshape=(None, *shape), dtype=dt,
                             chunks=(H * 8, *shape) if shape else (H * 8,))
        if args.pixels:
            f.create_dataset("pixels", shape=(0, 224, 224, 3),
                             maxshape=(None, 224, 224, 3), dtype=np.uint8,
                             chunks=(H, 224, 224, 3), **hdf5plugin.Blosc())
        ## metadata/ is skipped by diffuser's sequence_dataset, so it is a safe place
        ## for per-episode bookkeeping.
        g = f.create_group("metadata")
        for key, (dt, shape) in per_ep_specs.items():
            g.create_dataset(key, shape=(0, *shape), maxshape=(None, *shape), dtype=dt)

        def append(dset, arr):
            n0 = dset.shape[0]
            dset.resize(n0 + len(arr), axis=0)
            dset[n0:] = arr

        while kept < args.n_keep and attempts < max_attempts:
            ep = rollout(env, policy, base, args.seed + attempts, H,
                         args.min_d0, args.pixels)
            attempts += 1
            if ep is None:
                continue
            for key in flat_specs:
                append(f[key], ep[key])
            if args.pixels:
                append(f["pixels"], ep["pixels"])
            append(g["steps_to_goal"], np.array([ep["reached"]], np.int64))
            append(g["target"], ep["target"][None])
            append(g["start"], ep["start"][None])
            reached_all.append(ep["reached"])
            kept += 1
            if kept % 500 == 0:
                print(f"  {kept}/{args.n_keep} kept after {attempts} rollouts",
                      flush=True)

        g.create_dataset("episode_length", data=np.full(kept, H, np.int64))
        for k, v in vars(args).items():
            f.attrs[k] = v

    env.close()

    if kept < args.n_keep:
        print(f"WARNING: only {kept} episodes kept after {attempts} rollouts")

    reached = np.array(reached_all)
    print(f"\nkept {kept} / {attempts} rollouts ({kept / attempts:.0%})")
    print(f"steps to goal: mean={reached.mean():.1f} median={np.median(reached):.0f} "
          f"min={reached.min()} max={reached.max()}")
    print(f"padding fraction: {1 - reached.mean() / H:.0%}")
    print(f"wrote {kept * H:,} transitions to {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
