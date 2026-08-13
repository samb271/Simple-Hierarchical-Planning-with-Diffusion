"""TwoRoom dataset over DINOv2 patch tokens.

A drop-in `loader` for the training script that keeps the external tensor contract
`[batch, horizon, transition_dim]` with `transition_dim = action_dim + n_tokens*token_dim`.
The token axis is an internal detail of the denoiser, so GaussianDiffusion,
apply_conditioning, the loss weighting, Trainer and scripts/train.py are all untouched.

Only the data source changes: episodes come from the HDF5 that spwm trained its own
TwoRoom models on. Using that exact file -- not a regenerated equivalent -- is what
makes the comparison against LAGO airtight.

Two things differ from the maze2d loaders, both forced by scale:

  * Tokens are read from an fp16 memmap built once by scripts/encode_tworoom_tokens.py.
    They cannot live in the ReplayBuffer, which would hold them twice in float32.
  * Episodes are variable length (mean 23, max 71) and terminate at the goal, while the
    model needs a fixed horizon. Rather than padding on disk, the frame index is clamped
    to the episode's last frame ("arrive and hold"). That is dynamically valid here
    because a TwoRoom action is a displacement, `pos += a * speed`, so a zero action is
    genuinely stationary -- and it avoids encoding ~70% duplicate frames, cutting the
    token cache from ~157 GB to ~45 GB.

Tokens are consumed exactly as DINOv2 emits them: x_norm_patchtokens is already
LayerNorm'd, so no rescaling is applied. Their scale (std ~2.5, range ~[-18, 28]) is why
the config must set clip_denoised False.
"""

import os

import h5py
import numpy as np

from .sequence import Batch, GoalDataset

## The dataset and its 45 GB token cache are far too large to version, so where they live
## is a property of the machine. They are looked for beside the checkout first, which is
## the layout a fresh clone gets, then at the path used on the development box; either
## can be overridden outright, which is what a cluster wants when the data sits on
## $SCRATCH rather than next to the code.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_FILE = "tworoom_expert_spwm.h5"
SEARCH_DIRS = [
    os.environ.get("TWOROOM_DATA_DIR"),
    REPO_ROOT,
    os.path.join(REPO_ROOT, "datasets", "tworoom"),
    "/home/samuel/local/datasets/tworoom",
]

## resolved when `env` is not already a path
KNOWN_DATASETS = {
    ## the file spwm used to train its own TwoRoom models
    "tworoom-expert-v0": DATASET_FILE,
}


def resolve_h5(env):
    override = os.environ.get("TWOROOM_H5")
    if override:
        if not os.path.exists(override):
            raise FileNotFoundError(f"TWOROOM_H5={override!r} does not exist.")
        return override

    name = KNOWN_DATASETS.get(env, env)
    ## an explicit path, relative or absolute, is honoured as given
    if os.path.sep in name or os.path.exists(name):
        if not os.path.exists(name):
            raise FileNotFoundError(
                f"TwoRoom dataset {env!r} resolved to {name!r}, which does not exist.")
        return name

    tried = []
    for d in SEARCH_DIRS:
        if not d:
            continue
        candidate = os.path.join(d, name)
        tried.append(candidate)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"TwoRoom dataset {env!r} ({name}) not found. Looked in:\n  "
        + "\n  ".join(tried)
        + "\nPut it beside the checkout, or set TWOROOM_H5 / TWOROOM_DATA_DIR."
    )


class NullRenderer:
    """Stand-in for Maze2dRenderer.

    The training script always constructs a renderer, but Trainer only touches it when
    `sample_freq` is truthy. With `sample_freq: 0` this is never called; it exists so the
    training script needs no modification and so TwoRoom does not pull in d4rl.
    """

    def __init__(self, env=None, *args, **kwargs):
        self.env = env

    def composite(self, savepath, observations, **kwargs):
        return None


class TwoRoomTokenDataset(GoalDataset):
    """GoalDataset backed by spwm's TwoRoom HDF5, diffusing DINOv2 patch tokens."""

    def __init__(
        self,
        env="tworoom-expert-v0",
        horizon=80,
        ## frames each episode is held out to. Fixed and independent of `horizon`:
        ## the low level uses a short window (K+1) but must still sample it from
        ## anywhere in a full-length episode.
        episode_length=80,
        normalizer="LimitsNormalizer",
        preprocess_fns=(),
        max_path_length=None,
        max_n_episodes=20000,
        termination_penalty=None,
        use_padding=True,
        jump=1,
        jump_action=False,
        ## token-specific; supplied by the config, not by the training script
        token_cache=None,
    ):
        from .buffer import ReplayBuffer
        from .normalization import DatasetNormalizer

        self.h5_path = resolve_h5(env)
        self.env_name = env
        self.env = None  ## no gym environment; planning constructs its own
        self.horizon = horizon
        self.jump = jump
        self.jump_action = jump_action
        self.use_padding = use_padding

        with h5py.File(self.h5_path, "r") as f:
            self.ep_len = np.asarray(f["ep_len"], dtype=np.int64)
            self.ep_offset = np.asarray(f["ep_offset"], dtype=np.int64)
            actions = np.asarray(f["action"], dtype=np.float32)
            rewards = np.asarray(f["reward"], dtype=np.float32)
            states = np.asarray(f["observation"], dtype=np.float32)
        n_ep = len(self.ep_len)
        n_frames = int(self.ep_len.sum())

        ## make_indices computes max_start = min(path_length-1, max_path_length-horizon),
        ## so max_path_length must exceed horizon or no index is ever emitted -- and it
        ## fails silently, yielding an empty dataset.
        self.episode_length = episode_length
        self.max_path_length = max_path_length or (episode_length + 1)
        if self.max_path_length <= horizon:
            raise ValueError(
                f"max_path_length ({self.max_path_length}) must exceed horizon "
                f"({horizon}); otherwise make_indices yields no training samples."
            )
        if int(self.ep_len.max()) > episode_length:
            print(f"[ TwoRoomTokenDataset ] WARNING: "
                  f"{int((self.ep_len > episode_length).sum())} episodes exceed "
                  f"episode_length {episode_length} (max {int(self.ep_len.max())}) "
                  f"and will be cut before reaching the goal.")

        ## The buffer carries actions and the 10-dim state, already held out to the
        ## horizon so the stock normalizer and indexing work unchanged. Tokens are far
        ## too large to live here and are served from the memmap instead.
        fields = ReplayBuffer(max_n_episodes, self.max_path_length, termination_penalty)
        steps = np.arange(episode_length)
        for i in range(n_ep):
            o, L = int(self.ep_offset[i]), int(self.ep_len[i])
            idx = np.minimum(steps, L - 1) + o
            ## the terminal frame has no action taken from it, so spwm stores NaN
            ## there (one action row and one reward per episode). Actions are valid
            ## only up to L-2; from L-1 onward the agent holds, i.e. zero action.
            live = (steps < L - 1)[:, None]
            fields.add_path(dict(
                observations=states[idx],
                ## zero action past the goal: a zero displacement is exactly the "hold"
                ## that the repeated observation represents.
                actions=np.where(live, np.nan_to_num(actions[idx]), 0.0
                                 ).astype(np.float32),
                rewards=np.where(live[:, 0], np.nan_to_num(rewards[idx]), 0.0),
                terminals=np.zeros(episode_length, bool),
                timeouts=np.eye(episode_length, dtype=bool)[episode_length - 1],
            ))
        fields.finalize()
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths

        ## defaults beside the h5, but TWOROOM_TOKEN_CACHE can move it on its own -- on a
        ## cluster the 45 GB cache usually belongs on node-local scratch even when the
        ## (1 GB) h5 does not.
        self.token_cache = (token_cache or os.environ.get("TWOROOM_TOKEN_CACHE")
                            or os.path.splitext(self.h5_path)[0]
                            + ".dinov2_vits14.f16.npy")
        ## Encoding cannot happen here: DINOv2's hub code uses PEP-604 `X | None`
        ## annotations, needing py3.10+, while this repo is pinned to py3.9 by
        ## collections.Mapping. The cache is built by scripts/encode_tworoom_tokens.py
        ## in the `swmgen` env and merely memmapped here.
        if not os.path.exists(self.token_cache):
            raise FileNotFoundError(
                f"Token cache {self.token_cache} not found. Build it first:\n"
                f"  PYTHONPATH=. /home/samuel/miniconda3/envs/swmgen/bin/python \\\n"
                f"      scripts/encode_tworoom_tokens.py --h5 {self.h5_path}"
            )
        self.tokens = np.load(self.token_cache, mmap_mode="r")
        if len(self.tokens) != n_frames:
            raise ValueError(
                f"token cache has {len(self.tokens)} frames but the HDF5 has "
                f"{n_frames}; delete {self.token_cache} and re-encode."
            )
        self.n_tokens, self.token_dim = self.tokens.shape[1], self.tokens.shape[2]
        self.observation_dim = self.n_tokens * self.token_dim
        self.action_dim = fields.actions.shape[-1]

        ## actions reuse the stock normalizer; tokens are left as DINOv2 emits them
        self.normalizer = DatasetNormalizer(
            fields, normalizer, path_lengths=fields["path_lengths"])
        self.normalize(keys=["actions"])

        self.indices = self.make_indices(fields.path_lengths, horizon)
        print(f"[ TwoRoomTokenDataset ] {n_ep} episodes, {n_frames:,} real frames "
              f"(mean len {self.ep_len.mean():.1f}, max {int(self.ep_len.max())}), "
              f"{len(self.indices)} windows, horizon {horizon} jump {jump}, "
              f"obs_dim={self.observation_dim} "
              f"({self.n_tokens} tokens x {self.token_dim})")

    def _frame_indices(self, path_ind, start, end):
        """Absolute token-cache rows for a window, holding at the episode's last frame."""
        local = np.minimum(np.arange(start, end, self.jump),
                           self.ep_len[path_ind] - 1)
        return local + self.ep_offset[path_ind]

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        frames = self._frame_indices(path_ind, start, end)
        ## the held tail repeats one row many times; read each distinct row once and
        ## re-expand, which for a 23-frame episode held out to 80 is a large saving.
        uniq, inverse = np.unique(frames, return_inverse=True)
        observations = np.asarray(self.tokens[uniq], dtype=np.float32)[inverse]
        observations = observations.reshape(len(frames), self.observation_dim)

        actions = self.fields.normed_actions[path_ind, start:end].reshape(
            -1, self.jump * self.action_dim)

        conditions = self.get_conditions(observations)
        if self.jump_action == "none":
            trajectories = observations
        else:
            trajectories = np.concatenate([actions, observations], axis=-1)
        return Batch(trajectories, conditions)
