#!/usr/bin/env bash
# Train every TwoRoom variant sequentially, one GPU, for an overnight run.
#
#   screen -dmS tworoom scripts/run_tworoom_queue.sh
#   screen -r tworoom                 # attach
#   tail -f logs/queue/<config>.log   # or just watch a log
#
# Ordered so the most important results land first: the default complete Hierarchical
# Diffuser (K=4 high + low level), then the remaining K values, then the flat baseline
# -- which is by far the most expensive, since its model horizon is the full 80 steps.
#
# Sequential rather than concurrent: each run already keeps the GPU busy, and sharing it
# would only make every result arrive later.

set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)

PY=/home/samuel/miniconda3/envs/hd/bin/python
export PYTHONPATH=$ROOT
export LD_LIBRARY_PATH=/home/samuel/.mujoco/mujoco210/bin:/home/samuel/miniconda3/envs/hd/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia
export CPATH=/home/samuel/miniconda3/envs/hd/include
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl

DATASET=tworoom-expert-v0
LOGDIR=$ROOT/logs/queue
mkdir -p "$LOGDIR"

CONFIGS=(
  config.tworoom_ll_k4      # canary: diverged at step 3500 before clipping  ~0.7h
  config.tworoom_hl_k4      # complete HD, default K                        ~1.6h
  config.tworoom_hl_k8      #                                               ~1.0h
  config.tworoom_ll_k8      #                                               ~0.9h
  config.tworoom_hl_k2      # longest HL horizon (40), checkpointed         ~3.0h
  config.tworoom_ll_k2      #                                               ~0.4h
  config.tworoom_flat       # flat / low-level-only baseline, horizon 80    ~6.0h
)

SUMMARY=$LOGDIR/queue_summary.txt
echo "queue started $(date -Is)" | tee -a "$SUMMARY"

for cfg in "${CONFIGS[@]}"; do
  log=$LOGDIR/${cfg#config.}.log
  echo "=== $(date -Is)  START $cfg -> $log" | tee -a "$SUMMARY"
  start=$SECONDS
  # -u so progress is unbuffered and `tail -f` is truthful; this branch's Trainer
  # prints without flush=True, which otherwise makes logs look stalled.
  "$PY" -u scripts/train_tworoom.py --dataset "$DATASET" --config "$cfg" \
      > "$log" 2>&1
  rc=$?
  mins=$(( (SECONDS - start) / 60 ))
  if [ $rc -eq 0 ]; then
    last=$(grep -E "^[0-9]+:" "$log" | tail -1)
    echo "=== $(date -Is)  DONE  $cfg  (${mins}m)  ${last}" | tee -a "$SUMMARY"
  else
    # keep going: one bad config should not cost the whole night
    echo "=== $(date -Is)  FAIL  $cfg  (rc=$rc, ${mins}m) -- see $log" | tee -a "$SUMMARY"
    tail -5 "$log" | tee -a "$SUMMARY"
  fi
done

echo "queue finished $(date -Is)" | tee -a "$SUMMARY"
