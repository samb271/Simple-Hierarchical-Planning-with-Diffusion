#!/usr/bin/env bash
# Run configs once the currently-training job exits.
#
#   screen -dmS tworoom_flat scripts/run_after_current.sh config.tworoom_flat
#
# Unlike run_tworoom_followup.sh this waits on the training process itself, not on a line
# in the summary: the main queue has already written "queue finished", so a marker-based
# wait would start immediately and put two jobs on one 12 GB card.

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
SUMMARY=$LOGDIR/queue_summary.txt
SELF=$$

echo "=== $(date -Is)  waiting for the running job before $*" | tee -a "$SUMMARY"
## grep -v $SELF so the pattern does not match this script's own command line
while pgrep -f "train_tworoom.py" | grep -qv "^${SELF}$"; do sleep 60; done

for cfg in "$@"; do
  log=$LOGDIR/${cfg#config.}.log
  echo "=== $(date -Is)  START $cfg (rerun) -> $log" | tee -a "$SUMMARY"
  start=$SECONDS
  "$PY" -u scripts/train_tworoom.py --dataset "$DATASET" --config "$cfg" > "$log" 2>&1
  rc=$?
  mins=$(( (SECONDS - start) / 60 ))
  if [ $rc -eq 0 ]; then
    echo "=== $(date -Is)  DONE  $cfg  (${mins}m)  $(grep -E '^[0-9]+:' "$log" | tail -1)" \
        | tee -a "$SUMMARY"
  else
    echo "=== $(date -Is)  FAIL  $cfg  (rc=$rc, ${mins}m) -- see $log" | tee -a "$SUMMARY"
    tail -5 "$log" | tee -a "$SUMMARY"
  fi
done
echo "=== $(date -Is)  rerun finished" | tee -a "$SUMMARY"
