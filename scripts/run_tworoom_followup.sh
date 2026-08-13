#!/usr/bin/env bash
# Re-run configs that failed in the main queue, once that queue has finished.
#
#   screen -dmS tworoom_followup scripts/run_tworoom_followup.sh config.tworoom_hl_k4
#
# Waits on the "queue finished" line in queue_summary.txt rather than on the absence of a
# training process: the main queue is briefly process-free between its own runs, and
# starting here then would put two jobs on one 12 GB card and OOM both.

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

echo "followup waiting for main queue $(date -Is)" | tee -a "$SUMMARY"
while ! grep -q "queue finished" "$SUMMARY"; do sleep 60; done

for cfg in "$@"; do
  log=$LOGDIR/${cfg#config.}.log
  echo "=== $(date -Is)  START $cfg (followup) -> $log" | tee -a "$SUMMARY"
  start=$SECONDS
  "$PY" -u scripts/train_tworoom.py --dataset "$DATASET" --config "$cfg" \
      > "$log" 2>&1
  rc=$?
  mins=$(( (SECONDS - start) / 60 ))
  if [ $rc -eq 0 ]; then
    last=$(grep -E "^[0-9]+:" "$log" | tail -1)
    echo "=== $(date -Is)  DONE  $cfg  (${mins}m)  ${last}" | tee -a "$SUMMARY"
  else
    echo "=== $(date -Is)  FAIL  $cfg  (rc=$rc, ${mins}m) -- see $log" | tee -a "$SUMMARY"
    tail -5 "$log" | tee -a "$SUMMARY"
  fi
done

echo "followup finished $(date -Is)" | tee -a "$SUMMARY"
