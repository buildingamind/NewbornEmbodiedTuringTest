#!/usr/bin/env bash
# Fan the emissive-curve replication out over the available GPUs, one run per slot.
#
# Runs the cross product STYLES x POINTS x OFFSETS in batches of $NGPU, waiting
# between batches. Ordering is OFFSET-OUTERMOST on purpose: with two styles, four
# points and offsets {8,16} on 8 GPUs, the FIRST batch is a complete
# flat-vs-restyled comparison at 8 brains/point, and the second batch doubles it to
# 16. Killing the job after one batch still leaves a usable, balanced dataset.
#
# ⚠ PREBUILD THE FRAME CACHE FIRST, SERIALLY. A cold BC7 cache plus concurrent Kit
# boots deadlocks. This script refuses to fan out until it has run
# prebuild_frame_cache once, in this process, to completion.
#
#   ./examples/launch_emissive_replication.sh \
#       --styles "flat realistic" --offsets "8 16" --episodes 2000 \
#       --locomotion wheeled --out-root ~/nett_emissive_wheeled2000
#
# ⚠ locomotion and episodes are PINNED, never inherited. The runtime locomotion
# default flipped kinematic -> wheeled on 2026-07-24, and the driver's own default
# episode budget (1000) is a REDUCTION from the canonical config's 2000 that exists
# only to match the published curve's recipe. Both must be stated explicitly or a
# wave silently measures a different experiment than the one intended.
set -euo pipefail

STYLES="flat"
POINTS="300 650 1000 2000"
OFFSETS="8 16"
EPISODES=2000
LOCOMOTION="wheeled"
OUT_ROOT="$HOME/nett_emissive_replication"

while [ $# -gt 0 ]; do
  case "$1" in
    --styles)     STYLES="$2"; shift 2;;
    --points)     POINTS="$2"; shift 2;;
    --offsets)    OFFSETS="$2"; shift 2;;
    --episodes)   EPISODES="$2"; shift 2;;
    --locomotion) LOCOMOTION="$2"; shift 2;;
    --out-root)   OUT_ROOT="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

SRC_ISAAC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_A="$(cd "$SRC_ISAAC/../.." && pwd)/NewbornEmbodiedTuringTest_Private"
ISAAC_LAB="$REPO_A/isaac_lab"
PY="${NETT_PYTHON:-/home/zlaborde/code/.venv/nett-private/bin/python}"
BINDING="$HOME/code/isaac/videos/binding"
NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"

OUT_ROOT="${OUT_ROOT/#\~/$HOME}"
mkdir -p "$OUT_ROOT"

echo "[launch] styles='$STYLES' points='$POINTS' offsets='$OFFSETS'"
echo "[launch] episodes=$EPISODES locomotion=$LOCOMOTION gpus=$NGPU out=$OUT_ROOT"

echo "[launch] serial frame-cache prebuild (cold cache + concurrent Kit = deadlock)"
PYTHONPATH="$ISAAC_LAB/source" "$PY" "$ISAAC_LAB/scripts/prebuild_frame_cache.py" \
    --design-sheet "$BINDING/DesignSheet_Binding.csv" \
    --media-root "$BINDING/videos" --resolution 256 --frame-format bc7

slot=0
batch=1
for O in $OFFSETS; do
  for S in $STYLES; do
    for E in $POINTS; do
      LOG="$OUT_ROOT/${S}_e${E}_off${O}.log"
      echo "[launch] batch $batch GPU $slot  style=$S emissive=$E offset=$O -> $LOG"
      (
        cd "$SRC_ISAAC"
        CUDA_VISIBLE_DEVICES="$slot" OMNI_KIT_ACCEPT_EULA=YES \
        PYTHONPATH=".:$ISAAC_LAB/source" \
        "$PY" examples/replicate_emissive_curve.py \
            --emissive "$E" --brain-id-offset "$O" \
            --chamber-style "$S" --locomotion "$LOCOMOTION" \
            --episodes-train "$EPISODES" \
            --out-root "$OUT_ROOT" > "$LOG" 2>&1
      ) &
      slot=$((slot + 1))
      if [ "$slot" -ge "$NGPU" ]; then
        echo "[launch] batch $batch full ($NGPU runs); waiting"
        wait
        echo "[launch] batch $batch finished"
        slot=0
        batch=$((batch + 1))
      fi
    done
  done
done

if [ "$slot" -gt 0 ]; then
  echo "[launch] final batch ($slot runs); waiting"
  wait
fi
echo "[launch] all runs finished"
