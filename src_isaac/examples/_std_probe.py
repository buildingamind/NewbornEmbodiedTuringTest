"""Print the latest learning-signal scalars from a brain's tfevents.
Usage: _std_probe.py <brain_dir-or-tfevents-glob-root> [steps_per_episode] [num_envs]
Emits one line: STEP=<trainer_step> EP=<approx_episode> STD=<x> REWARD=<x> VLOSS=<x>
plus a verdict tag when past the 700-episode decision point.
"""
import glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = sys.argv[1]
steps_per_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 500
num_envs = int(sys.argv[3]) if len(sys.argv) > 3 else 16

files = sorted(glob.glob(f"{root}/**/events.out.tfevents.*", recursive=True))
if not files:
    print("NO_TFEVENTS"); sys.exit(0)
ea = EventAccumulator(files[0], size_guidance={"scalars": 0}); ea.Reload()

def last(tag):
    try:
        s = ea.Scalars(tag)
        return (s[-1].step, s[-1].value) if s else (None, None)
    except KeyError:
        return (None, None)

step, std = last("Policy / Standard deviation")
_, rew = last("Reward / Total reward (mean)")
_, vloss = last("Loss / Value loss")
if step is None:
    print("NO_DATA_YET"); sys.exit(0)
# trainer step -> episodes: each env runs step/steps_per_ep episodes, x num_envs
ep = int(step * num_envs / steps_per_ep)
verdict = ""
if ep >= 700:
    if std is not None and std > 0.85:
        verdict = " VERDICT=KILL(std_stuck_high)"
    elif std is not None and std < 0.75:
        verdict = " VERDICT=LEARNING(std_converging)"
    else:
        verdict = " VERDICT=WATCH"
print(f"STEP={step} EP~{ep} STD={std:.3f} REWARD={rew:.1f} VLOSS={vloss:.4f}{verdict}")
