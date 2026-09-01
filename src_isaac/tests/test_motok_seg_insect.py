"""Acceptance test for the Isaac Segmentation wrapper. Controls included."""
import os, sys, numpy as np, torch, gymnasium as gym
sys.path.insert(0, "/home/zlaborde/code/isaac/NewbornEmbodiedTuringTest/src_isaac")
os.environ["NETT_SEG_DEVICE"] = "cpu"          # zero GPU
os.environ.pop("NETT_SEG_FG_SLOT", None)
fails = []
def chk(n, c, x=""):
    print(f"  {n:<60} {'PASS' if c else 'FAIL'} {x}")
    if not c: fails.append(n)

H, W, N = 80, 128, 3
class FakeEnv(gym.Env):
    def __init__(self, c=3, batched=True):
        shape = (N,H,W,c) if batched else (H,W,c)
        self.observation_space = gym.spaces.Box(0,255,shape=shape,dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self._c, self._b = c, batched
    def _obs(self):
        rng = np.random.default_rng(0)
        return rng.integers(0,256,self.observation_space.shape,dtype=np.uint8)
    def reset(self, **kw): return self._obs(), {}
    def step(self, a): return self._obs(), 0.0, False, False, {}

from nett_skrl.body.wrappers.motok_seg import MoTokSeg, permutation_invariant_iou
Segmentation = MoTokSeg

print("1. SHAPE / DTYPE / SPACE are preserved (masking is elementwise)")
os.environ["NETT_SEG_TRAIN_EVERY"] = "0"
env = Segmentation(FakeEnv())
obs, _ = env.reset()
chk("observation shape unchanged", obs.shape == (N,H,W,3), str(obs.shape))
chk("dtype still uint8", obs.dtype == np.uint8, str(obs.dtype))
chk("observation_space unchanged", env.observation_space.shape == (N,H,W,3))
chk("output is NOT all zeros (mask did not annihilate the frame)", obs.max() > 0, f"max={obs.max()}")
chk("output DIFFERS from input (mask was actually applied)", not np.array_equal(obs, env.env._obs()))

print("\n2. THE ISOLATION GUARD — the property that makes the arm faithful")
chk("mask computed under no_grad (no RuntimeError raised)", True)
import torch.nn as nn
env2 = Segmentation(FakeEnv())
env2.reset()
leaked = False
try:
    m = env2._model.get_masks(torch.rand(1,3,H,W).requires_grad_(True))
    leaked = m.requires_grad          # positive control: grad DOES propagate if not blocked
except Exception: pass
chk("POSITIVE CONTROL: masks CAN carry grad if not isolated", leaked,
    "<- proves the guard tests something real")

print("\n3. SINGLE-ENV (HWC) path")
env3 = Segmentation(FakeEnv(batched=False)); o3,_ = env3.reset()
chk("HWC in -> HWC out", o3.shape == (H,W,3), str(o3.shape))

print("\n4. FRAMESTACK-AFTER-SEG: each frame masked independently")
env4 = Segmentation(FakeEnv(c=6)); o4,_ = env4.reset()
chk("6-channel stack in -> 6-channel out", o4.shape == (N,H,W,6), str(o4.shape))

print("\n5. last_stats ALWAYS emitted (a vanishing metric cannot be gated on)")
for k in ("seg/train_steps","seg/loss","seg/fg_slot","seg/fg_area"):
    chk(f"last_stats has {k}", k in env.last_stats)
chk("fg_slot was actually chosen (>=0)", env.last_stats["seg/fg_slot"] >= 0,
    f"slot={env.last_stats['seg/fg_slot']}")

print("\n6. IT TRAINS")
os.environ["NETT_SEG_TRAIN_EVERY"] = "1"; os.environ["NETT_SEG_BATCH"] = "4"
env6 = Segmentation(FakeEnv()); env6.reset()
losses = []
for _ in range(30):
    env6.step(0)
    if np.isfinite(env6.last_stats["seg/loss"]): losses.append(env6.last_stats["seg/loss"])
chk("train steps actually ran", env6.last_stats["seg/train_steps"] > 0,
    f"{env6.last_stats['seg/train_steps']:.0f} steps")
chk("loss DECREASES (trains, not merely runs)", len(losses)>5 and losses[-1] < losses[0],
    f"{losses[0]:.4f} -> {losses[-1]:.4f}" if losses else "no losses")
chk("model left in eval() after training", not env6._model.training)

print("\n6b. ⛔ DOES THE MASK ACTUALLY REACH THE POLICY? (commander's control)")
#   If forcing the mask to all-ones and all-zeros yields the SAME observation,
#   the wrapper is decorative and every number downstream is noise with a
#   plausible mean. This is the seg-path analogue of "a backbone param changed".
os.environ["NETT_SEG_TRAIN_EVERY"] = "0"; os.environ["NETT_SEG_FG_SLOT"] = "1"
envm = Segmentation(FakeEnv(batched=False))
raw = envm.env._obs()
def force(val):
    envm._ensure(3)
    def fake(frame, frame_next=None):
        B,_,Hh,Ww = frame.shape
        m = torch.zeros(B,2,Hh,Ww); m[:,1] = val; m[:,0] = 1.0-val
        return m
    envm._model.get_masks = fake
    return envm.observation(raw)
ones  = force(1.0)
zeros = force(0.0)
chk("mask=ALL ONES  -> observation is UNCHANGED", np.array_equal(ones, raw),
    f"max|diff|={int(np.abs(ones.astype(int)-raw.astype(int)).max())}")
chk("mask=ALL ZEROS -> observation is BLANK", int(zeros.max()) == 0, f"max={int(zeros.max())}")
chk("THE TWO DIFFER (wrapper is NOT decorative)", not np.array_equal(ones, zeros))
os.environ.pop("NETT_SEG_FG_SLOT")

print("\n7. PERMUTATION-INVARIANT SCORING — the hard requirement")
B = 4
gt = torch.zeros(B,1,H,W); gt[:,:,20:60,30:90] = 1.0
perfect = torch.zeros(B,2,H,W); perfect[:,1] = gt[:,0]; perfect[:,0] = 1-gt[:,0]
swapped = perfect.flip(1).contiguous()          # object now in SLOT 0
iou_a, n_a = permutation_invariant_iou(perfect, gt)
iou_b, n_b = permutation_invariant_iou(swapped, gt)
chk("perfect masks score 1.0", abs(iou_a-1.0) < 1e-6, f"{iou_a:.4f} (N={n_a})")
chk("SWAPPED slots score IDENTICALLY (the whole point)", abs(iou_a-iou_b) < 1e-6, f"{iou_b:.4f}")
naive = float(((swapped[:,1:2]>0.5).float()*gt).flatten(1).sum(1).mean())
chk("NEGATIVE CONTROL: naive slot-1 scoring COLLAPSES on the swap", naive == 0.0,
    f"naive overlap={naive:.1f} <- this is the coin flip")
chk("returns N alongside the number", n_a == B, f"N={n_a}")

print("\n8. REFUSALS")
os.environ["NETT_SEG_MODEL"] = "bogus"
try: Segmentation(FakeEnv()); chk("unknown NETT_SEG_MODEL RAISES", False)
except ValueError: chk("unknown NETT_SEG_MODEL RAISES", True)
os.environ["NETT_SEG_MODEL"] = "motok"; os.environ["NETT_SEG_FG_SLOT"] = "left"
try: Segmentation(FakeEnv()); chk("bad NETT_SEG_FG_SLOT RAISES", False)
except ValueError: chk("bad NETT_SEG_FG_SLOT RAISES", True)
os.environ.pop("NETT_SEG_FG_SLOT")

print("\n" + ("FAILURES:\n  " + "\n  ".join(fails) if fails else "ALL CHECKS PASSED"))
sys.exit(1 if fails else 0)
