# Issue 6 — Unity step-count issues and 1-episode overrun

**Status**: Open — fix planned
**Component**: Unity C# — `ChickAgent.cs`, `LogChannel.cs`
**Priority**: P1

This issue has two distinct sub-problems.

---

## 6a — Off-by-one: X steps requested → X+1 lines logged

### Description

For `steps_per_episode = X`, `LogChannel.cs` records X+1 rows per episode instead of X. This can misalign SB3 replay buffer sizes if the buffer is sized based on `steps_per_episode`.

### Root Cause

The off-by-one is likely caused by the ML-Agents episode lifecycle. When `EpisodeInterrupted()` fires at step X, Python's next `env.step()` call may trigger one more `OnActionReceived()` → `RecordStep()` before `OnEpisodeBegin()` resets the counter.

This needs investigation to confirm. The sequence to audit:

1. `EpisodeInterrupted()` fires at step X in `ChickAgent.cs`
2. Python calls `env.step()` one more time before receiving the `done` signal
3. Unity processes this step, calling `OnActionReceived()` and `RecordStep()` in `LogChannel.cs`
4. `OnEpisodeBegin()` resets the counter — but the extra step is already logged

### Fix

Needs investigation first. Once confirmed, add a guard in `RecordStep()` (or `OnActionReceived()`) that checks whether the episode counter has already hit the limit before logging the step.

---

## 6b — 1-episode overrun: short 2-step episode at end

### Description

After the final full episode, one spurious short episode (containing only steps 0 and 1) appears in the log. This corrupts the logged data for the last trial.

### Root Cause

`LogBuffer` was designed to buffer the first steps of each episode (where `step < 2 * DECISION_PERIOD = 2`) and only flush them once the episode has enough steps — filtering out these short terminal episodes. However, both `NewEpisode()` and `Close()` unconditionally flush `LogBuffer`, so the buffered steps 0 and 1 always get written regardless of whether the episode was long enough.

### Fix

In `NewEpisode()` and `Close()`, only flush `LogBuffer` if the buffered episode had enough steps. Add a step counter for the buffered episode and discard the buffer if the count is below the threshold.

```csharp
// Pseudocode for Close() / NewEpisode():
if (bufferedEpisodeStepCount >= MIN_STEPS_THRESHOLD) {
    FlushLogBuffer();
} else {
    ClearLogBuffer();  // discard short/terminal episode
}
```

---

## Affected Files

- `ChickAgent.cs` — step counter and episode lifecycle (6a)
- `LogChannel.cs` — `RecordStep()`, `NewEpisode()`, `Close()`, `LogBuffer` flush logic (6a and 6b)
