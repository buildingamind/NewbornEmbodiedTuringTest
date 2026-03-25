# NETT Issues — Prioritized TODO

## P0 — Blocking Correctness

- [ ] **P0** Fix Issue 1: ghost loading bar in memory estimation (`src/nett/nett.py`) — move `remove()` into `finally` block
- [ ] **P0** Fix Issue 2: crash-resilient `task_waiter` (`src/nett/nett.py`) — catch worker exceptions, log, free GPU memory, continue
- [ ] **P0** Fix Issue 3: singleton re-use across `NETT.run()` (`executor.py`, `loading_bar_queue.py`, `memory.py`) — remove `@singleton` from Executor and LoadingBarQueue; add `reset()` to MemoryManager

## P1 — Important

- [ ] **P1** Fix Issue 4: heartbeat during eval to keep training Unity alive (`src/nett/brain/brain.py`) — background thread calling `training_env.reset()` in `_KeepAliveEvalCallback`
- [ ] **P1** Fix Issue 5: Unity graceful close via `Application.Quit()` (`ChickAgent.cs`) — uncomment or re-add the quit call
- [ ] **P1** Fix Issue 6a: investigate X+1 step logging off-by-one (`ChickAgent.cs`, `LogChannel.cs`) — audit ML-Agents episode lifecycle, add guard in `RecordStep()`
- [ ] **P1** Fix Issue 6b: fix `LogBuffer` to discard short episodes (`LogChannel.cs`) — add step counter, only flush if count meets threshold in `NewEpisode()` and `Close()`
