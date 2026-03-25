# NETT Known Issues

This directory tracks known bugs and planned fixes for the NETT project.

## Issue Index

| # | Title | Component | Status | File |
|---|-------|-----------|--------|------|
| 1 | Memory-estimation loading bar never removed | Python (`nett.py`) | Open | [issue-1-loading-bar.md](issue-1-loading-bar.md) |
| 2 | Worker crash crashes the loading bar queue | Python (`nett.py`) | Open | [issue-2-process-crash.md](issue-2-process-crash.md) |
| 3 | Singletons break on second `NETT.run()` | Python (`executor.py`, `loading_bar_queue.py`, `memory.py`) | Open | [issue-3-singleton-reuse.md](issue-3-singleton-reuse.md) |
| 4 | Training Unity process times out during long eval callbacks | Python (`brain.py`) | Open | [issue-4-unity-timeout.md](issue-4-unity-timeout.md) |
| 5 | Unity won't close gracefully, force-kills | Unity C# (`ChickAgent.cs`) | Open | [issue-5-unity-close.md](issue-5-unity-close.md) |
| 6 | Unity step-count issues and 1-episode overrun | Unity C# (`ChickAgent.cs`, `LogChannel.cs`) | Open | [issue-6-unity-overstep.md](issue-6-unity-overstep.md) |

## Priority Summary

- **P0** (blocking correctness): Issues 1, 2, 3
- **P1** (important but not blocking): Issues 4, 5, 6

See [TODO.md](TODO.md) for the full prioritized checklist.
