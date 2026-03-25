# Issue 3 — Singletons break on second NETT.run()

**Status**: Open — fix planned
**Component**: Python — `src/nett/utils/executor.py`, `src/nett/utils/loading_bar_queue.py`, `src/nett/utils/memory.py`
**Priority**: P0

## Description

Calling `NETT.run()` a second time in the same Python session returns stale, already-shutdown singleton instances for `Executor`, `LoadingBarQueue`, and `MemoryManager`. Any operation on these dead instances raises an error.

## Root Cause

The `@singleton` decorator caches instances indefinitely in a module-level dict. It has no mechanism to detect that a cached instance has been shut down and needs to be recreated.

- **`Executor`** (`src/nett/utils/executor.py`): After `__exit__` calls `ProcessPoolExecutor.shutdown()`, the singleton cache still holds the dead instance. The next `run()` call gets this dead executor and cannot submit tasks.
- **`LoadingBarQueue`** (`src/nett/utils/loading_bar_queue.py`): After `__exit__` stops the Manager process, the cached instance's queue is broken. New tasks that try to push progress updates get `BrokenPipeError` or similar.
- **`MemoryManager`** (`src/nett/utils/memory.py`): After `nvmlShutdown()` is called, any subsequent NVML call on the cached instance raises `NVMLError`.

## Fix

**Option A (preferred for `Executor` and `LoadingBarQueue`)**: Remove the `@singleton` decorator. These are already used as context managers — constructing a fresh instance per `run()` is safe and correct.

**Option B (preferred for `MemoryManager`)**: Keep the singleton but add a `reset()` classmethod that clears the cached instance, re-initializes NVML, and resets internal state. Call `MemoryManager.reset()` at the start of each `NETT.run()`.

```python
# Example reset() for MemoryManager:
@classmethod
def reset(cls):
    try:
        nvmlShutdown()
    except NVMLError:
        pass
    cls._instance = None
```

## Affected Files

- `src/nett/utils/executor.py` — remove `@singleton`
- `src/nett/utils/loading_bar_queue.py` — remove `@singleton`
- `src/nett/utils/memory.py` — add `reset()` classmethod
