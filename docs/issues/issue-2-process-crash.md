# Issue 2 — Worker crash crashes the loading bar queue

**Status**: Open — fix planned
**Component**: Python — `src/nett/nett.py`
**Priority**: P0

## Description

If any worker process raises an unhandled exception, the exception propagates to the main thread and tears down the entire `Executor` context manager while other tasks are still running. This sends `"close"` to the `LoadingBarQueue`, shutting down the Manager process while live tasks are still trying to write updates to it.

## Root Cause

`task_waiter()` in `src/nett/nett.py` calls `done_future.result()` with no exception handling:

```python
# Current (broken) pattern:
for future in as_completed(futures):
    done_future = futures[future]
    done_future.result()
```

When the exception propagates out of `task_waiter()`, it exits the `with Executor() as executor:` block, which calls `__exit__` → `shutdown()` on the `ProcessPoolExecutor`. Simultaneously, the `LoadingBarQueue` context manager's `__exit__` sends `"close"` to the queue Manager. Any still-running worker that tries to push a progress update at this point will encounter a broken Manager connection.

## Fix

Wrap `done_future.result()` in a `try/except`, log the error, free GPU memory for the failed task, and continue processing the remaining futures.

```python
# Fixed pattern:
for future in as_completed(futures):
    done_future = futures[future]
    try:
        done_future.result()
    except Exception as e:
        logger.error(f"Task failed: {e}")
        memory_manager.free(done_future.device)
```

## Affected Files

- `src/nett/nett.py` — `task_waiter()`
- `src/nett/utils/memory.py` — GPU memory release on failure
