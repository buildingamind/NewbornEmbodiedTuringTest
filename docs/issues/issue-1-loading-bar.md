# Issue 1 — Memory-estimation loading bar never removed

**Status**: Open — fix planned
**Component**: Python — `src/nett/nett.py`
**Priority**: P0

## Description

When memory estimation for a task raises an exception, the loading bar entry for that task is never cleaned up. The bar remains visible as a "ghost bar" for the duration of the run.

## Root Cause

In `NETT._calculate_task_memory()`, `executor.loading_bar.remove(label)` is only called in the `try` block. If the memory-estimation task raises, execution jumps to the `except`/`finally` block. The `finally` block only cleans up the temporary directory — it does not remove the loading bar entry.

```python
# Current (broken) pattern:
try:
    ...
    executor.loading_bar.remove(label)
finally:
    shutil.rmtree(tmp_dir)
```

## Fix

Move `executor.loading_bar.remove(label)` into the `finally` block so it always runs regardless of whether the estimation task succeeds or fails.

```python
# Fixed pattern:
try:
    ...
finally:
    executor.loading_bar.remove(label)
    shutil.rmtree(tmp_dir)
```

## Affected Files

- `src/nett/nett.py` — `NETT._calculate_task_memory()`
