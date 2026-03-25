# Issue 4 — Training Unity process times out during long eval callbacks

**Status**: Open — fix planned
**Component**: Python — `src/nett/brain/brain.py`
**Priority**: P1

## Description

During a long eval-callback run (which can take ~30 minutes), the training Unity process sits completely idle. The video-sync watchdog in `ChickAgent.cs` interprets this idle period as a deadlock and kills the process. This was worked around by disabling the watchdog, but the underlying issue remains.

## Root Cause

`_KeepAliveEvalCallback` in `src/nett/brain/brain.py` resets the training environment *after* eval completes, but does nothing *during* the eval. If the eval takes ~30 minutes, the training Unity executable receives no steps or resets for the entire duration. The Unity watchdog (when enabled) times out and terminates the process.

The watchdog was disabled as a workaround, which means other real deadlocks would also go undetected.

## Fix

Extend `_KeepAliveEvalCallback` with a background heartbeat thread that periodically calls `training_env.reset()` while the eval callback is running. No Unity C# changes are needed.

```python
import threading

class _KeepAliveEvalCallback(EventCallback):
    def __init__(self, eval_callback, training_env, heartbeat_interval=60, **kwargs):
        super().__init__(eval_callback, **kwargs)
        self.training_env = training_env
        self.heartbeat_interval = heartbeat_interval
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread = None

    def _heartbeat(self):
        while not self._stop_heartbeat.wait(self.heartbeat_interval):
            self.training_env.reset()

    def _on_event(self) -> bool:
        self._stop_heartbeat.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._heartbeat_thread.start()
        result = super()._on_event()
        self._stop_heartbeat.set()
        self._heartbeat_thread.join()
        return result
```

## Affected Files

- `src/nett/brain/brain.py` — `_KeepAliveEvalCallback`
