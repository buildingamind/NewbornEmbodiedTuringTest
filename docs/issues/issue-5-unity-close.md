# Issue 5 — Unity won't close gracefully, force-kills

**Status**: Open — fix planned
**Component**: Unity C# — `ChickAgent.cs`
**Priority**: P1

## Description

When Python closes the ML-Agents connection (e.g., at the end of training), Unity does not shut down cleanly. The ML-Agents Python client's timeout fires and force-kills the Unity process via an external OS-level kill. This can leave transient files behind, corrupt log output, and interfere with cleanup on Windows.

## Root Cause

`Application.Quit()` was present in `ChickAgent.cs` but was commented out. As a result, there is no code path that exits the Unity application when Python closes the connection. Python-side `env.close()` kills the process externally, and the ML-Agents Python client's timeout then fires as a fallback to force-kill if the external kill is slow.

## Fix

Uncomment (or re-add) the `Application.Quit()` call in `ChickAgent.cs` so that Unity exits cleanly when ML-Agents signals the episode/episode group is done and the connection is closing.

The call should be placed in the appropriate ML-Agents lifecycle method that fires when the communicator connection is terminated (e.g., `OnDisable()` or a communicator disconnect handler), not just at episode end.

## Affected Files

- `ChickAgent.cs` — re-enable `Application.Quit()`
