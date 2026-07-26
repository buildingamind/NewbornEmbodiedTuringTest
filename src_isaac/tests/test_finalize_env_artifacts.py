"""The worker must flush the env's artifacts BEFORE its os._exit teardown.

Regression guard: _exit_worker_cleanly ends in ``atexit.register(os._exit, 0)``,
which skips interpreter finalization, so anything only written by NETTEnv.close()
was lost -- measured as ZERO profiler JSONs across every run on disk, plus the last
grid-video round's manifest. _finalize_env_artifacts unwraps to the real env and
calls its Kit-free ``finalize_artifacts()``.
"""
from __future__ import annotations

import logging

from nett_skrl.runtime.task_runner import _finalize_env_artifacts

logger = logging.getLogger("test")


class _Env:
    def __init__(self):
        self.finalized = 0

    def finalize_artifacts(self):
        self.finalized += 1


class _Wrapper:
    """Mimics the skrl/body wrapper chain: exposes the inner env as ``_env``."""

    def __init__(self, inner):
        self._env = inner


def test_finalizes_a_bare_env():
    env = _Env()
    _finalize_env_artifacts(env, logger)
    assert env.finalized == 1


def test_unwraps_nested_wrappers():
    env = _Env()
    _finalize_env_artifacts(_Wrapper(_Wrapper(env)), logger)
    assert env.finalized == 1


def test_prefers_unwrapped_attribute():
    class _Skrl:
        def __init__(self, inner):
            self.unwrapped = inner

    env = _Env()
    _finalize_env_artifacts(_Skrl(env), logger)
    assert env.finalized == 1


def test_missing_finalize_is_not_fatal():
    class _Plain:
        pass

    _finalize_env_artifacts(_Plain(), logger)  # must not raise


def test_finalize_exception_is_swallowed():
    class _Boom:
        def finalize_artifacts(self):
            raise RuntimeError("teardown blew up")

    # A teardown failure must never fail an otherwise-finished run.
    _finalize_env_artifacts(_Boom(), logger)


def test_cycle_does_not_hang():
    class _Loop:
        pass

    a, b = _Loop(), _Loop()
    a._env, b._env = b, a
    _finalize_env_artifacts(a, logger)  # bounded walk + seen-set: must return
