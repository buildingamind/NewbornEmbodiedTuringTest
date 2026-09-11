"""Preflight for the one environment variable Kit reads before anything else runs.

⛔ WHY THIS IS A MODULE AND NOT A PASTED `if`. The check below was written on
2026-09-10 into `examples/gate_a_resume.py`, after it had already cost two launches
there. On 2026-09-11 it cost a third -- in `examples/capture_observations.py`, a
different entry point, which did not have it. The lesson had been written down, in
code, with a comment explaining the cost, and it still did not travel, because it was
written at the INSTANCE rather than at the CLASS.

⛔ WHAT THE FAILURE LOOKS LIKE, WHICH IS WHY IT COSTS A LAUNCH EVERY TIME. Kit
bootstraps at import and, with the licence unaccepted, calls `input()` for "Do you
accept the EULA?". Under `nohup`/CI there is no stdin, so that raises EOFError, the
bootstrap SystemExits, and the surfaced message is `Unable to bootstrap inner kit
kernel: EOF when reading a line` -- or, through NETT, `Task validation failed (exit
code 1)` with the real cause 40 lines up a traceback about something else. Nothing in
either message names the licence or the variable.

⚠ THE CHECK IS CHEAP AND THE FAILURE IS NOT. Booting Kit far enough to hit the prompt
takes seconds to minutes depending on the entry point; a job launched under `nohup`
overnight can burn the slot and report a cause that sends the reader to the wrong file.
"""

from __future__ import annotations

import os
import sys

_MESSAGE = (
    "⛔ OMNI_KIT_ACCEPT_EULA is unset and there is no tty to answer the licence prompt "
    "on. Kit's import-time bootstrap will read stdin, get EOF, and exit -- which "
    "surfaces as 'Unable to bootstrap inner kit kernel: EOF when reading a line', or "
    "as 'Task validation failed (exit code 1)' with the real cause buried in a "
    "subprocess traceback. Re-invoke with OMNI_KIT_ACCEPT_EULA=YES."
)


def eula_blocks_boot(env=None, stdin=None) -> bool:
    """True iff Kit will hit an unanswerable licence prompt in this process.

    Pure and injectable so it can be tested without a Kit install: the accept variable
    must be set, OR there must be a tty capable of answering the prompt.
    """
    env = os.environ if env is None else env
    stdin = sys.stdin if stdin is None else stdin
    if env.get("OMNI_KIT_ACCEPT_EULA"):
        return False
    try:
        return not stdin.isatty()
    except Exception:
        # A replaced or closed stdin cannot answer a prompt either.
        return True


def require_eula_or_explain(env=None, stdin=None, stream=None) -> bool:
    """Print the explanation and return False if Kit cannot boot here; else True.

    Callers that can return an exit code should do so; `Environment.load` raises instead,
    because by then there is no caller left to return to.
    """
    if not eula_blocks_boot(env, stdin):
        return True
    print(_MESSAGE, file=sys.stderr if stream is None else stream)
    return False
