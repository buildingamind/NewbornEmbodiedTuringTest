"""Run the nett_skrl end-to-end smoke training.

Usage (PYTHONPATH=src_isaac):
    /home/zach/nett_private/bin/python src_isaac/examples/run_smoke.py \
        --config src_isaac/examples/smoke.yaml \
        --output /tmp/smoke_run

The script wraps the boilerplate of constructing a ``NETT(...)`` and calling
``.run(output_path=...)``. Useful as both a smoke test and a usage example.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/nett_skrl_run"))
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    from nett_skrl import NETT

    NETT([str(args.config)]).run(output_path=str(args.output), verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
