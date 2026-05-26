"""CLI entry point — ``python -m nett_skrl --config foo.yaml`` or ``nett-skrl ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="nett-skrl")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to YAML/JSON config (see schema.json).")
    parser.add_argument("--output", type=Path, default=Path("./nett_skrl_run"),
                        help="Output directory for logs + checkpoints.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-task subprocess stdout.")
    args = parser.parse_args()

    from .nett import NETT

    NETT([str(args.config)]).run(output_path=str(args.output), verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
