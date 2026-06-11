"""Rerun final test/analysis for an existing example training run."""

from __future__ import annotations

import argparse
import copy
import importlib
import logging
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb

from _train_common import assert_target_preferences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("module", help="example module name, e.g. train_compact_cnn")
    parser.add_argument("run_name", help="existing run directory name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("nett.rerun_test")

    module = importlib.import_module(args.module)
    config = copy.deepcopy(module.CONFIG)
    config["name"] = args.run_name
    config["episodes"] = {"train": 0, "test": 1}
    output = Path(module.OUTPUT)
    run_dir = output / args.run_name

    log.info("rerunning final test only: %s", run_dir)
    NETT(config).run(output_path=str(output), devices=[0], verbose=True)
    log.info("test complete; analyzing %s", run_dir)
    out = analyze(run_dir)
    log_analysis_to_wandb(run_dir, out)
    assert_target_preferences(out)
    log.info("analysis passed: %s", out)


if __name__ == "__main__":
    main()
