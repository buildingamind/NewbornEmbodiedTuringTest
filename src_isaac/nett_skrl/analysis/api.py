"""Native post-run analysis helpers for the Isaac/skrl backend.

Each public function consumes a NETT run directory (the path produced by
``NETT(...).run(output_path=...)``) and produces a sibling analysis tree:

* :func:`train_viz` — parses each brain's tfevents file, plots the reward
  curve per condition, writes ``train_rewards.csv`` + one PNG per condition.
* :func:`test_viz` — loads the LogChannel test CSVs and computes the percent
  of steps each brain spent looking at the *correct* monitor (per imprint
  condition × test condition), writes ``test_preferences.csv`` + one bar
  chart per imprint condition.
* :func:`analyze` — runs both of the above and writes a summary JSON.
* :func:`merge` — concatenates the analysis CSVs from multiple runs and
  regenerates the aggregated plots so per-cohort comparisons can be made.
* :func:`normalize_isaac_output` — copies the Isaac run tree into a legacy-
  shaped layout (``recordings/agent``, ``recordings/chamber``) so older
  tooling that expected the Unity-era directory shape still works.

matplotlib and tensorboard are optional dependencies; imports are lazy and
each function fails with a helpful message if they're missing.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("nett.analysis")

import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Chamber geometry defaults — match ``NETTEnvCfg`` so the gaze-direction
# heuristic in ``test_viz`` matches the world the agents actually trained in.
# Override via the ``chamber_half_x`` argument if a config customised these.
# ---------------------------------------------------------------------------

DEFAULT_CHAMBER_HALF_X = 33.15
DEFAULT_CHAMBER_HALF_Y = 21.0

_REWARD_TAG_HINTS = ("Reward/Total reward (mean)", "Reward / Total reward (mean)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze(
    run_path: str | Path,
    output_path: str | Path | None = None,
    *,
    chamber_half_x: float = DEFAULT_CHAMBER_HALF_X,
) -> Path:
    """Run train + test analysis end-to-end on one NETT output dir.

    Produces an ``analysis/`` sibling tree with both viz subdirs and a
    ``summary.json`` capturing per-condition headline numbers (final-train
    mean reward and test correct-monitor preference percent).
    """
    root = Path(run_path)
    out = Path(output_path) if output_path else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    train_out = train_viz(root, out / "train")
    test_out = test_viz(root, out / "test", chamber_half_x=chamber_half_x)

    summary = _summary_from_outputs(train_out, test_out)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return out


def train_viz(
    run_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Plot reward curves per brain from tfevents under ``wandb_runs/``."""
    root = Path(run_path)
    out = Path(output_path) if output_path else root / "analysis_train"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, int, float]] = []  # (condition, brain, step, reward)
    matplotlib.use("Agg", force=False)
    EA = _import_event_accumulator()

    for cond_dir in _condition_dirs(root):
        condition = cond_dir.name
        per_brain: dict[str, list[tuple[int, float]]] = {}
        for brain_dir in sorted((cond_dir / "wandb_runs").glob("brain_*")):
            tfevents = sorted(brain_dir.glob("events.out.tfevents.*"))
            if not tfevents:
                continue
            curve = _reward_curve_from_tfevents(EA, tfevents[0])
            if not curve:
                continue
            per_brain[brain_dir.name] = curve
            rows.extend((condition, brain_dir.name, s, v) for s, v in curve)

        if per_brain and plt is not None:
            fig, ax = plt.subplots(figsize=(8, 5))
            for brain, curve in sorted(per_brain.items()):
                steps, vals = zip(*curve)
                ax.plot(steps, vals, label=brain, linewidth=1.5)
            ax.set_xlabel("Training step")
            ax.set_ylabel("Mean reward")
            ax.set_title(f"Training reward — {condition}")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(out / f"train_reward_{condition}.png", dpi=120)
            plt.close(fig)

    _write_csv(
        out / "train_rewards.csv", ["condition", "brain", "step", "reward"], rows
    )
    return out


def test_viz(
    run_path: str | Path,
    output_path: str | Path | None = None,
    *,
    chamber_half_x: float = DEFAULT_CHAMBER_HALF_X,
) -> Path:
    """Compute the correct-monitor preference per brain × test condition."""
    root = Path(run_path)
    out = Path(output_path) if output_path else root / "analysis_test"
    out.mkdir(parents=True, exist_ok=True)

    matplotlib.use("Agg", force=False)
    rows: list[tuple[str, str, str, int, float]] = []
    # (imprint, test_cond, brain_env_id, n_steps, correct_pct)

    for cond_dir in _condition_dirs(root):
        imprint = cond_dir.name
        logs = cond_dir / "logs"
        for csv_path in sorted(logs.glob("test_*.csv")):
            for row in _test_preference_rows(csv_path, imprint, chamber_half_x):
                rows.append(row)

        # Per-imprint bar chart aggregating across brains for each test cond.
        if plt is not None:
            _plot_test_preferences(
                plt, out, imprint, [r for r in rows if r[0] == imprint]
            )

    _write_csv(
        out / "test_preferences.csv",
        ["imprint", "test_condition", "brain_env_id", "n_steps", "correct_pct"],
        rows,
    )
    return out


def log_analysis_to_wandb(
    run_dir: str | Path,
    analysis_dir: str | Path | None = None,
) -> None:
    """Resume per-condition W&B runs and upload analysis PNGs and summary scalars.

    Reads ``run_dir/config.yaml`` to recover the experiment name and wandb
    settings, then for each condition resumes brain_1's run (using
    :func:`nett_skrl.recording.wandb.wandb_run_id`) and uploads:

    * ``analysis/train_reward_{condition}.png`` as ``analysis/train_reward_curve``
    * ``analysis/test_preference_{condition}.png`` as ``analysis/test_preference``
    * Per-condition scalars from ``analysis/summary.json``

    Does nothing when ``brain.wandb.mode`` is ``"disabled"`` or when
    ``config.yaml`` is absent.
    """
    import json
    import yaml as _yaml

    run_dir = Path(run_dir)
    analysis_dir = Path(analysis_dir) if analysis_dir else run_dir / "analysis"

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        logger.warning("log_analysis_to_wandb: no config.yaml at %s", config_path)
        return

    with config_path.open() as f:
        config = _yaml.safe_load(f)

    run_name = config.get("name", run_dir.name)
    brain_cfg = config.get("brain") or {}
    wandb_cfg = brain_cfg.get("wandb") or {}
    mode = wandb_cfg.get("mode", "online")
    if mode == "disabled":
        return

    try:
        import wandb
        from nett_skrl.recording.wandb import wandb_run_id
    except ImportError as exc:
        logger.warning("log_analysis_to_wandb: missing dependency (%s)", exc)
        return

    project = wandb_cfg.get("project", "nett-skrl")
    entity = wandb_cfg.get("entity")

    summary: dict = {}
    summary_path = analysis_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)

    for cond_dir in sorted(run_dir.iterdir()):
        if not cond_dir.is_dir() or cond_dir.name.startswith((".", "_")):
            continue
        if not ((cond_dir / "logs").exists() or (cond_dir / "wandb_runs").exists()):
            continue
        condition = cond_dir.name

        run_id = wandb_run_id(run_name, condition, 1)
        init_kwargs: dict = {
            "id": run_id,
            "resume": "allow",
            "project": project,
            "mode": mode,
            "dir": str(run_dir),
        }
        if entity:
            init_kwargs["entity"] = entity

        try:
            run = wandb.init(**init_kwargs)
        except Exception:
            logger.warning(
                "log_analysis_to_wandb: wandb.init failed for condition %s",
                condition,
                exc_info=True,
            )
            continue

        if run is None:
            continue

        try:
            payload: dict = {}

            train_png = analysis_dir / "train" / f"train_reward_{condition}.png"
            if train_png.exists():
                payload["analysis/train_reward_curve"] = wandb.Image(str(train_png))

            test_png = analysis_dir / "test" / f"test_preference_{condition}.png"
            if test_png.exists():
                payload["analysis/test_preference"] = wandb.Image(str(test_png))

            if "train" in summary and condition in summary["train"]:
                for brain, data in summary["train"][condition].items():
                    val = data.get("final_reward_tail_mean")
                    if val is not None:
                        payload[f"analysis/train_final_reward_{brain}"] = float(val)

            if "test" in summary and condition in summary["test"]:
                for tc, data in summary["test"][condition].items():
                    val = data.get("correct_pct_mean")
                    if val is not None:
                        payload[f"analysis/test_{tc}_correct_pct"] = float(val)

            if payload:
                run.log(payload)
        except Exception:
            logger.warning(
                "log_analysis_to_wandb: error logging condition %s", condition, exc_info=True
            )
        finally:
            try:
                run.finish()
            except Exception:
                pass


def merge(paths: Iterable[str | Path], output_path: str | Path) -> Path:
    """Combine multiple analysis output trees + re-aggregate the CSVs.

    Each input path should be a directory produced by :func:`analyze` (or a
    subset of its children — :func:`train_viz` / :func:`test_viz` outputs
    work too). Files are copied side-by-side, then ``train_rewards.csv`` /
    ``test_preferences.csv`` are concatenated and the corresponding plots
    regenerated against the combined data.
    """
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        src = Path(path)
        if src.exists():
            _copy_tree(src, out)

    matplotlib.use("Agg", force=False)
    train_csv = _find_first(out, "train_rewards.csv")
    if train_csv and plt is not None:
        _replot_train_from_csv(plt, train_csv)
    test_csv = _find_first(out, "test_preferences.csv")
    if test_csv and plt is not None:
        _replot_test_from_csv(plt, test_csv)
    return out


def normalize_isaac_output(
    run_path: str | Path, output_path: str | Path | None = None
) -> Path:
    """Mirror an Isaac run directory into the legacy-shaped layout."""
    src = Path(run_path)
    dst = Path(output_path) if output_path is not None else src / "analysis_legacy"
    dst.mkdir(parents=True, exist_ok=True)
    _copy_tree(src / "logs", dst / "logs")
    _copy_tree(src / "recordings" / "egocentric", dst / "recordings" / "agent")
    _copy_tree(src / "recordings" / "chamber", dst / "recordings" / "chamber")
    return dst


# Pytest's test-collection rule treats top-level callables named ``test_*``
# as test functions; opt out for the test-viz public API.
test_viz.__test__ = False


__all__ = [
    "DEFAULT_CHAMBER_HALF_X",
    "DEFAULT_CHAMBER_HALF_Y",
    "analyze",
    "log_analysis_to_wandb",
    "merge",
    "normalize_isaac_output",
    "test_viz",
    "train_viz",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _condition_dirs(root: Path) -> list[Path]:
    """Yield per-condition output directories under one run dir."""
    if not root.exists():
        return []
    out: list[Path] = []
    for child in sorted(root.iterdir()):
        # Skip anything obviously not a condition tree.
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        if (child / "logs").exists() or (child / "wandb_runs").exists():
            out.append(child)
    return out


def _reward_curve_from_tfevents(EA, tfevents_path: Path) -> list[tuple[int, float]]:
    ea = EA(str(tfevents_path))
    ea.Reload()
    scalar_tags = ea.Tags().get("scalars", [])
    tag = next((t for t in _REWARD_TAG_HINTS if t in scalar_tags), None)
    if tag is None:
        tag = next(
            (t for t in scalar_tags if "reward" in t.lower() and "mean" in t.lower()),
            None,
        )
    if tag is None:
        return []
    return [(int(s.step), float(s.value)) for s in ea.Scalars(tag)]


def looking_at_monitor(
    agent_x: float, agent_z: float, yaw_deg: float, half_x: float
) -> str:
    """Return ``"left"`` or ``"right"`` for the monitor the agent's forward
    vector points more toward, given the NETT yaw convention
    (``forward = (-sin(yaw), cos(yaw))``)."""
    rad = math.radians(yaw_deg)
    fx, fy = -math.sin(rad), math.cos(rad)
    # Monitor centers sit on the X-walls at world y = 0.
    left_dx, left_dy = -half_x - agent_x, -agent_z
    right_dx, right_dy = half_x - agent_x, -agent_z
    left_norm = math.hypot(left_dx, left_dy) or 1.0
    right_norm = math.hypot(right_dx, right_dy) or 1.0
    left_dot = (left_dx * fx + left_dy * fy) / left_norm
    right_dot = (right_dx * fx + right_dy * fy) / right_norm
    return "left" if left_dot > right_dot else "right"


def _test_preference_rows(
    csv_path: Path,
    imprint: str,
    half_x: float,
) -> list[tuple[str, str, str, int, float]]:
    """Aggregate per (env_id, test_cond) preference percentages from one CSV."""
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            buckets[(row["env_id"], row["test.cond"])].append(row)

    out: list[tuple[str, str, str, int, float]] = []
    for (env_id, test_cond), rows in buckets.items():
        if not rows:
            continue
        correct = 0
        for row in rows:
            try:
                ax = float(row["agent.x"])
                az = float(row["agent.z"])
                yaw = float(row["agent.angle"])
            except (KeyError, ValueError):
                continue
            looking = looking_at_monitor(ax, az, yaw, half_x)
            if looking == row.get("correct.monitor", ""):
                correct += 1
        out.append((imprint, test_cond, env_id, len(rows), correct / len(rows)))
    return out


def _plot_test_preferences(plt, out_dir: Path, imprint: str, rows: list[tuple]) -> None:
    """Bar chart of correct-monitor pct per test condition, error bars across brains."""
    if not rows:
        return
    # rows: (imprint, test_cond, env_id, n_steps, correct_pct)
    by_cond: dict[str, list[float]] = defaultdict(list)
    for _, test_cond, _, _, pct in rows:
        by_cond[test_cond].append(pct)
    if not by_cond:
        return
    labels = sorted(by_cond)
    means = [sum(by_cond[c]) / len(by_cond[c]) for c in labels]
    stds = [_stddev(by_cond[c]) for c in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    ax.bar(labels, means, yerr=stds, capsize=4, color="#3a6ea5", alpha=0.85)
    ax.axhline(0.5, linestyle="--", color="grey", linewidth=1, label="chance")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of steps facing correct monitor")
    ax.set_title(f"Test preference — imprint {imprint}")
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / f"test_preference_{imprint}.png", dpi=120)
    plt.close(fig)


def _summary_from_outputs(train_dir: Path, test_dir: Path) -> dict:
    summary: dict = {"train": {}, "test": {}}
    train_csv = train_dir / "train_rewards.csv"
    if train_csv.exists():
        per_brain_last: dict[tuple[str, str], list[float]] = defaultdict(list)
        with train_csv.open() as f:
            for row in csv.DictReader(f):
                per_brain_last[(row["condition"], row["brain"])].append(
                    float(row["reward"])
                )
        for (cond, brain), values in per_brain_last.items():
            tail = values[-max(1, len(values) // 4) :]
            summary["train"].setdefault(cond, {})[brain] = {
                "final_reward_tail_mean": sum(tail) / len(tail),
                "scalar_count": len(values),
            }

    test_csv = test_dir / "test_preferences.csv"
    if test_csv.exists():
        per_cond: dict[tuple[str, str], list[float]] = defaultdict(list)
        with test_csv.open() as f:
            for row in csv.DictReader(f):
                per_cond[(row["imprint"], row["test_condition"])].append(
                    float(row["correct_pct"])
                )
        for (imp, tc), values in per_cond.items():
            summary["test"].setdefault(imp, {})[tc] = {
                "correct_pct_mean": sum(values) / len(values),
                "correct_pct_stddev": _stddev(values),
                "n_brains": len(values),
            }
    return summary


def _replot_train_from_csv(plt, csv_path: Path) -> None:
    """Re-generate per-condition train reward plots from the merged CSV."""
    per_cond_brain: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            per_cond_brain[row["condition"]][row["brain"]].append(
                (int(row["step"]), float(row["reward"]))
            )
    out_dir = csv_path.parent
    for condition, per_brain in per_cond_brain.items():
        if not per_brain:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for brain, curve in sorted(per_brain.items()):
            curve.sort()
            steps, vals = zip(*curve)
            ax.plot(steps, vals, label=brain, linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean reward")
        ax.set_title(f"Training reward (merged) — {condition}")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"merged_train_reward_{condition}.png", dpi=120)
        plt.close(fig)


def _replot_test_from_csv(plt, csv_path: Path) -> None:
    """Re-generate per-imprint test preference bars from the merged CSV."""
    rows_by_imprint: dict[str, list[tuple]] = defaultdict(list)
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            rows_by_imprint[row["imprint"]].append(
                (
                    row["imprint"],
                    row["test_condition"],
                    row["brain_env_id"],
                    int(row["n_steps"]),
                    float(row["correct_pct"]),
                )
            )
    for imprint, rows in rows_by_imprint.items():
        _plot_test_preferences(plt, csv_path.parent, f"merged_{imprint}", rows)


def _stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _find_first(root: Path, name: str) -> Path | None:
    matches = list(root.rglob(name))
    return matches[0] if matches else None


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _import_event_accumulator():
    """Lazy tensorboard import; raises ImportError with a helpful message."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as e:
        raise ImportError(
            "tensorboard is required to parse train reward curves from tfevents. "
            "Install with `pip install tensorboard`."
        ) from e
    return EventAccumulator
