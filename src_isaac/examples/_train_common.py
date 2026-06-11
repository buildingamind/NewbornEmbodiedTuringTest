"""Shared post-run checks for NETT training examples."""

from __future__ import annotations

import json
from pathlib import Path

TARGET_TEST_CONDITIONS = ("1color", "2color", "2shape&color")
MIN_CORRECT_PCT = 0.65
REST_CORRECT_PCT = 0.90


def assert_target_preferences(
    analysis_dir: str | Path,
    *,
    threshold: float = MIN_CORRECT_PCT,
    rest_threshold: float = REST_CORRECT_PCT,
    conditions: tuple[str, ...] | None = None,
) -> None:
    """Fail if the final analysis misses or falls short on target conditions.

    ``rest`` is evaluated separately at ``rest_threshold`` (default 90 %) since
    it is the training-identical baseline and should be the easiest condition to
    pass; failure there indicates the agent did not learn the task at all.

    ``conditions`` overrides ``TARGET_TEST_CONDITIONS`` when set, allowing
    per-script customisation (e.g. GuessWhatMoves checks ``2shape`` only).
    """
    summary_path = Path(analysis_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"analysis summary not found: {summary_path}")

    with summary_path.open() as f:
        summary = json.load(f)

    target_conditions = conditions if conditions is not None else TARGET_TEST_CONDITIONS
    tests = summary.get("test", {})
    missing: list[str] = []
    failed: list[str] = []
    for imprint, by_condition in sorted(tests.items()):
        for condition in target_conditions:
            value = by_condition.get(condition, {}).get("correct_pct_mean")
            if value is None:
                missing.append(f"{imprint}/{condition}")
            elif float(value) <= threshold:
                failed.append(f"{imprint}/{condition}={float(value):.3f}")

        rest_value = by_condition.get("rest", {}).get("correct_pct_mean")
        if rest_value is None:
            missing.append(f"{imprint}/rest")
        elif float(rest_value) <= rest_threshold:
            failed.append(f"{imprint}/rest={float(rest_value):.3f}")

    if not tests:
        missing.extend((*target_conditions, "rest"))

    if missing or failed:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if failed:
            details.append(f"failed: {', '.join(failed)}")
        raise RuntimeError("target preference check failed (" + "; ".join(details) + ")")
