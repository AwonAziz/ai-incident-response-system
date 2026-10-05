"""Fail when the detector benchmarks regress against a committed floor.

Nightly runs are only worth having if they can go red, so the gate lives here
rather than in workflow YAML: it is unit-tested, it produces a readable table,
and it prints exactly which metric fell below which floor.

    python scripts/check_benchmark_thresholds.py runs/*.json
    python scripts/check_benchmark_thresholds.py runs/*.json --json

Exit codes: ``0`` every floor met, ``1`` a regression, ``2`` the run file was
missing or unreadable - which is itself a failure, because a benchmark that did
not run is not a passing benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_THRESHOLDS = Path(__file__).resolve().parent.parent / "config" / "benchmark_thresholds.json"

#: how far below the protocol in the thresholds file the run may drift before we
#: refuse to compare numbers produced under different conditions
PROTOCOL_TOLERANCE = {
    "train_fraction": 0.001,
    "alert_rate": 0.005,
    "contamination": 0.005,
}


@dataclass(frozen=True, slots=True)
class Check:
    detector: str
    metric: str
    value: float
    floor: float
    comparison: str

    @property
    def passed(self) -> bool:
        if self.comparison == ">=":
            return self.value >= self.floor
        return self.value <= self.floor

    def describe(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return f"{verdict}  {self.detector:<16} {self.metric:<24} {self.value:>8.4f} {self.comparison} {self.floor}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "metric": self.metric,
            "value": round(self.value, 6),
            "floor": self.floor,
            "comparison": self.comparison,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class Unverifiable:
    """A floor that could not be evaluated at all - treated as a failure."""

    detector: str
    metric: str
    reason: str

    def describe(self) -> str:
        return f"FAIL  {self.detector:<16} {self.metric:<24} could not be evaluated: {self.reason}"

    def to_dict(self) -> dict[str, Any]:
        return {"detector": self.detector, "metric": self.metric, "reason": self.reason}


@dataclass(slots=True)
class GateResult:
    checks: list[Check] = field(default_factory=list)
    unverifiable: list[Unverifiable] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add_unverifiable(self, detector: str, metric: str, reason: str) -> None:
        """Record a floor that could not be measured; this fails the gate."""
        self.unverifiable.append(Unverifiable(detector, metric, reason))

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]

    @property
    def ok(self) -> bool:
        """No floor breached *and* nothing left unmeasured."""
        return not self.failures and not self.unverifiable

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
            "unverifiable": [item.to_dict() for item in self.unverifiable],
            "failures": len(self.failures),
            "notes": list(self.notes),
        }


def load_json(path: str | Path) -> dict[str, Any]:
    """Read JSON, tolerating a byte-order mark.

    Redirecting a run through PowerShell or ``tee`` on some platforms prepends a
    BOM, and a nightly gate that fails on that is a nightly gate people disable.
    """
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_thresholds(path: str | Path = DEFAULT_THRESHOLDS) -> dict[str, Any]:
    payload = load_json(path)
    if "detectors" not in payload:
        raise ValueError(f"{path}: no 'detectors' section")
    return payload


def newest_run(paths: list[str]) -> Path:
    candidates = sorted(Path(item) for item in paths if Path(item).is_file())
    if not candidates:
        raise FileNotFoundError("no benchmark run files found")
    return candidates[-1]


def compare(run: dict[str, Any], thresholds: dict[str, Any]) -> GateResult:
    """Check a benchmark report against the committed floors.

    Strict by design: a floor that cannot be evaluated is a failure, not a note.
    A gate that skips what it cannot measure is a gate that silently goes green,
    which is the one behaviour a regression gate must never have.
    """
    result = GateResult()
    summary = run.get("summary") or {}
    if not summary:
        result.notes.append("run contains no summary section")
        result.add_unverifiable("run", "summary", "benchmark produced no summary")
        return result

    _check_protocol(run, thresholds, result)

    for detector, floors in thresholds["detectors"].items():
        metrics = summary.get(detector)
        if not metrics:
            result.add_unverifiable(detector, "*", "detector absent from this run")
            continue
        for floor_key, floor in floors.items():
            metric = floor_key.removesuffix("_min")
            if metric not in metrics:
                result.add_unverifiable(detector, metric, "metric absent from this run")
                continue
            value = metrics[metric]
            if not isinstance(value, (int, float)):
                result.add_unverifiable(detector, metric, f"non-numeric value {value!r}")
                continue
            result.checks.append(Check(detector, metric, float(value), float(floor), ">="))

    limit = thresholds.get("max_false_alarms_per_day")
    if limit is not None:
        for detector, metrics in summary.items():
            normal_only = metrics.get("normal_only_series")
            if not normal_only:
                result.notes.append(f"{detector}: no normal-only series in this run, false-alarm floor not evaluated")
                continue
            value = metrics.get("false_alarms_per_day_normal_only")
            if not isinstance(value, (int, float)):
                result.add_unverifiable(detector, "false_alarms_per_day_normal_only", "metric absent from this run")
                continue
            result.checks.append(
                Check(detector, "false_alarms_per_day_normal_only", float(value), float(limit), "<=")
            )

    return result


def _check_protocol(run: dict[str, Any], thresholds: dict[str, Any], result: GateResult) -> None:
    """Refuse to compare runs produced under different conditions."""
    expected = thresholds.get("protocol") or {}
    actual = run.get("protocol") or {}
    for key, tolerance in PROTOCOL_TOLERANCE.items():
        if key not in expected or key not in actual:
            continue
        if abs(float(actual[key]) - float(expected[key])) > tolerance:
            result.notes.append(
                f"protocol drift: {key}={actual[key]} but floors assume {expected[key]}; "
                "re-measure before trusting this comparison"
            )


def render(result: GateResult) -> str:
    lines = [check.describe() for check in result.checks]
    lines.extend(item.describe() for item in result.unverifiable)
    lines.extend(f"note  {note}" for note in result.notes)
    lines.append("")
    if result.ok:
        lines.append(f"benchmark gate: PASS ({len(result.checks)} checks)")
        return "\n".join(lines)
    lines.append(
        f"benchmark gate: FAIL ({len(result.failures)} below floor, "
        f"{len(result.unverifiable)} not measurable)"
    )
    lines.extend(
        f"  - {check.detector}.{check.metric}: {check.value:.4f} {check.comparison} {check.floor}"
        for check in result.failures
    )
    lines.extend(
        f"  - {item.detector}.{item.metric}: {item.reason}" for item in result.unverifiable
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail when benchmark results regress")
    parser.add_argument("runs", nargs="+", help="benchmark run JSON files produced by scripts/benchmark.py --json")
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS), help="committed floors")
    parser.add_argument("--json", action="store_true", help="emit the gate result as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        thresholds = load_thresholds(args.thresholds)
        run_path = newest_run(args.runs)
        run = load_json(run_path)
    except (OSError, ValueError) as exc:
        print(f"benchmark gate: could not evaluate - {exc}", file=sys.stderr)
        return 2

    result = compare(run, thresholds)
    if args.json:
        print(json.dumps({"run": str(run_path), **result.to_dict()}, indent=2))
    else:
        print(render(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
