"""Benchmark the detectors on real AWS CloudWatch data.

    python scripts/fetch_nab_dataset.py            # once
    python scripts/benchmark.py                    # full comparison table
    python scripts/benchmark.py --json             # machine readable
    python scripts/benchmark.py --alert-rate 0.01  # tighter alert budget
    python scripts/benchmark.py --markdown docs/evaluation.md --run-tag nightly

Writes an experiment record to ``runs/<timestamp>.json`` (protocol, dataset,
per-series results, aggregates) so results are reproducible and diffable.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PROJECT_ROOT
from src.data import load_nab_dataset
from src.detection.evaluation import (
    DEFAULT_ALERT_RATE,
    EvaluationProtocol,
    benchmark,
    dataset_summary,
    default_scorers,
)

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "nab" / "realAWSCloudwatch"
LABELS_PATH = PROJECT_ROOT / "data" / "raw" / "nab" / "combined_windows.json"
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_DOC = PROJECT_ROOT / "docs" / "evaluation.md"

SCORER_LABELS = {
    "isolation_forest": "IsolationForest (this repo)",
    "rolling_zscore": "Rolling z-score",
    "ewma": "EWMA control chart",
    "global_threshold": "Global mean+k-sigma",
    "profile_threshold": "cloud_profiles.yaml warn_high",
}

COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("scorer", "detector", ""),
    ("event_recall", "event recall", "{:.2f}"),
    ("precision", "point precision", "{:.3f}"),
    ("recall", "point recall", "{:.3f}"),
    ("f1", "point F1", "{:.3f}"),
    ("precision_strict", "point precision (strict)", "{:.3f}"),
    ("mean_detection_delay_minutes", "delay (min)", "{:.0f}"),
    ("false_alarms_per_day", "false alarms/day", "{:.2f}"),
    ("average_precision", "avg precision", "{:.3f}"),
    ("roc_auc", "ROC AUC", "{:.3f}"),
)
NORMAL_ONLY_COLUMN = "false_alarms_per_day_normal_only"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark detectors on real AWS CloudWatch telemetry")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="directory holding the NAB series")
    parser.add_argument("--labels", default=None, help="path to combined_windows.json")
    parser.add_argument("--train-fraction", type=float, default=0.4, help="fraction of each series used for training")
    parser.add_argument("--alert-rate", type=float, default=DEFAULT_ALERT_RATE, help="target alert rate on training data")
    parser.add_argument("--contamination", type=float, default=0.05, help="IsolationForest contamination")
    parser.add_argument("--n-estimators", type=int, default=150, help="IsolationForest trees")
    parser.add_argument("--zscore-k", type=float, default=3.0, help="sigma multiple for the rolling z-score scorer")
    parser.add_argument("--history-window", type=int, default=30, help="rolling window length in samples")
    parser.add_argument("--ablate", default="", help="comma separated features to drop (feature ablation study)")
    parser.add_argument("--ewma-span", type=int, default=20, help="EWMA span")
    parser.add_argument("--proximity-minutes", type=float, default=60.0, help="credit a hit this long outside a window")
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N series")
    parser.add_argument("--markdown", default=None, help="also write a markdown report to this path")
    parser.add_argument("--scorers", default="", help="comma separated subset of scorers to run (default: all)")
    parser.add_argument("--run-tag", default="manual", help="label stored with the run record")
    parser.add_argument("--json", action="store_true", help="print the full result document as JSON")
    return parser


def load_series(args: argparse.Namespace):
    labels = Path(args.labels) if args.labels else LABELS_PATH
    series = load_nab_dataset(args.data_root, labels, max_gap_multiple=3)
    return series[: args.limit] if args.limit else series


def protocol_from_args(args: argparse.Namespace) -> EvaluationProtocol:
    ablated = tuple(token.strip() for token in args.ablate.split(",") if token.strip())
    return EvaluationProtocol(
        train_fraction=args.train_fraction,
        alert_rate=args.alert_rate,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        history_window=args.history_window,
        ablated_features=ablated,
        zscore_k=args.zscore_k,
        ewma_span=args.ewma_span,
        proximity=timedelta(minutes=args.proximity_minutes),
    )


def select_scorers(args: argparse.Namespace, protocol: EvaluationProtocol):
    """Scorer subset requested on the command line (all of them by default)."""
    factories = default_scorers(protocol)
    if not args.scorers:
        return factories
    wanted = {token.strip() for token in args.scorers.split(",") if token.strip()}
    known = {factory({}).name for factory in factories}
    unknown = wanted - known
    if unknown:
        raise SystemExit(f"unknown scorer(s): {sorted(unknown)}; known: {sorted(known)}")
    return [factory for factory in factories if factory({}).name in wanted]


def markdown_table(report: dict) -> str:
    summary = report["summary"]
    header = "| " + " | ".join(label for _, label, _ in COLUMNS) + " |"
    divider = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    rows = [header, divider]
    for scorer, metrics in summary.items():
        cells = [SCORER_LABELS.get(scorer, scorer)]
        for key, _, fmt in COLUMNS[1:]:
            value = metrics.get(key)
            cells.append(fmt.format(value) if isinstance(value, (int, float)) else "-")
        rows.append("| " + " | ".join(cells) + " |")
    legend = "\n".join(f"* `{key}` - {text}" for key, text in SCORER_LABELS.items())
    return f"{legend}\n\n" + "\n".join(rows)


def markdown_report(report: dict, series) -> str:
    protocol = report["protocol"]
    dataset = report["dataset"]
    per_series_caveat = (
        "* each model trains on the training slice of the same series: there is no cross-series "
        "transfer, so these numbers are a lower bound on what a pooled model would achieve"
    )
    normal_only_intro = (
        "The benchmark ships series whose evaluation slice contains no anomaly at all (including one "
        "control series with no windows). They cannot contribute recall, but they are the cleanest "
        "measure of how often an on-call engineer would have been paged for nothing."
    )
    lines = [
        "# Benchmark: real AWS CloudWatch telemetry",
        "",
        f"Generated {report['generated_at']} (`{report['run_tag']}`).",
        "",
        "## Protocol",
        "",
        f"* dataset: NAB `realAWSCloudwatch` - {dataset['series']} series, {dataset['points']} points, "
        f"{dataset['windows']} labelled windows",
        f"* temporal split: first {protocol['train_fraction']:.0%} trains, remainder is scored",
        f"* operating point: threshold = {protocol['alert_rate']:.1%} quantile of each detector's *training* scores",
        f"* proximity credit: {protocol['proximity_minutes']:.0f} min outside a window",
        f"* history window: {protocol['history_window']} samples, minimum {protocol['min_history']}",
        f"* IsolationForest: contamination {protocol['contamination']}, {protocol['n_estimators']} trees, "
        f"seed {protocol['random_state']}",
        *(
            [f"* ablation: dropped features {', '.join(protocol['ablated_features'])}"]
            if protocol.get("ablated_features")
            else []
        ),
        "",
        "## Results (macro-average over series with an evaluation window)",
        "",
        markdown_table(report),
        "",
        "## Caveats",
        "",
        *(f"* {warning}" for warning in report["warnings"]),
        per_series_caveat,
        "",
        "## Per-series event recall",
        "",
        "| detector | series scored | windows | windows detected |",
        "| --- | --- | --- | --- |",
    ]
    for scorer, metrics in report["summary"].items():
        lines.append(
            f"| {SCORER_LABELS.get(scorer, scorer)} | {metrics['scored_series']} | "
            f"{metrics['windows']} | {metrics['windows_detected']} |"
        )
    lines += [
        "",
        "## False alarms on normal-only series",
        "",
        normal_only_intro,
        "",
        "| detector | false alarms/day |",
        "| --- | --- |",
    ]
    for scorer, metrics in report["summary"].items():
        lines.append(f"| {SCORER_LABELS.get(scorer, scorer)} | {metrics[NORMAL_ONLY_COLUMN]:.2f} |")
    lines += ["", "## Series", "", "| series | metric | points | windows |", "| --- | --- | --- | --- |"]
    for item in series:
        info = item.summary()
        lines.append(f"| `{item.name}` | {info['metric']} | {info['points']} | {info['windows']} |")
    return "\n".join(lines) + "\n"


def save_run(report: dict) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = report["generated_at"].replace(":", "").replace("-", "").replace("+00:00", "Z")
    path = RUNS_DIR / f"benchmark-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.data_root).is_dir():
        print(f"error: {args.data_root} not found - run scripts/fetch_nab_dataset.py first", file=sys.stderr)
        return 2

    series = load_series(args)
    if not series:
        print("error: no series loaded", file=sys.stderr)
        return 2

    print(
        f"evaluating {len(series)} series | train_fraction={args.train_fraction} "
        f"alert_rate={args.alert_rate} contamination={args.contamination}",
        flush=True,
    )
    selected = select_scorers(args, protocol_from_args(args))
    report = benchmark(series, protocol_from_args(args), selected)
    report["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["run_tag"] = args.run_tag
    report["dataset"] = dataset_summary(series)

    run_path = save_run(report)
    if args.markdown:
        target = Path(args.markdown)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown_report(report, series), encoding="utf-8")
        print(f"markdown written to {target}")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print()
        print(markdown_table(report))
        print()
        for scorer, metrics in report["summary"].items():
            print(
                f"{SCORER_LABELS.get(scorer, scorer):32s} "
                f"scored={metrics['scored_series']}/{metrics['series']} "
                f"normal_only={metrics['normal_only_series']} "
                f"false_alarms/day(normal-only)={metrics['false_alarms_per_day_normal_only']:.2f}"
            )
        print(f"\nrun record: {run_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
