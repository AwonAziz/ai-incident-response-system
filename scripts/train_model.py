"""Train the anomaly detection model.

Examples:
    python scripts/train_model.py                        # 500 rounds, defaults
    python scripts/train_model.py --samples 200 --contamination 0.03
    python scripts/train_model.py --input data/sample/baseline_metrics.json
    python scripts/train_model.py --export-sample data/sample/baseline_metrics.json --samples 40
    python scripts/train_model.py --with-anomalies --evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from src.detection.trainer import (
    build_training_set,
    evaluate_on_labels,
    load_dataset,
    save_dataset,
    train_detector,
)


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train the Isolation Forest anomaly detector")
    parser.add_argument("--samples", type=int, default=settings.baseline_samples, help="collection rounds to generate")
    parser.add_argument("--contamination", type=float, default=settings.model_contamination, help="expected anomaly fraction")
    parser.add_argument("--n-estimators", type=int, default=settings.model_n_estimators, help="number of trees")
    parser.add_argument("--seed", type=int, default=settings.random_seed, help="RNG seed")
    parser.add_argument("--output", default=settings.model_path, help="where to write the model bundle")
    parser.add_argument("--input", default=None, help="train from a saved JSON dataset instead of generating")
    parser.add_argument("--export-sample", default=None, help="also write the generated dataset as JSON")
    parser.add_argument("--export-limit", type=int, default=None, help="cap the number of exported rows")
    parser.add_argument("--with-anomalies", action="store_true", help="mix injected anomalies into the training set")
    parser.add_argument("--evaluate", action="store_true", help="score the model against known injected labels")
    parser.add_argument("--min-train-samples", type=int, default=200, help="guard against an empty training set")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    if args.input:
        samples = load_dataset(args.input)
        source = f"{args.input} ({len(samples)} samples)"
    else:
        samples, _ = build_training_set(
            rounds=max(1, args.samples),
            seed=args.seed,
            inject_anomaly=args.with_anomalies,
            settings=settings,
        )
        source = f"generated ({len(samples)} samples from {args.samples} rounds)"

    if len(samples) < args.min_train_samples:
        print(f"error: need at least {args.min_train_samples} training samples, got {len(samples)}", file=sys.stderr)
        return 2

    if args.export_sample:
        path = save_dataset(samples, args.export_sample, limit=args.export_limit)
        print(f"dataset written to {path}")

    detector = train_detector(
        samples,
        settings=settings,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        random_state=args.seed,
    )
    output = detector.save(args.output)

    report = {
        "source": source,
        "model_path": str(output),
        **detector.stats,
    }

    if args.evaluate:
        report["evaluation"] = _evaluate(detector, args.seed, settings)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"trained on {report['source']}")
        print(f"model saved to {report['model_path']}")
        print(f"  samples      : {report['training_samples']}")
        print(f"  metrics      : {', '.join(report['training_metrics'])}")
        print(f"  trees        : {report['n_estimators']} (contamination {report['contamination']})")
        print(f"  features     : {', '.join(report['features'])}")
        print(f"  offset       : {report['decision_offset']:.6f}")
        if "evaluation" in report:
            metrics = report["evaluation"]
            print(
                "  labelled run : "
                f"precision {metrics['precision']:.3f} recall {metrics['recall']:.3f} "
                f"f1 {metrics['f1']:.3f} over {metrics['support']} warm samples "
                f"({metrics['predicted_positive']} flagged / {metrics['actual_positive']} expected)"
            )

    return 0


def _evaluate(detector, seed: int, settings) -> dict[str, float | int]:
    """Score a fresh, anomaly-bearing batch against a rule-derived ground truth.

    A sample counts as a true anomaly when it breaches a HIGH or CRITICAL static
    threshold from ``cloud_profiles.yaml``. That gives an honest, reproducible
    precision/recall number instead of an invented one. Cold-window samples are
    excluded, because the detector deliberately refuses to flag them.
    """
    from src.core.enums import Severity
    from src.detection.rule_engine import RuleEngine

    samples, _ = build_training_set(rounds=60, seed=seed + 1, inject_anomaly=True, settings=settings)
    engine = RuleEngine(settings.profiles)
    warm = [metric for metric in samples if metric.window is not None and metric.window.is_ready(5)]
    labels = [
        1
        if any(v.severity >= Severity.HIGH for v in engine.evaluate(metric))
        else 0
        for metric in warm
    ]
    metrics = evaluate_on_labels(detector, warm, labels)
    metrics["cold_samples_excluded"] = len(samples) - len(warm)
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
