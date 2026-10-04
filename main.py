#!/usr/bin/env python3
"""AI-Powered Incident Response System - command line entry point.

All orchestration lives in :mod:`src.pipeline`; this module only parses
arguments, wires optional components (dashboard, control API) and prints the
session summary.

Usage:
    python main.py                     # live dashboard
    python main.py --train             # retrain the model first
    python main.py --no-dashboard      # headless (logs + console alerts)
    python main.py --inject-anomaly    # periodically degrade a resource
    python main.py --api               # expose the control API on :8080
    python main.py --max-ticks 10      # run a fixed number of cycles (CI/demo)
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from typing import Any

from config.settings import enable_utf8_console, get_settings, setup_logging
from src.api.control_server import ControlServer
from src.pipeline import Pipeline, PipelineConfig

logger = logging.getLogger("main")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI-Powered Incident Response System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train", action="store_true", help="re-train the model before starting")
    parser.add_argument("--no-dashboard", action="store_true", help="run headless (no Rich UI)")
    parser.add_argument("--inject-anomaly", action="store_true", help="periodically inject test degradations")
    parser.add_argument("--inject-every", type=int, default=None, metavar="N", help="ticks between injections")
    parser.add_argument("--max-ticks", type=int, default=None, metavar="N", help="stop after N cycles")
    parser.add_argument("--no-retrain-watch", action="store_true", help="disable automatic retraining on drift")
    parser.add_argument("--no-notify", action="store_true", help="disable the console notifier")
    parser.add_argument("--api", action="store_true", help="start the control API (health/stats/inject)")
    parser.add_argument("--api-host", default=None, help="control API bind address")
    parser.add_argument("--api-port", type=int, default=None, help="control API port")
    parser.add_argument("--api-token", default=None, help="bearer token for the control API (or API_TOKEN)")
    parser.add_argument("--no-persist", action="store_true", help="keep incidents in memory only")
    parser.add_argument("--database", default=None, help="path to the incident database")
    parser.add_argument("--no-restore", action="store_true", help="ignore incidents left open by a previous run")
    parser.add_argument("--seed", type=int, default=None, help="override the RNG seed (reproducible runs)")
    parser.add_argument("--log-level", default=None, help="DEBUG | INFO | WARNING | ERROR")
    parser.add_argument("--summary-json", action="store_true", help="print the final summary as JSON")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    settings = get_settings()
    overrides: dict[str, Any] = {
        "settings": settings,
        "retrain": args.train,
        "dashboard": not args.no_dashboard,
        "inject_anomaly": args.inject_anomaly,
        "auto_retrain": not args.no_retrain_watch,
        "console_notifications": not args.no_notify,
        "seed": args.seed,
        "max_ticks": args.max_ticks,
        "api_enabled": args.api or settings.api_enabled,
    }
    if args.inject_every is not None:
        overrides["inject_every_ticks"] = max(1, args.inject_every)
    if args.api_host is not None:
        overrides["api_host"] = args.api_host
    if args.api_port is not None:
        overrides["api_port"] = args.api_port
    if args.api_token is not None:
        overrides["api_token"] = args.api_token
    if args.no_persist:
        overrides["persistence"] = False
    if args.database is not None:
        overrides["database_path"] = args.database
    if args.no_restore:
        overrides["restore"] = False
    return PipelineConfig(**overrides)


def print_summary(summary: dict[str, Any], as_json: bool = False) -> None:
    if as_json:
        import json

        print(json.dumps(summary, indent=2, default=str))
        return
    incidents = summary["incidents"]
    detector = summary["detector"]
    print("\nSession summary")
    print(f"  cycles run          : {summary['ticks']}")
    print(f"  metrics processed   : {summary['metrics_processed']}")
    print(f"  anomalies detected  : {summary['anomalies_detected']}")
    print(f"  incidents created   : {incidents['total_created']}")
    print(f"  incidents resolved  : {incidents['total_resolved']} (auto: {incidents['total_auto_resolved']})")
    print(f"  active at shutdown  : {incidents['active_count']} {incidents['by_severity']}")
    print(f"  escalations         : {incidents['total_escalated']}")
    print(f"  dedup suppressed    : {summary['triage']['suppressed']}")
    print(f"  model               : {detector['version']} ({detector['training_samples']} samples)")
    print(f"  model anomaly rate  : {float(detector['anomaly_rate']) * 100:.2f}%")
    print(f"  notifications sent  : {summary['notifications']['sent']} (failed {summary['notifications']['failed']})")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    enable_utf8_console()  # Rich draws box lines and sparklines
    setup_logging(args.log_level)
    config = config_from_args(args)

    print("\nAI-Powered Incident Response System")
    print("   multi-cloud | ML anomaly detection | automated triage\n")

    pipeline = Pipeline(config)

    def _stop(signum: int, _frame: object) -> None:
        logger.info("signal %s received - stopping pipeline", signum)
        pipeline.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except (ValueError, OSError):  # pragma: no cover - non-main thread / unsupported
            pass

    try:
        pipeline.prepare()
        server = None
        if config.api_enabled:
            server = ControlServer(
                pipeline,
                host=config.api_host or pipeline.settings.api_host,
                port=config.api_port or pipeline.settings.api_port,
                token=config.api_token or pipeline.settings.api_token,
            ).start()
            if server.auth_required:
                print(f"control API ready at {server.url}  (bearer token required)")
            else:
                print(f"control API ready at {server.url}  (no API_TOKEN set - it is open)")
        summary = pipeline.run()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        logger.info("interrupted")
        summary = pipeline.summary()
    finally:
        if "server" in locals() and server is not None:
            server.stop()

    print_summary(summary, as_json=args.summary_json)
    persistence = summary.get("persistence") or {}
    if persistence.get("enabled"):
        restored = persistence.get("restored_on_start", 0)
        print(
            f"  incident database  : {persistence['path']}"
            + (f"  ({restored} open incident(s) restored)" if restored else "")
        )
    else:
        print("  incident database  : disabled (in-memory only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
