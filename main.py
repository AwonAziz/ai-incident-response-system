#!/usr/bin/env python3
"""
AI-Powered Incident Response System
-------------------------------------
Main entry point. Orchestrates the full pipeline:

  1. Load trained ML model
  2. Start metric collectors (AWS, Azure, GCP)
  3. Score each metric with Isolation Forest + Rule Engine
  4. Triage anomalies into prioritized incidents
  5. Route notifications (Console, Slack, PagerDuty)
  6. Render live terminal dashboard

Usage:
    python main.py                    # Normal run (loads existing model)
    python main.py --train            # Re-train model first, then run
    python main.py --no-dashboard     # Pipeline only, no Rich UI
    python main.py --inject-anomaly   # Inject a spike every 30s for demo
"""

import sys
import os
import time
import logging
import argparse
import signal

# ── Logging setup ─────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/system.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")

# ── Project imports ───────────────────────────────────────────────────
from src.ingestion import AWSCollector, AzureCollector, GCPCollector
from src.detection import AnomalyDetector, RuleEngine
from src.triage import TriageEngine, IncidentManager, generate_hints
from src.notifications.notifier import NotificationRouter
from src.dashboard.live_dashboard import LiveDashboard
from config.settings import (
    COLLECTION_INTERVAL_SECONDS,
    DEDUP_WINDOW_SECONDS,
    AUTO_RESOLVE_MINUTES,
    MODEL_PATH,
    SLACK_WEBHOOK_URL,
    PAGERDUTY_ROUTING_KEY,
    QUIET_MODE,
    BASELINE_SAMPLES,
)


# ── Graceful shutdown ─────────────────────────────────────────────────
_running = True

def _handle_signal(sig, frame):
    global _running
    logger.info("Shutdown signal received — stopping pipeline...")
    _running = False

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── Pipeline ─────────────────────────────────────────────────────────

def build_pipeline(args):
    """Instantiate and wire up all pipeline components."""

    # Anomaly detector
    detector = AnomalyDetector(contamination=0.05, n_estimators=150)

    if args.train or not os.path.exists(MODEL_PATH):
        logger.info("Training model on fresh baseline data...")
        from src.ingestion import AWSCollector as A, AzureCollector as Az, GCPCollector as G
        baseline = []
        aws_c = A(inject_anomaly=False)
        az_c  = Az(inject_anomaly=False)
        gcp_c = G(inject_anomaly=False)
        for _ in range(BASELINE_SAMPLES):
            baseline.extend(aws_c.collect())
            baseline.extend(az_c.collect())
            baseline.extend(gcp_c.collect())
        stats = detector.train(baseline)
        logger.info(f"Training complete: {stats}")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        detector.save(MODEL_PATH)
    else:
        logger.info(f"Loading model from {MODEL_PATH}")
        detector.load(MODEL_PATH)

    # Collectors
    collectors = [
        AWSCollector(inject_anomaly=False),
        AzureCollector(inject_anomaly=False),
        GCPCollector(inject_anomaly=False),
    ]
    anomaly_collectors = [
        AWSCollector(inject_anomaly=True),
        AzureCollector(inject_anomaly=True),
        GCPCollector(inject_anomaly=True),
    ]

    rule_engine      = RuleEngine()
    triage_engine    = TriageEngine(dedup_window_seconds=DEDUP_WINDOW_SECONDS)
    incident_manager = IncidentManager()
    router           = NotificationRouter(
        slack_webhook=SLACK_WEBHOOK_URL,
        pagerduty_key=PAGERDUTY_ROUTING_KEY,
        quiet_mode=QUIET_MODE,
    )
    dashboard        = LiveDashboard(incident_manager, detector)

    return {
        "detector":         detector,
        "collectors":       collectors,
        "anomaly_collectors": anomaly_collectors,
        "rule_engine":      rule_engine,
        "triage_engine":    triage_engine,
        "incident_manager": incident_manager,
        "router":           router,
        "dashboard":        dashboard,
    }


def process_tick(components: dict, use_anomaly: bool = False) -> list:
    """Run one collection + detection + triage cycle. Returns new incidents."""
    collectors = components["anomaly_collectors"] if use_anomaly else components["collectors"]
    detector         = components["detector"]
    rule_engine      = components["rule_engine"]
    triage_engine    = components["triage_engine"]
    incident_manager = components["incident_manager"]
    router           = components["router"]
    dashboard        = components["dashboard"]

    all_metrics   = []
    new_incidents = []

    for collector in collectors:
        metrics = collector.collect_and_track()
        scored  = detector.score_batch(metrics)
        all_metrics.extend(scored)

        for metric in scored:
            violations = rule_engine.evaluate(metric)
            hints      = generate_hints(metric)
            incident   = triage_engine.triage(metric, violations, hints)

            if incident:
                incident_manager.add(incident)
                router.notify(incident)
                new_incidents.append(incident)

    # Auto-resolve stale medium/low incidents
    incident_manager.auto_resolve_old(max_age_minutes=AUTO_RESOLVE_MINUTES)

    # Push to dashboard
    dashboard.update_metrics(all_metrics)

    return new_incidents


def run_with_dashboard(components: dict, args):
    """Main loop with live Rich dashboard."""
    from rich.live import Live
    dashboard = components["dashboard"]
    tick = 0

    logger.info("Starting pipeline with live dashboard...")

    with Live(dashboard.get_renderable(), refresh_per_second=2, screen=True) as live:
        while _running:
            tick += 1
            inject = args.inject_anomaly and tick % 6 == 0   # spike every ~30s
            process_tick(components, use_anomaly=inject)
            live.update(dashboard.get_renderable())
            time.sleep(COLLECTION_INTERVAL_SECONDS)

    logger.info("Pipeline stopped.")


def run_headless(components: dict, args):
    """Main loop without dashboard (logs only)."""
    tick = 0
    logger.info("Starting pipeline (headless mode)...")

    while _running:
        tick += 1
        inject = args.inject_anomaly and tick % 6 == 0
        new_incidents = process_tick(components, use_anomaly=inject)
        if new_incidents:
            logger.info(f"Tick {tick}: {len(new_incidents)} new incident(s)")
        time.sleep(COLLECTION_INTERVAL_SECONDS)

    logger.info("Pipeline stopped.")


# ── Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Incident Response System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train",          action="store_true", help="Re-train the model before starting")
    parser.add_argument("--no-dashboard",   action="store_true", help="Run headless (no Rich UI)")
    parser.add_argument("--inject-anomaly", action="store_true", help="Periodically inject test spikes")
    args = parser.parse_args()

    print("\n🛡️  AI-Powered Incident Response System")
    print("   Multi-cloud | ML anomaly detection | Automated triage\n")

    components = build_pipeline(args)

    if args.no_dashboard:
        run_headless(components, args)
    else:
        run_with_dashboard(components, args)

    # Final summary
    stats = components["incident_manager"].stats
    print(f"\n📊 Session Summary:")
    print(f"   Total incidents : {stats['total_created']}")
    print(f"   Resolved        : {stats['total_resolved']}")
    print(f"   Active          : {stats['active_count']}")


if __name__ == "__main__":
    main()
