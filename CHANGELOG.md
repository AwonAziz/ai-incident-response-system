# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project is not
yet versioned past `1.0.0`.

## [1.0.0] - 2026-10-04

The first version in which the documented system is the implemented system.

### Added

**Pipeline** - a multi-cloud AIOps pipeline: collect, score, apply rules, rank
root causes, triage, deduplicate, notify, auto-resolve, render.

- Unified `Metric` model with JSON round-trip, and a `SeriesHistory` that owns
  the bounded per-series rolling window for the whole pipeline.
- Seeded telemetry simulators for AWS, Azure and GCP, plus a `ReplayCollector`
  that streams real labelled telemetry through the same code path.
- Isolation Forest detector over seven dimensionless features, with confidence
  interpolated from the training decision distribution and carried in the model
  bundle; static threshold rules compiled from `cloud_profiles.yaml` with
  per-cloud overrides.
- Root-cause hypotheses with evidence, including cross-signal correlation rules.
- Triage with blended severity scoring, service-level deduplication, escalation
  and SLA-risk estimation.
- Thread-safe incident lifecycle: open, acknowledge, resolve, auto-resolve,
  event feed.
- Notification channels for console, Slack, PagerDuty, email and generic
  webhook, with severity routing, retries, cooldowns and visible dry runs.
- Rich terminal dashboard, and a stdlib HTTP control API
  (`/health /stats /metrics /incidents /events /inject`) with bearer-token auth.
- SQLite write-through incident store; open incidents and the deduplication
  registry are restored on restart.
- Model drift monitor with automatic retraining.

**Evaluation** - five detectors benchmarked on real AWS CloudWatch telemetry
(NAB `realAWSCloudwatch`, 17 series, 62,049 points) under one protocol: temporal
split, operating points from training quantiles, point and event metrics, and a
leave-one-out feature ablation. `docs/evaluation.md` holds the generated report;
`docs/adr/` records the reasoning behind each decision.

**Project** - 426 tests at 95% branch coverage, ruff-clean, CI across Python
3.10-3.13 including an end-to-end smoke run, a non-root container image with a
healthcheck, and a README that reports measured numbers with their caveats
attached.

### Fixed

Found while building the above, recorded because each one changed the
implementation rather than just the tests:

- A shared `SeriesHistory` was being forked silently: an empty window registry
  is falsy under `__len__`, so `history or SeriesHistory(...)` handed each
  collector its own window and the training and serving feature vectors
  disagreed.
- Raw `value`/`mean`/`std` features were meaningless after a single global
  scaler across percent, milliseconds and bytes per second - a 97.5% CPU spike
  scored *below* a 451 rps request count. Replaced with dimensionless features.
- The confidence calibration used a sigmoid over the standardised decision
  value, which mapped a real anomaly to 0.49. Replaced with interpolation from
  the training distribution.
- The alert threshold was taken from a training quantile that included
  cold-start samples, so the artefacts of a cold window suppressed every real
  detection. The operating point now uses warm samples only, matching the rule
  the detector already applies.
- The dedup registry listener had the wrong signature for the manager's event
  hook, and the exception was swallowed - so deduplication state was never
  cleared when an incident closed.
- `NaN` values in the real dataset parsed happily and poisoned every statistic;
  non-finite values are now treated as missing data.
- `AnomalyWindow` could be serialised but not parsed back, so any route that
  returned a window could not be reloaded.
- An oversized API body was rejected before the request was drained, so the
  client saw a connection reset instead of a 413.

### Removed

- `pandas`: the single accessor that used it could crash the interpreter inside
  the pyarrow string backend. The data layer parses CSV and JSON with the
  standard library.