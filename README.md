# AI-Powered Incident Response System

An AIOps pipeline that ingests telemetry from three clouds, scores every sample
with an Isolation Forest plus static threshold rules, turns breaches into ranked
incidents with root-cause hypotheses, routes them to notification channels, and
renders the whole thing in a live terminal dashboard. It also exposes a small
HTTP control API so an external process can inspect or inject incidents.

Everything in this repository runs: the collectors simulate their own telemetry
(seeded, so runs are reproducible), and the pipeline, API, dashboard, scripts
and tests are wired together and covered by tests.

```
collect -> score (ML) -> rules -> root cause -> triage -> dedup -> register -> notify -> auto-resolve -> dashboard
```

---

## Contents

- [Quick start](#quick-start)
- [What it actually does](#what-it-actually-does)
- [Does it work on real data?](#does-it-work-on-real-data)
- [What happens when it restarts](#what-happens-when-it-restarts)
- [Architecture](#architecture)
- [Repository map](#repository-map)
- [Detection model](#detection-model)
- [Triage, severity and SLA](#triage-severity-and-sla)
- [Notifications](#notifications)
- [Control API](#control-api)
- [Configuration](#configuration)
- [Docker](#docker)
- [Development](#development)
- [Extending it](#extending-it)
- [Roadmap](#roadmap)
- [License](#license)

---

## Quick start

Requires Python 3.10+ (developed and tested on 3.10-3.13).

```bash
git clone https://github.com/AwonAziz/ai-incident-response-system.git
cd ai-incident-response-system

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# optional: configure thresholds / credentials
cp .env.example .env

python scripts/train_model.py     # train the Isolation Forest (~10s)
python main.py                    # live dashboard
```

Useful variations:

```bash
python main.py --no-dashboard          # headless: logs + console alerts only
python main.py --train                 # retrain before starting
python main.py --inject-anomaly         # degrade a resource every few ticks
python main.py --max-ticks 20           # fixed-length run (CI, demos)
python main.py --api --api-port 8080   # also serve the control API
python main.py --summary-json          # machine readable session summary
```

On first run without a trained model the pipeline trains one automatically, so
`python main.py` works on a fresh clone.

Trigger a one-off incident without the API:

```bash
python scripts/simulate_incident.py --cloud all --count 3
python scripts/simulate_incident.py --metric error_rate --value 7.5 --format text
python scripts/simulate_incident.py --cloud aws --count 2 --push http://127.0.0.1:8080/inject
```

---

## What it actually does

| Capability | Implementation |
| --- | --- |
| Multi-cloud ingestion | Seeded telemetry simulator per cloud (AWS / Azure / GCP), unified `Metric` model |
| Real-data replay | `ReplayCollector` streams labelled real telemetry through the same pipeline |
| Rolling statistics | One shared bounded window per series (`SeriesHistory`) - 30 samples by default |
| Anomaly detection | Isolation Forest over 7 dimensionless features + calibrated confidence |
| Static rules | `cloud_profiles.yaml` thresholds, per-cloud overrides, z-score spike rule |
| Measured evaluation | 5 detectors benchmarked on real AWS data, temporal split, event metrics, ablation |
| Root cause | Ranked hypotheses with evidence, including cross-signal correlation rules |
| Triage | Blended severity scoring, service-level deduplication, escalation, SLA risk |
| Incident lifecycle | Open / acknowledge / resolve / auto-resolve, thread-safe registry, event feed |
| Durability | SQLite write-through store; open incidents and dedup state survive a restart |
| Notifications | Console, Slack, PagerDuty, email, generic webhook with retries, cooldowns, dry runs |
| Dashboard | Rich terminal UI: per-cloud panels, model stats, incident table, lifecycle feed |
| Control API | `/health` `/stats` `/metrics` `/incidents` `/events` `/inject`, bearer-token auth (stdlib only) |
| Drift handling | Population-stability monitor on feature means, automatic retraining |

Telemetry is **simulated**, not real: there are no cloud SDK calls in this
repository. The simulator is deterministic for a given seed, and a real
collector only has to subclass `BaseCollector` and emit `Metric` objects
(see [Extending it](#extending-it)).

---

## Does it work on real data?

A seeded simulator can prove the plumbing but not the approach - a detector and
the data it was tuned on will always agree. So the detectors are also benchmarked
against **real telemetry with published ground truth**: Numenta's Anomaly
Benchmark, `realAWSCloudwatch` subset - 17 real AWS CloudWatch series (EC2/RDS CPU
utilisation, disk write bytes, network in, ELB request count), 62,049 points,
30 labelled anomaly windows.

```bash
python scripts/fetch_nab_dataset.py      # 1.7 MiB, pinned by sha256, nothing committed
python scripts/benchmark.py              # writes docs/evaluation.md + runs/<timestamp>.json
python scripts/benchmark.py --markdown docs/evaluation.md --run-tag reproducible
```

**Protocol** (see [ADR-0002](docs/adr/0002-evaluation-protocol.md)): split once by
position - first 40% trains, the rest is scored, no shuffling. Every detector's
threshold is the 2% quantile of *its own training scores*, so all of them alert at
the same rate by construction and none is tuned on the evaluation slice. The
IsolationForest is ranked on its continuous decision value, not on its deliberately
coarse confidence. Point-adjust is not used.

### Headline result

`history_window=288` (24h at the 5-minute NAB sampling rate), macro-average over
the 10 series that have an anomaly window in their evaluation slice:

| detector | ROC AUC | avg precision | event recall | delay (min) | false alarms/day |
| --- | --- | --- | --- | --- | --- |
| **IsolationForest (this repo)** | **0.616** | **0.232** | 0.92 | 180 | 10.26 |
| Rolling z-score | 0.555 | 0.187 | 0.92 | 244 | 3.93 |
| Global mean+3σ | 0.539 | 0.152 | 0.92 | 164 | 7.80 |
| `cloud_profiles.yaml` `warn_high` | 0.529 | 0.144 | 0.95 | 118 | 6.56 |
| EWMA control chart | 0.476 | 0.117 | 0.97 | 158 | 5.30 |

Three findings worth more than the ranking table:

1. **The detector ranks best and pages most.** Highest AUC and average precision
   of the five, but 10.3 false alarms/day on the normal-only series versus 3.9 for
   a rolling z-score. If you care about alert fatigue, this is *not* a drop-in win,
   and the honest production shape is "IsolationForest as ranker, control-chart
   gating on the top-scoring points".
2. **The hand-written thresholds do not transfer.** The
   `cloud_profiles.yaml` `warn_high` values are tuned for percentages of
   *simulated* telemetry; on real, normalised CloudWatch series they barely fire.
   That baseline is in the table on purpose.
3. **Window length mattered more than the model.** Sweeping the rolling window:

   | history window | IsolationForest AUC | false alarms/day |
   | --- | --- | --- |
   | 30 samples (2.5h) | 0.560 | 4.73 |
   | 96 samples (8h) | 0.579 | 9.51 |
   | 288 samples (24h) | 0.616 | 10.26 |

   The anomalies are multi-hour level shifts; a 2.5-hour window cannot see the
   baseline it is deviating from. Ranking improves with memory and noise rises with
   it - that trade-off, not "IsolationForest good, baselines bad", is the finding.

### Feature ablation (leave one out, `history_window=288`)

| dropped feature | ROC AUC | avg precision | false alarms/day |
| --- | --- | --- | --- |
| *(none - full set)* | 0.616 | 0.232 | 10.26 |
| `z_score` | 0.599 | 0.206 | 11.08 |
| `value_over_mean` | 0.603 | 0.216 | 8.45 |
| `band_position` | 0.614 | 0.229 | 7.85 |
| `history_coverage` | 0.620 | 0.230 | 10.98 |
| `pct_change` | 0.617 | 0.253 | 7.63 |
| `delta_over_std` | 0.621 | 0.254 | 11.17 |
| `coefficient_of_variation` | **0.631** | 0.230 | 9.78 |

`z_score` and `value_over_mean` carry the signal. `coefficient_of_variation` looks
actively harmful on this data, and dropping `pct_change` / `band_position` buys
~25% fewer false alarms at the same ranking quality - a 3-feature model is the
obvious next experiment.

Reproduce any row:

```bash
python scripts/benchmark.py --history-window 288 --ablate z_score --scorers isolation_forest
```

### Caveats, stated plainly

- Point-wise precision is **structurally capped**: NAB's windows are 5-39 hours
  wide, roughly 10% of each series, so a detector that fires continuously scores
  recall 1.0 at precision ~0.1. Event recall and false alarms/day are the metrics
  that survive that.
- The model trains per series on that series' own history. There is no
  cross-series transfer, so these are a lower bound on what a pooled model would
  do.
- `realAWSCloudwatch` is a deliberately hard subset - NAB's own published
  leaderboards show most detectors barely beating random on it. An AUC of 0.62 is
  weak in absolute terms; what makes it useful is that it is measured against
  baselines on the same protocol with the same alert budget.
- Ten of seventeen series contribute to recall. The other seven have no anomaly in
  their evaluation slice and are reported separately as a false-alarm test.

---

```
┌──────────────────────────────────────────────────────────────────┐
│                    MULTI-CLOUD COLLECTORS                        │
│   AWS            Azure              GCP                          │
│   EC2/Lambda     AKS/VM/AppService  GKE/Compute/CloudSQL         │
│   cpu, mem,      cpu, mem, req      cpu, mem, network,           │
│   latency, err,  rate, resp, err    query latency, err          │
│   disk io                                                        │
└───────────────┬──────────────────────────────────────────────────┘
                │  Metric(name, value, unit, cloud, resource, service,
                │         region, window=RollingStats)
                ▼
┌──────────────────────────────────────────────────────────────────┐
│  SHARED ROLLING WINDOWS  (SeriesHistory, per series)             │
│  mean / std / min / max / delta / pct_change / span              │
└───────────────┬──────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────┐
│  DETECTION                                                       │
│  FeatureEngineer  7 dimensionless features + history_coverage    │
│  AnomalyDetector   IsolationForest -> score, confidence, drift   │
│  RuleEngine        warn/critical/band/z-score thresholds         │
└───────────────┬──────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────┐
│  TRIAGE                                                           │
│  RootCause       ranked hypotheses + evidence                    │
│  TriageEngine    score -> severity, dedup window, escalation, SLA │
│  IncidentManager lifecycle, stats, event feed                    │
└───────────────┬──────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────┐
│  NOTIFICATIONS  router -> console | slack | pagerduty | email |   │
│                 webhook   (severity routed, retried, rate limited)│
└───────────────┬──────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────┐
│  SURFACES   Rich live dashboard  |  HTTP control API (:8080)      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Repository map

```
ai-incident-response-system/
├── main.py                     # thin CLI: argparse, signals, summary
├── pyproject.toml               # ruff + pytest + coverage config
├── config/
│   ├── settings.py              # env/.env + YAML loading, Settings dataclass
│   └── cloud_profiles.yaml      # services, metric bands, thresholds, weights
├── src/
│   ├── core/                    # enums (Cloud/Severity/IncidentStatus), Clock, RollingStats
│   ├── data/                    # NAB loader: labelled windows, temporal split, train-derived bands
│   ├── ingestion/               # Metric model, SeriesHistory, BaseCollector, AWS/Azure/GCP, Replay
│   ├── detection/               # FeatureEngineer, AnomalyDetector, RuleEngine, trainer, evaluation
│   ├── triage/                  # root_cause, TriageEngine, IncidentManager
│   ├── notifications/           # notifier (router/console), slack, pagerduty, email, webhook
│   ├── dashboard/               # Rich live dashboard
│   ├── api/                     # stdlib HTTP control plane
│   └── pipeline.py              # orchestration: tick(), run(), inject(), summary()
├── scripts/
│   ├── fetch_nab_dataset.py     # download + sha256 manifest for the real dataset
│   ├── benchmark.py             # detector comparison on real data
│   ├── check_benchmark_thresholds.py  # regression gate for the benchmark
│   ├── train_model.py           # training + labelled evaluation CLI
│   └── simulate_incident.py     # generate/push anomalies
├── docs/
│   ├── evaluation.md            # generated benchmark report
│   └── adr/                     # decisions, with the reasoning
├── tests/                       # 455 tests across 13 modules (95% coverage)
├── models/                      # isolation_forest.pkl (generated, git-ignored)
├── data/                        # fetched dataset + incident database (git-ignored)
├── runs/                        # benchmark run records (generated, git-ignored)
├── Dockerfile / docker-compose.yml
└── .github/workflows/           # ci, nightly benchmark gate, security scanning
```

---

## Detection model

**Features are dimensionless by design.** The pipeline mixes percent,
milliseconds, rps and Mbps in one model, so a raw `value` column would be
meaningless after a single global scaler (97% CPU sits *below* 1100 rps). Every
feature is therefore relative to that series' own history:

| Feature | Meaning |
| --- | --- |
| `z_score` | deviation from the rolling mean, in sigma (clipped at 12) |
| `pct_change` | change vs the previous sample, in percent |
| `value_over_mean` | current value / rolling mean |
| `coefficient_of_variation` | rolling std / mean |
| `delta_over_std` | step size relative to rolling volatility |
| `band_position` | 0-1 inside the metric's expected band, >1 above it |
| `history_coverage` | fraction of `MIN_HISTORY` samples available |

`band_position` is what turns a unit into a signal, and `history_coverage`
stops cold-start samples from being scored as if they were steady state.

**Calibrated confidence.** `IsolationForest.decision_function` output is
interpolated against anchors derived from the *training* distribution, so the
contamination percentile maps to 0.5 and the most anomalous tail saturates at
0.95. The anchors are stored in the model bundle, so a reloaded model scores
exactly like the one that was trained. Confidence is then damped by
`history_coverage`, and a sample whose window is not warm is never flagged.

**Cold-start behaviour.** A series needs `MIN_HISTORY` (default 5) samples
before the ML verdict counts. Rule breaches are unaffected - they are absolute,
not statistical - so nothing is missed during warm-up, only held back.

**Drift watch.** Every scored batch updates `DriftMonitor`, which compares
recent per-batch feature means against the training distribution. When the score
exceeds `DRIFT_THRESHOLD` after `DRIFT_MIN_BATCHES` batches,
`Pipeline` retrains the model (disable with `--no-retrain-watch`).

**Labelled evaluation.** Unsupervised models have no natural accuracy figure, so
this repo refuses to invent one. `--evaluate` generates a fresh anomaly-bearing
batch and labels every sample with the static rule engine (HIGH or CRITICAL
breach = anomaly), then reports precision/recall/F1 against the model's own
predictions:

```bash
$ python scripts/train_model.py --evaluate
  labelled run : precision 0.672 recall 0.983 f1 0.798 over 3080 warm samples
                (344 flagged / 235 expected)
```

Treat this as *agreement with the simulator's own thresholds*, not as
real-world efficacy: the model also flags samples that are out of band but below
the escalation thresholds, which is exactly what the precision figure measures.
The contamination / precision-recall trade-off (same seed, default 500-round
training set) is reproducible:

| `MODEL_CONTAMINATION` | precision | recall | F1 |
| --- | --- | --- | --- |
| **0.02 (default)** | **0.672** | **0.983** | **0.798** |
| 0.03 | 0.537 | 0.983 | 0.695 |
| 0.05 | 0.429 | 0.987 | 0.598 |
| 0.08 | 0.311 | 0.996 | 0.474 |

---

## Triage, severity and SLA

1. **Gate.** An incident is raised only if a rule fired, or the model's
   confidence is at least `ML_MIN_CONFIDENCE`.
2. **Score.** `sum(severity_factor x rule_weight) + z-score bonus + blast radius`,
   where the blast radius counts ML-flagged sibling signals on the same resource
   or service.
3. **Severity** is the *worse* of the score-derived severity and the strongest
   breached rule, so a `critical_high` breach is always CRITICAL.
4. **Deduplication** is per `(cloud, service, metric)` fingerprint. A repeat
   inside `DEDUP_WINDOW` increments the occurrence count; a repeat that is
   *more severe* escalates the existing incident instead of opening a new one; a
   repeat after the window supersedes the stale incident and restarts the SLA
   clock.
5. **SLA risk.** Acknowledgement targets come from
   `sla_targets_minutes` (critical 15m, high 60m, medium 4h, low 24h).
   Estimated time-to-resolve grows with cause uncertainty (low confidence) and
   blast radius; if the estimate exceeds the target the incident is flagged
   `sla_breached` and the notification renders it as AT RISK.
6. **Auto-resolution.** MEDIUM/LOW incidents that go quiet for
   `AUTO_RESOLVE_MINUTES` are resolved automatically. CRITICAL and HIGH are
   never auto-resolved - closing them silently would hide an unacknowledged
   outage.

Root-cause analysis ranks hypotheses from three sources: the breaching metric
itself, correlated signals in the same batch (latency + errors implies a
dependency problem rather than a capacity one), and the rolling statistics
(z-score, ratio to mean, rate of change). A correlation rule only explains a
metric that participates in it, so a disk-IO spike is never blamed on the
latency/errors pair that happens to fire next to it.

---

## Notifications

| Channel | Default severities | Requires |
| --- | --- | --- |
| `console` | all (LOW suppressed by `QUIET_MODE`) | - |
| `slack` | critical, high, medium | `SLACK_WEBHOOK_URL` |
| `pagerduty` | critical, high | `PAGERDUTY_ROUTING_KEY` |
| `email` | critical, high, medium | `EMAIL_TO` + `EMAIL_SMTP_HOST` |
| `webhook` | all | `WEBHOOK_URL` |

Every channel is registered even when it is unconfigured, and reports `skipped`
(dry run) instead of silently vanishing from the routing table. Delivery failures
never propagate into the pipeline: each channel retries with exponential
backoff, honours a per-channel cooldown, and returns a `NotificationResult`.
PagerDuty events are deduplicated on the incident fingerprint, so one open
problem stays one PagerDuty incident.

---

## What happens when it restarts

Incidents, their lifecycle state, their SLA deadlines and the deduplication
registry all survive a restart
([ADR-0005](docs/adr/0005-incident-persistence-and-auth.md)).

```bash
python main.py --no-dashboard --max-ticks 6          # run 1: creates incidents
python main.py --no-dashboard --max-ticks 3          # run 2: restores them
```

```
[INFO] src.pipeline - incident persistence enabled: .../data/incidents.db
[INFO] src.pipeline - restored 32 open incident(s) from .../data/incidents.db
  active at shutdown  : 32 {'LOW': 9, 'MEDIUM': 8, 'HIGH': 6, 'CRITICAL': 9}
  incident database  : .../data/incidents.db  (32 open incident(s) restored)
```

* **SQLite, stdlib only**, WAL mode, one connection per operation - the control
  API serves requests on its own threads.
* Write-through on create / acknowledge / resolve / escalate. A repeated
  observation only bumps a counter, flushed every 10th repeat, so an alert storm
  is a handful of writes rather than one per tick.
* **Only open incidents are restored.** Resolved history stays in the file for the
  record but is not resurrected - a restart should not re-page someone for a
  problem they already dismissed.
* The dedup registry is rebuilt from the restored incidents, so the *same* problem
  after a restart increments its occurrence count instead of raising a duplicate.
* A storage failure is logged and swallowed: losing the audit trail must never
  stop detection.
* `--no-persist`, `--database PATH`, `--no-restore` turn it off or point it
  elsewhere.

---

## Control API

```bash
python main.py --no-dashboard --api        # serves on 0.0.0.0:8080
```

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | liveness, uptime, tick count, active incidents *(public)* |
| GET | `/stats` | summary + incidents + dashboard + model state |
| GET | `/metrics` | per-cloud telemetry summary |
| GET | `/incidents?status=&limit=` | active incidents, or filter by status |
| GET | `/events?limit=` | incident lifecycle feed |
| POST | `/incidents/<id>/ack` | acknowledge |
| POST | `/incidents/<id>/resolve` | resolve |
| POST | `/inject` | push metrics through the full detection path |

**Authentication** is a bearer token (`API_TOKEN`, or `--api-token`), compared with
`hmac.compare_digest`, with `WWW-Authenticate: Bearer` on 401. `/health` is exempt
so the Docker healthcheck needs no credentials. With no token set the API runs open
and logs a warning once - fine on `localhost`, obvious anywhere else. The
rejected-request count is exposed on `/health`, so a client sending the wrong token
is visible rather than mysterious.

`POST /inject` accepts either a compact sample or full metric documents:

```bash
curl -H "Authorization: Bearer $API_TOKEN" \
  -X POST localhost:8080/inject -H 'content-type: application/json' \
  -d '{"cloud":"aws","metric":"cpu_utilization","value":99.0}'

curl -H "Authorization: Bearer $API_TOKEN" \
  -X POST localhost:8080/inject -H 'content-type: application/json' \
  -d '{"metrics":[{"name":"error_rate","value":7.2,"unit":"%","cloud":"gcp",
                   "resource_id":"gke-prod-cluster","service":"GKE",
                   "region":"us-central1","confidence":0.9,"is_anomaly":true}]}'

curl -H "Authorization: Bearer $API_TOKEN" \
  -X POST localhost:8080/inject -H 'content-type: application/json' \
  -d '{"cloud":"azure","metric":"*","count":2}'
```

From the repo, `scripts/simulate_incident.py --push` reads `$API_TOKEN` for you.

It is a stdlib `ThreadingHTTPServer`: no web framework, bounded body size, JSON
errors, and a `handle()` function that is unit-tested without opening a socket.

---

## Configuration

Thresholds and topology live in `config/cloud_profiles.yaml` (per metric: normal
band, warn/critical thresholds, spike z-score, triage weight; per cloud: service
list, metric list, and optional overrides). Everything else is environment based
- copy `.env.example` to `.env`, or export the variables.

| Variable | Default | Effect |
| --- | --- | --- |
| `COLLECTION_INTERVAL` | `5` | seconds between collection cycles |
| `DEDUP_WINDOW` | `60` | seconds before the same fingerprint re-alerts |
| `AUTO_RESOLVE_MINUTES` | `10` | idle time before MEDIUM/LOW auto-resolve |
| `INJECT_EVERY_TICKS` | `6` | injection cadence with `--inject-anomaly` |
| `MODEL_CONTAMINATION` | `0.02` | expected anomaly fraction |
| `MODEL_N_ESTIMATORS` | `150` | forest size |
| `BASELINE_SAMPLES` | `500` | collection rounds used for training |
| `HISTORY_WINDOW` / `MIN_HISTORY` | `30` / `5` | rolling window and warm-up |
| `ML_MIN_CONFIDENCE` | `0.6` | gate for ML-only detections |
| `DRIFT_THRESHOLD` / `DRIFT_MIN_BATCHES` | `0.35` / `10` | retraining trigger |
| `AUTO_RESOLVE_SEVERITIES` | `medium,low` | severities eligible for auto-resolve |
| `QUIET_MODE` | `false` | suppress LOW severity console alerts |
| `RANDOM_SEED` | `42` | reproducible telemetry and training |
| `API_TOKEN` | *(unset)* | bearer token for the control API; unset means open |
| `PERSISTENCE_ENABLED` | `true` | write incidents to SQLite |
| `DATABASE_PATH` | `data/incidents.db` | incident database location |
| `RESTORE_ON_START` | `true` | reload incidents left open by a previous run |

Tuning order of operations when alerts feel noisy: raise `DEDUP_WINDOW`, then
`ML_MIN_CONFIDENCE`, then lower `MODEL_CONTAMINATION`, then tighten the
`warn_high`/`critical_high` values in the YAML.

---

## Docker

```bash
docker compose up -d                       # pipeline + control API on :8080
docker compose --profile demo up           # plus an anomaly injector every 60s
docker compose logs -f incident-response
```

The image trains the model at build time, runs as a non-root user, declares a
`HEALTHCHECK` against `/health`, and persists `models/` and `logs/` through bind
mounts. The demo profile pushes anomalies from a second container through the
control API rather than faking them locally.

---

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt

pytest -q                                   # 455 tests, ~40s
pytest --cov=config --cov=src --cov-report=term-missing
ruff check .                                # lint (config in pyproject.toml)

# benchmark on real data (needs the dataset, fetched once)
python scripts/fetch_nab_dataset.py
python scripts/benchmark.py --markdown docs/evaluation.md
python scripts/check_benchmark_thresholds.py runs/*.json
```

The suite is organised by layer - `test_core`, `test_config`, `test_data`,
`test_ingestion`, `test_detection`, `test_evaluation`, `test_triage`,
`test_store`, `test_benchmark_gate`, `test_notifications`, `test_dashboard`,
`test_pipeline`, `test_api` - and uses a `ManualClock` plus a fixed RNG seed, so
dedup windows, SLA maths, retry backoff, auto-resolution and the evaluation
maths are all tested deterministically and instantly. No test performs real
network I/O, and the benchmark is opt-in rather than part of the test path.

### Continuous integration

| Workflow | Trigger | What it gates |
| --- | --- | --- |
| `ci.yml` | push, PR | ruff, the suite on Python 3.10-3.13 with coverage, then a smoke job that trains a model, runs the pipeline headless and pushes an incident through the control API |
| `benchmark.yml` | nightly 03:17 UTC, manual, path-filtered PRs | re-runs all five detectors on the real dataset and **fails if a committed floor is breached** |
| `security.yml` | push, PR, weekly | CodeQL `security-and-quality` for Python, a gating `pip-audit` on runtime requirements, and a report-only audit of the resolved environment |

The benchmark gate is the interesting one. Unit tests prove the code does what it
says; they cannot tell you the detector still works. So the nightly job
re-measures ROC AUC, average precision, event recall and false-alarm rate
against the floors in `config/benchmark_thresholds.json`, and a feature change
that quietly destroys the signal turns the build red. A floor that *cannot be
evaluated* - a missing metric, a missing detector - fails the gate too, because a
regression gate that skips what it cannot measure is a gate that silently goes
green.

```bash
# reproduce a nightly failure locally
python scripts/benchmark.py --history-window 288 --json
python scripts/check_benchmark_thresholds.py runs/*.json   # exit 1 on regression
```

All third-party actions are pinned to commit SHAs rather than tags, so a moved
tag cannot turn the security workflow into an attack vector.

---

## Extending it

**Add a cloud or a real collector.** Subclass the interface and emit `Metric`
objects; nothing else in the pipeline changes.

```python
from src.core.enums import Cloud
from src.ingestion.base_collector import BaseCollector
from src.ingestion.metric_schema import Metric

class CloudWatchCollector(BaseCollector):
    cloud = Cloud.AWS

    def collect(self) -> list[Metric]:
        return [Metric(name="cpu_utilization", value=42.0, unit="%", cloud=self.cloud,
                       resource_id="i-abc", service="EC2", region="us-east-1")]
```

Register it in `src/ingestion/__init__.py::COLLECTORS` and add its service list
and metric thresholds to `cloud_profiles.yaml`. Uncomment the matching SDK line
in `requirements.txt` (`boto3`, `azure-mgmt-monitor`, `google-cloud-monitoring`).

**Add a threshold rule.** Add the metric band to `cloud_profiles.yaml`; the rule
engine compiles `warn_high`, `critical_high`, `outside_normal_band` and
`zscore_spike` automatically.

**Add a notification channel.** Subclass `BaseNotifier`, implement `_deliver`,
and register it in `NotificationRouter`. Retries, cooldowns, severity routing and
statistics come for free.

---

## Roadmap

- [ ] Real cloud SDK collectors (CloudWatch / Azure Monitor / Cloud Monitoring)
- [ ] Pooled cross-series model instead of per-series training, with per-metric heads
- [ ] Use the ablation result: 3-feature model (`z_score`, `value_over_mean`, `band_position`)
      and control-chart gating to cut false alarms
- [ ] Seasonal features (daily/weekly decomposition) - NAB anomalies are multi-hour
      level shifts and none of the current features model a daily cycle
- [ ] Predictive maintenance: per-resource failure-risk score from degradation trends
- [ ] Incident grouping by root cause across clouds, not just per service+metric
- [ ] Prometheus exposition endpoint and a load benchmark (p95 tick latency at 10k series)
- [ ] Rule authoring CLI with a shadow-mode comparison run

---

## License

MIT - see [LICENSE](LICENSE). Built by [Awon Aziz](https://github.com/AwonAziz).
