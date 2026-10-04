# 0002. Evaluation protocol: temporal split, training-quantile operating point, event metrics

* Status: accepted
* Date: 2026-10-04

## Context

Unsupervised detectors have no natural "correct" threshold, and time series have
a natural leak. Three decisions follow from that, and each one was wrong in a
first implementation that the benchmark itself exposed.

## Decision

**1. Split once, by position.** Training is everything before the cut, scoring
everything after. `split_series()` never shuffles.

**2. The operating point is a training quantile.** Every detector's threshold is
the `(1 - alert_rate)` quantile of *its own training scores* (2% by default).
Nothing is tuned on the evaluation slice, and all detectors alert at the same
rate by construction, so the comparison is about ranking quality rather than
about who happened to pick a lucky threshold.

*First implementation bug:* the quantile included cold-start samples - the first
few points of every series, whose rolling window is not yet warm. Those samples
are unusual for a reason that has nothing to do with anomalies, so they sat at
the top of the training quantile and suppressed **every** real detection. The
detector already refuses to flag cold samples (`is_anomaly` requires full
history coverage); the operating point now honours the same rule
(`_warm_mask`). This was found by the synthetic tests, not by the real data.

**3. Event metrics alongside point metrics.** NAB windows are 5-39 hours wide -
about 10% of each series - so point-wise precision is structurally capped near
the base rate and recall of 1.0 is achievable by alerting continuously.
Reported for every detector: point precision/recall/F1, *event recall* (was the
window hit at all), *detection delay*, and *false alarms per day* outside
windows.

**4. Rank on a continuous score.** The IsolationForest is ranked on its
`decision_function`, not on `confidence`. Confidence is deliberately coarse -
it is a triage bucket for humans and Slack cards - and its plateaus create ties
that make ROC AUC meaningless.

**5. No point-adjust.** Crediting every point after the first hit inflates
precision and recall for detectors that detect each event once. It is not used.

**6. Report false alarms on normal-only series separately.** The benchmark ships
seven series whose evaluation slice contains no anomaly (including one control
series with no windows at all). They cannot contribute recall, so averaging them
into precision would dilute it - but they are the purest measure of how often an
on-call engineer would have been paged for nothing.

## Consequences

* The headline comparison is defensible: same data, same split, same alert rate,
  different algorithms.
* Numbers are reproducible: the protocol is serialised into every run record
  under `runs/`, and the dataset is pinned by sha256.
* The results include findings that do not flatter the project (the IsolationForest
  ranks best but pages most). That is the point of running them.

## Revisit when

A labelled dataset with tight, unambiguous anomaly intervals arrives, at which
point point-wise precision stops being structurally capped and the event metrics
can be demoted to a secondary role.