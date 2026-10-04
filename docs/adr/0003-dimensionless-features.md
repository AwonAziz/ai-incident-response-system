# 0003. Detection features are dimensionless

* Status: accepted
* Date: 2026-10-04 (revised after first implementation)

## Context

The pipeline mixes percent, milliseconds, bytes per second and requests per
second in one model. The first implementation fed raw `value`, `mean`, `std` and
`delta` into a single `StandardScaler`.

That is broken, and it was visibly broken: an EC2 CPU reading of 97.5 (out of
100) landed *below* an ELB request count of 451 (out of 1100) on the scaled
axis, and the detector scored a real 97.5% CPU spike as *less* anomalous than a
moderate traffic spike. A single scaler across incompatible units destroys the
signal in every column that carries a unit.

## Decision

Every feature is dimensionless by construction:

| Feature | Why it is unit-free |
| --- | --- |
| `z_score` | deviation from the series' own mean, in its own sigma |
| `pct_change` | ratio of two samples of the same metric |
| `value_over_mean` | ratio |
| `coefficient_of_variation` | std / mean |
| `delta_over_std` | step size relative to the series' own volatility |
| `band_position` | position within that metric's expected range, 0-1 inside |
| `history_coverage` | a fraction, never a sample value |

`band_position` is what replaces the raw `value`: "how far outside its own normal
range" is comparable across CPU percent and bytes per second. The band comes
from configuration for the simulator (`cloud_profiles.yaml`) and from **training
quantiles** for real data (`src/data/timeseries.py::derive_bands`), because real
telemetry has no universal normal range and the evaluation split must not
contribute to it.

`history_coverage` is what lets the model see cold-start samples instead of
silently scoring them as steady state, and the detector refuses to flag a sample
whose window is not warm.

## Consequences

* One scaler and one forest are valid across all metrics and clouds.
* Features are interpretable: every column means the same thing for every metric,
  so `feature_report` and the ablation study say something real.
* A cost: absolute magnitude information is only available through `band_position`,
  which is quantile-derived. A metric whose "normal" band happens to be wide will
  look quieter than one whose band is tight.
* The ablation study (README) confirms the trade-off is worth it - dropping
  `coefficient_of_variation` *improves* ranking AUC from 0.616 to 0.631, so that
  feature is on the shortlist for removal.

## Revisit when

A metric arrives whose anomalies are best expressed as an absolute level rather
than a deviation (for example, a saturation ceiling). `band_position` covers that
case if the band is configured from the saturation point rather than a quantile.