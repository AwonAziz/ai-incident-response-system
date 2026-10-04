# Benchmark: real AWS CloudWatch telemetry

Generated 2026-10-04T06:22:05+00:00 (`reproducible`).

## Protocol

* dataset: NAB `realAWSCloudwatch` - 17 series, 62049 points, 30 labelled windows
* temporal split: first 40% trains, remainder is scored
* operating point: threshold = 2.0% quantile of each detector's *training* scores
* proximity credit: 60 min outside a window
* history window: 288 samples, minimum 5
* IsolationForest: contamination 0.05, 150 trees, seed 42

## Results (macro-average over series with an evaluation window)

* `isolation_forest` - IsolationForest (this repo)
* `rolling_zscore` - Rolling z-score
* `ewma` - EWMA control chart
* `global_threshold` - Global mean+k-sigma
* `profile_threshold` - cloud_profiles.yaml warn_high

| detector | event recall | point precision | point recall | point F1 | point precision (strict) | delay (min) | false alarms/day | avg precision | ROC AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EWMA control chart | 0.97 | 0.124 | 0.025 | 0.039 | 0.124 | 158 | 5.30 | 0.117 | 0.476 |
| Global mean+k-sigma | 0.92 | 0.284 | 0.179 | 0.118 | 0.284 | 164 | 7.80 | 0.152 | 0.539 |
| IsolationForest (this repo) | 0.92 | 0.218 | 0.168 | 0.139 | 0.218 | 180 | 10.26 | 0.232 | 0.616 |
| cloud_profiles.yaml warn_high | 0.95 | 0.257 | 0.464 | 0.150 | 0.257 | 118 | 6.56 | 0.144 | 0.529 |
| Rolling z-score | 0.92 | 0.295 | 0.075 | 0.109 | 0.295 | 244 | 3.93 | 0.187 | 0.555 |

## Caveats

* point-wise precision is capped by benchmark window width; judge on-call impact with event recall and false alarms per day
* thresholds come from training quantiles over warm-window samples, never from tuning on the evaluation split
* each model trains on the training slice of the same series: there is no cross-series transfer, so these numbers are a lower bound on what a pooled model would achieve
* NAB's realAWSCloudwatch subset is deliberately hard - published leaderboards show most detectors barely beat random on it
* the model trains per series, on that series' own history; there is no cross-series transfer, so these numbers are a lower bound on what a pooled model would do

## Per-series event recall

| detector | series scored | windows | windows detected |
| --- | --- | --- | --- |
| EWMA control chart | 10 | 15 | 14 |
| Global mean+k-sigma | 10 | 15 | 13 |
| IsolationForest (this repo) | 10 | 15 | 13 |
| cloud_profiles.yaml warn_high | 10 | 15 | 14 |
| Rolling z-score | 10 | 15 | 13 |

## False alarms on normal-only series

The benchmark ships series whose evaluation slice contains no anomaly at all (including one control series with no windows). They cannot contribute recall, but they are the cleanest measure of what an on-call engineer would have been paged for.

| detector | false alarms/day |
| --- | --- |
| EWMA control chart | 8.55 |
| Global mean+k-sigma | 4.47 |
| IsolationForest (this repo) | 11.57 |
| cloud_profiles.yaml warn_high | 1.07 |
| Rolling z-score | 7.02 |

## Series

| series | metric | points | windows |
| --- | --- | --- | --- |
| `ec2_cpu_utilization_24ae8d` | cpu_utilization | 4032 | 2 |
| `ec2_cpu_utilization_53ea38` | cpu_utilization | 4032 | 2 |
| `ec2_cpu_utilization_5f5533` | cpu_utilization | 4032 | 2 |
| `ec2_cpu_utilization_77c1ca` | cpu_utilization | 4032 | 1 |
| `ec2_cpu_utilization_825cc2` | cpu_utilization | 4032 | 1 |
| `ec2_cpu_utilization_ac20cd` | cpu_utilization | 3566 | 1 |
| `ec2_cpu_utilization_c6585a` | cpu_utilization | 4032 | 0 |
| `ec2_cpu_utilization_fe7f93` | cpu_utilization | 4032 | 3 |
| `ec2_disk_write_bytes_1ef3de` | disk_write_bytes | 2118 | 1 |
| `ec2_disk_write_bytes_c0d644` | disk_write_bytes | 4032 | 3 |
| `ec2_network_in_257a54` | network_in_bytes | 4032 | 1 |
| `ec2_network_in_5abac7` | network_in_bytes | 2117 | 2 |
| `elb_request_count_8c0756` | request_count | 4032 | 2 |
| `grok_asg_anomaly` | grok_request_count | 4621 | 3 |
| `iio_us-east-1_i-a2eb1cd9_NetworkIn` | network_in_bytes | 1243 | 2 |
| `rds_cpu_utilization_cc0c53` | cpu_utilization | 4032 | 2 |
| `rds_cpu_utilization_e47b3b` | cpu_utilization | 4032 | 2 |
