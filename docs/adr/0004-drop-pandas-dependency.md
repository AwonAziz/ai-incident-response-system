# 0004. The data layer stays stdlib-only

* Status: accepted
* Date: 2026-10-04

## Context

`pandas` was in `requirements.txt` from the original scaffold and nothing
imported it. It was added back for exactly one method,
`LabelledSeries.to_frame()`, which returned a `DataFrame` for ad-hoc plotting and
analysis.

Under pytest on Windows/Python 3.13 that method crashed the interpreter with an
access violation inside pandas 3's pyarrow string-array backend
(`pandas/core/arrays/string_arrow.py::_from_sequence`), reproducing through the
test runner but not in a bare interpreter. A convenience accessor that can take
down the process is not a convenience.

## Decision

`src/data/timeseries.py` parses NAB CSVs with `csv` and JSON labels with
`json`, and exposes `to_records()` (plain dicts) instead of `to_frame()`.
`pandas` is removed from the runtime requirements.

## Consequences

* The evaluation path installs and runs without pandas.
* 62,049 real points load in well under a second.
* Data analysis is done in the two places that need it - the benchmark's
  aggregates (pure Python) and ad-hoc exploration (a notebook, where pandas can
  be installed separately).
* If a future feature genuinely needs columnar operations - cross-series joins,
  resampling to a common grid, seasonal decomposition - pandas returns as a
  dependency of *that* module, with the crash investigated first.

## Revisit when

Someone adds a feature that needs real dataframe operations (seasonal
decomposition being the likely candidate). Re-add pandas, pin a version, and
verify the test suite does not crash the interpreter under the pinned version.