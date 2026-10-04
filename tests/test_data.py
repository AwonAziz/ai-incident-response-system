"""Real-dataset layer: parsing, labelling, temporal splitting, band derivation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.data.timeseries import (
    ANOMALOUS,
    AnomalyWindow,
    LabelledSeries,
    derive_bands,
    load_nab_dataset,
    load_nab_windows,
    load_series,
    metric_name_for,
    parse_timestamp,
    split_time,
    unit_for,
)


def _csv(path: Path, rows: list[tuple[str, str]]) -> Path:
    lines = ["timestamp,value", *(f"{stamp},{value}" for stamp, value in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _series(values: list[float], labels: list[int], *, step: int = 300, start: str = "2014-01-01 00:00:00") -> LabelledSeries:
    first = parse_timestamp(start)
    stamps = tuple(first + timedelta(seconds=step * index) for index in range(len(values)))
    return LabelledSeries(
        name="synthetic",
        metric="cpu_utilization",
        timestamps=stamps,
        values=tuple(values),
        labels=tuple(labels),
        resource_id="cpu_utilization/synthetic",
    )


class TestParseTimestamp:
    @pytest.mark.parametrize(
        "value",
        ["2014-02-14 14:30:00", "2014-02-14T14:30:00", "2014-02-14 14:30:00.500000"],
    )
    def test_accepts_nab_formats(self, value: str) -> None:
        parsed = parse_timestamp(value)
        assert parsed.tzinfo is timezone.utc
        assert (parsed.year, parsed.month, parsed.day) == (2014, 2, 14)

    def test_rejects_garbage(self) -> None:
        with pytest.raises(ValueError, match="unrecognised timestamp"):
            parse_timestamp("not a timestamp")


class TestAnomalyWindow:
    def test_rejects_inverted_window(self) -> None:
        start = datetime(2014, 1, 2, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="ends before"):
            AnomalyWindow(start=start, end=start - timedelta(hours=1))

    def test_contains_with_padding(self) -> None:
        start = datetime(2014, 1, 1, tzinfo=timezone.utc)
        window = AnomalyWindow(start=start, end=start + timedelta(hours=1))
        assert window.contains(start)
        assert window.contains(start + timedelta(hours=1))
        assert not window.contains(start - timedelta(minutes=1))
        assert window.contains(start - timedelta(minutes=30), pad=timedelta(minutes=45))
        assert not window.contains(start - timedelta(minutes=30))

    def test_duration_and_serialisation(self) -> None:
        start = datetime(2014, 1, 1, tzinfo=timezone.utc)
        window = AnomalyWindow(start=start, end=start + timedelta(hours=2))
        assert window.duration == timedelta(hours=2)
        assert AnomalyWindow.from_strings(*window.to_dict().values()) == window


class TestMetricNames:
    @pytest.mark.parametrize(
        ("stem", "expected"),
        [
            ("ec2_cpu_utilization_24ae8d", "cpu_utilization"),
            ("rds_cpu_utilization_cc0c53", "cpu_utilization"),
            ("ec2_disk_write_bytes_1ef3de", "disk_write_bytes"),
            ("elb_request_count_8c0756", "request_count"),
            ("grok_asg_anomaly", "grok_request_count"),
            ("iio_us-east-1_i-a2eb1cd9_NetworkIn", "network_in_bytes"),
            ("something_unknown", "something_unknown"),
        ],
    )
    def test_mapping(self, stem: str, expected: str) -> None:
        assert metric_name_for(stem) == expected

    def test_units(self) -> None:
        assert unit_for("cpu_utilization") == "%"
        assert unit_for("unknown_metric") == ""


class TestLoadSeries:
    def test_labels_points_inside_windows(self, tmp_path: Path) -> None:
        rows = [(f"2014-01-01 00:{minute:02d}:00", str(minute)) for minute in range(10)]
        path = _csv(tmp_path / "ec2_cpu_utilization_test.csv", rows)
        window = AnomalyWindow.from_strings("2014-01-01 00:04:00", "2014-01-01 00:05:00")

        series = load_series(path, [window])
        assert series.size == 10
        assert series.metric == "cpu_utilization"
        assert series.unit == "%"
        assert series.resource_id.endswith("ec2_cpu_utilization_test")
        assert series.dropped_points == 0
        assert [label for label in series.labels if label == ANOMALOUS] == [ANOMALOUS, ANOMALOUS]
        assert series.positive_labels == 2

    def test_missing_values_are_dropped_and_counted(self, tmp_path: Path) -> None:
        rows = [
            ("2014-01-01 00:00:00", "1.0"),
            ("2014-01-01 00:05:00", ""),
            ("2014-01-01 00:10:00", "3.0"),
        ]
        path = _csv(tmp_path / "rds_cpu_utilization_x.csv", rows)
        series = load_series(path)
        assert series.size == 2
        assert series.dropped_points == 1
        assert series.values == (1.0, 3.0)

    def test_nan_values_are_dropped(self, tmp_path: Path) -> None:
        path = _csv(
            tmp_path / "ec2_network_in_x.csv",
            [("2014-01-01 00:00:00", "NaN"), ("2014-01-01 00:05:00", "5")],
        )
        series = load_series(path)
        assert series.values == (5.0,)
        assert series.dropped_points == 1

    def test_large_gaps_are_dropped_without_poisoning_the_step_estimate(self, tmp_path: Path) -> None:
        rows = [(f"2014-01-01 00:{minute:02d}:00", "1.0") for minute in range(0, 60, 5)]
        rows.append(("2014-01-02 00:00:00", "9.0"))  # 24h hole
        path = _csv(tmp_path / "ec2_cpu_utilization_gap.csv", rows)
        series = load_series(path, max_gap_multiple=3)
        assert series.size == len(rows) - 1
        assert series.dropped_points == 1
        assert series.step_seconds == 300.0

    def test_gap_filter_is_off_by_default(self, tmp_path: Path) -> None:
        rows = [("2014-01-01 00:00:00", "1.0"), ("2014-01-02 00:00:00", "9.0")]
        path = _csv(tmp_path / "ec2_cpu_utilization_nogap.csv", rows)
        assert load_series(path).size == 2

    def test_missing_columns_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "ec2_cpu_utilization_bad.csv"
        path.write_text("time,val\n2014-01-01 00:00:00,1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing column"):
            load_series(path)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = _csv(tmp_path / "ec2_cpu_utilization_empty.csv", [])
        with pytest.raises(ValueError, match="no usable rows"):
            load_series(path)


class TestSplitTime:
    def test_split_is_temporal_and_disjoint(self) -> None:
        series = _series(list(range(100)), [0] * 100)
        train, evaluation = split_time(series, 0.6)
        assert train.size == 60
        assert evaluation.size == 40
        assert train.timestamps[-1] < evaluation.timestamps[0]
        assert train.values[-1] == 59
        assert evaluation.values[0] == 60
        assert train.size + evaluation.size == series.size

    def test_windows_are_kept_only_when_fully_inside(self) -> None:
        # 100 points at 5-minute spacing: train covers 0-294 min, evaluation 300-495 min
        early = AnomalyWindow.from_strings("2014-01-01 00:10:00", "2014-01-01 00:20:00")
        late = AnomalyWindow.from_strings("2014-01-01 06:00:00", "2014-01-01 06:10:00")
        straddling = AnomalyWindow.from_strings("2014-01-01 04:30:00", "2014-01-01 06:30:00")
        series = LabelledSeries(
            name="synthetic",
            metric="cpu_utilization",
            timestamps=_series(list(range(100)), [0] * 100).timestamps,
            values=tuple(float(index) for index in range(100)),
            labels=(0,) * 100,
            windows=(early, late, straddling),
        )
        train, evaluation = split_time(series, 0.6)
        # the straddling window spans the cut, so neither slice keeps it
        assert train.window_count == 1
        assert evaluation.window_count == 1
        assert train.windows[0] == early
        assert evaluation.windows[0] == late

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 1.5])
    def test_invalid_fraction(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="train_fraction"):
            split_time(_series([1.0] * 10, [0] * 10), fraction)


class TestDeriveBands:
    def test_uses_percentiles(self) -> None:
        values = [float(index) for index in range(1000)]
        series = _series(values, [0] * 1000)
        bands = derive_bands([series])
        low, high = bands["cpu_utilization"]
        assert low == pytest.approx(9.99, abs=1)
        assert high == pytest.approx(989.0, abs=1)

    def test_constant_series_is_widened(self) -> None:
        series = _series([5.0] * 50, [0] * 50)
        low, high = derive_bands([series])["cpu_utilization"]
        assert high > low

    def test_bands_are_metric_keyed(self) -> None:
        a = _series([1.0, 2.0], [0, 0])
        b = LabelledSeries(
            name="other",
            metric="network_in_bytes",
            timestamps=a.timestamps,
            values=a.values,
            labels=a.labels,
        )
        assert set(derive_bands([a, b])) == {"cpu_utilization", "network_in_bytes"}


class TestLabelledSeries:
    def test_ragged_series_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ragged"):
            LabelledSeries(
                name="x",
                metric="cpu_utilization",
                timestamps=(datetime.now(timezone.utc),),
                values=(1.0, 2.0),
                labels=(0,),
            )

    def test_iteration_and_length(self) -> None:
        series = _series([1.0, 2.0, 3.0], [0, 1, 0])
        assert len(series) == 3
        assert [label for _, _, label in series] == [0, 1, 0]

    def test_span_step_and_windows(self) -> None:
        series = _series([1.0, 2.0, 3.0], [0, 0, 0])
        assert series.step == timedelta(seconds=300)
        assert series.step_seconds == 300.0
        assert series.span == timedelta(seconds=600)
        assert not series.is_labelled
        assert series.window_count == 0

    def test_slice_keeps_order_and_windows(self) -> None:
        series = _series([float(index) for index in range(10)], [0] * 10)
        assert series.slice(2, 5).values == (2.0, 3.0, 4.0)
        assert series.slice(5).size == 5
        assert series.slice(0, 100).size == 10

    def test_relabel_applies_padding(self) -> None:
        start = parse_timestamp("2014-01-01 00:00:00")
        stamps = tuple(start + timedelta(minutes=5 * index) for index in range(6))
        window = AnomalyWindow(start=stamps[2], end=stamps[3])
        series = LabelledSeries(
            name="x",
            metric="cpu_utilization",
            timestamps=stamps,
            values=tuple(float(index) for index in range(6)),
            labels=(0,) * 6,
            windows=(window,),
        )
        assert series.relabel().positive_labels == 2
        assert series.relabel(pad=timedelta(minutes=10)).positive_labels == 6

    def test_to_records_and_summary(self) -> None:
        series = _series([1.0, 2.0], [0, 1])
        records = series.to_records()
        assert records[0]["value"] == 1.0
        assert records[1]["label"] == 1
        assert records[0]["timestamp"].startswith("2014-01-01")
        json.dumps(records)
        info = series.summary()
        assert info["points"] == 2
        assert info["anomalous_points"] == 1
        assert info["anomalous_fraction"] == 0.5
        json.dumps(info)  # must stay JSON serialisable


class TestNabDataset:
    def _fixtures(self, tmp_path: Path) -> tuple[Path, Path]:
        directory = tmp_path / "realAWSCloudwatch"
        directory.mkdir()
        rows = [(f"2014-01-01 00:{minute:02d}:00", "1.0") for minute in range(30)]
        _csv(directory / "ec2_cpu_utilization_a.csv", rows)
        _csv(directory / "ec2_cpu_utilization_control.csv", rows)
        windows = tmp_path / "combined_windows.json"
        windows.write_text(
            json.dumps(
                {
                    "realAWSCloudwatch/ec2_cpu_utilization_a.csv": [
                        ["2014-01-01 00:05:00", "2014-01-01 00:10:00"]
                    ],
                    "realKnownCause/x.csv": [["2014-01-01 00:00:00", "2014-01-01 00:01:00"]],
                }
            ),
            encoding="utf-8",
        )
        return directory, windows

    def test_loads_dataset_and_filters_other_datasets(self, tmp_path: Path) -> None:
        directory, windows = self._fixtures(tmp_path)
        series = load_nab_dataset(directory, windows)
        assert {item.name for item in series} == {
            "ec2_cpu_utilization_a",
            "ec2_cpu_utilization_control",
        }

    def test_unlabelled_series_are_kept_and_flagged(self, tmp_path: Path) -> None:
        directory, windows = self._fixtures(tmp_path)
        series = {item.name: item for item in load_nab_dataset(directory, windows)}
        assert series["ec2_cpu_utilization_control"].is_labelled is False
        assert series["ec2_cpu_utilization_a"].is_labelled is True

    def test_unlabelled_series_can_be_excluded(self, tmp_path: Path) -> None:
        directory, windows = self._fixtures(tmp_path)
        series = load_nab_dataset(directory, windows, include_unlabelled=False)
        assert [item.name for item in series] == ["ec2_cpu_utilization_a"]

    def test_windows_loader_ignores_other_datasets(self, tmp_path: Path) -> None:
        _, windows = self._fixtures(tmp_path)
        loaded = load_nab_windows(windows)
        assert set(loaded) == {"ec2_cpu_utilization_a"}

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        _, windows = self._fixtures(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_nab_dataset(tmp_path / "absent", windows)
