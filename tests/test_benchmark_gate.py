"""The benchmark regression gate.

A gate that cannot go red is worse than no gate, so most of these tests are
about failure: a floor that is breached, a metric that vanished, a detector that
vanished, a run produced under different conditions, and a run file that is
missing entirely.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_benchmark_thresholds import (
    Check,
    Unverifiable,
    compare,
    load_json,
    load_thresholds,
    main,
    newest_run,
    render,
)

THRESHOLDS = {
    "protocol": {"history_window": 288, "train_fraction": 0.4, "alert_rate": 0.02, "contamination": 0.05},
    "detectors": {
        "isolation_forest": {"roc_auc_min": 0.55, "event_recall_min": 0.8},
        "ewma": {"roc_auc_min": 0.42},
    },
    "max_false_alarms_per_day": 20.0,
}

GOOD_RUN = {
    "protocol": dict(THRESHOLDS["protocol"]),
    "summary": {
        "isolation_forest": {
            "roc_auc": 0.616,
            "event_recall": 0.92,
            "normal_only_series": 7,
            "false_alarms_per_day_normal_only": 11.5,
        },
        "ewma": {
            "roc_auc": 0.476,
            "normal_only_series": 7,
            "false_alarms_per_day_normal_only": 8.5,
        },
    },
}


def run(**overrides) -> dict:
    payload = json.loads(json.dumps(GOOD_RUN))
    payload.update(overrides)
    return payload


class TestCheck:
    def test_greater_or_equal(self) -> None:
        assert Check("d", "m", 0.6, 0.55, ">=").passed
        assert not Check("d", "m", 0.5, 0.55, ">=").passed

    def test_less_or_equal(self) -> None:
        assert Check("d", "m", 9.0, 20.0, "<=").passed
        assert not Check("d", "m", 21.0, 20.0, "<=").passed

    def test_describe_and_serialise(self) -> None:
        check = Check("isolation_forest", "roc_auc", 0.616, 0.55, ">=")
        assert "PASS" in check.describe()
        assert check.to_dict()["passed"] is True
        assert "FAIL" in Check("d", "m", 0.1, 0.5, ">=").describe()


class TestUnverifiable:
    def test_is_a_failure_not_a_note(self) -> None:
        result = compare(run(summary={}), THRESHOLDS)
        assert result.ok is False
        assert result.unverifiable

    def test_describes_itself(self) -> None:
        assert "could not be evaluated" in Unverifiable("d", "m", "missing").describe()


class TestComparePasses:
    def test_healthy_run_passes_every_floor(self) -> None:
        result = compare(run(), THRESHOLDS)
        assert result.ok is True
        assert result.failures == []
        assert {check.metric for check in result.checks} >= {
            "roc_auc",
            "event_recall",
            "false_alarms_per_day_normal_only",
        }

    def test_min_suffix_maps_onto_the_metric_name(self) -> None:
        """Floors are written `roc_auc_min`; the run reports `roc_auc`."""
        result = compare(run(), THRESHOLDS)
        assert all(not check.metric.endswith("_min") for check in result.checks)

    def test_false_alarm_check_uses_the_inverted_comparison(self) -> None:
        result = compare(run(), THRESHOLDS)
        check = next(item for item in result.checks if item.metric == "false_alarms_per_day_normal_only")
        assert check.comparison == "<="


class TestCompareFails:
    def test_metric_below_floor(self) -> None:
        payload = run()
        payload["summary"]["isolation_forest"]["roc_auc"] = 0.41
        result = compare(payload, THRESHOLDS)
        assert result.ok is False
        assert [check.metric for check in result.failures] == ["roc_auc"]

    def test_missing_metric_is_unmeasurable_not_passing(self) -> None:
        payload = run()
        del payload["summary"]["isolation_forest"]["event_recall"]
        result = compare(payload, THRESHOLDS)
        assert result.ok is False
        assert any(item.metric == "event_recall" for item in result.unverifiable)

    def test_missing_detector_is_unmeasurable(self) -> None:
        payload = run()
        del payload["summary"]["ewma"]
        result = compare(payload, THRESHOLDS)
        assert result.ok is False
        assert any(item.detector == "ewma" for item in result.unverifiable)

    def test_non_numeric_metric_is_unmeasurable(self) -> None:
        payload = run()
        payload["summary"]["isolation_forest"]["roc_auc"] = "n/a"
        result = compare(payload, THRESHOLDS)
        assert result.ok is False
        assert any(item.metric == "roc_auc" for item in result.unverifiable)

    def test_empty_summary_fails(self) -> None:
        result = compare(run(summary={}), THRESHOLDS)
        assert result.ok is False
        assert "summary" in result.notes[0]

    def test_excess_false_alarms_fails(self) -> None:
        payload = run()
        payload["summary"]["isolation_forest"]["false_alarms_per_day_normal_only"] = 45.0
        result = compare(payload, THRESHOLDS)
        assert result.ok is False
        assert any(check.metric == "false_alarms_per_day_normal_only" for check in result.failures)

    def test_no_normal_only_series_skips_rather_than_invents(self) -> None:
        payload = run()
        payload["summary"]["ewma"]["normal_only_series"] = 0
        result = compare(payload, THRESHOLDS)
        assert result.ok is True
        assert not any(
            check.detector == "ewma" and check.metric == "false_alarms_per_day_normal_only" for check in result.checks
        )
        assert any("no normal-only series" in note for note in result.notes)


class TestProtocolDrift:
    def test_drift_is_flagged_but_does_not_by_itself_fail(self) -> None:
        payload = run(protocol=dict(THRESHOLDS["protocol"], train_fraction=0.6))
        result = compare(payload, THRESHOLDS)
        assert any("protocol drift" in note for note in result.notes)

    def test_matching_protocol_produces_no_note(self) -> None:
        assert not compare(run(), THRESHOLDS).notes


class TestIo:
    def test_newest_run_is_last_sorted(self, tmp_path: Path) -> None:
        first = tmp_path / "run-1.json"
        second = tmp_path / "run-2.json"
        first.write_text("{}", encoding="utf-8")
        second.write_text("{}", encoding="utf-8")
        assert newest_run([str(first), str(second)]) == second

    def test_newest_run_skips_missing_files(self, tmp_path: Path) -> None:
        only = tmp_path / "run-1.json"
        only.write_text("{}", encoding="utf-8")
        assert newest_run([str(only), str(tmp_path / "absent.json")]) == only

    def test_newest_run_raises_when_nothing_exists(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            newest_run([str(tmp_path / "absent.json")])

    def test_byte_order_mark_is_tolerated(self, tmp_path: Path) -> None:
        path = tmp_path / "run.json"
        path.write_bytes(b"\xef\xbb\xbf" + json.dumps(GOOD_RUN).encode())
        assert load_json(path)["summary"]["isolation_forest"]["roc_auc"] == 0.616

    def test_committed_thresholds_are_valid(self) -> None:
        floors = load_thresholds()
        assert "isolation_forest" in floors["detectors"]
        assert floors["protocol"]["history_window"] == 288

    def test_thresholds_without_detectors_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"protocol": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="detectors"):
            load_thresholds(path)


class TestRender:
    def test_pass_text(self) -> None:
        assert "PASS" in render(compare(run(), THRESHOLDS))

    def test_fail_text_names_the_metric(self) -> None:
        payload = run()
        payload["summary"]["ewma"]["roc_auc"] = 0.1
        text = render(compare(payload, THRESHOLDS))
        assert "FAIL" in text
        assert "ewma.roc_auc" in text


class TestCli:
    def _write(self, tmp_path: Path, payload: dict) -> Path:
        path = tmp_path / "run.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _thresholds(self, tmp_path: Path, payload: dict) -> Path:
        path = tmp_path / "thresholds.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_exit_zero_when_healthy(self, tmp_path: Path, capsys) -> None:
        run_file = self._write(tmp_path, GOOD_RUN)
        floors = self._thresholds(tmp_path, THRESHOLDS)
        assert main([str(run_file), "--thresholds", str(floors)]) == 0
        assert "PASS" in capsys.readouterr().out

    def test_exit_one_on_regression(self, tmp_path: Path, capsys) -> None:
        payload = run()
        payload["summary"]["isolation_forest"]["roc_auc"] = 0.2
        run_file = self._write(tmp_path, payload)
        floors = self._thresholds(tmp_path, THRESHOLDS)
        assert main([str(run_file), "--thresholds", str(floors)]) == 1
        assert "FAIL" in capsys.readouterr().out

    def test_exit_two_when_no_run_file(self, tmp_path: Path, capsys) -> None:
        floors = self._thresholds(tmp_path, THRESHOLDS)
        code = main([str(tmp_path / "absent.json"), "--thresholds", str(floors)])
        assert code == 2
        assert "could not evaluate" in capsys.readouterr().err

    def test_json_output(self, tmp_path: Path, capsys) -> None:
        run_file = self._write(tmp_path, GOOD_RUN)
        floors = self._thresholds(tmp_path, THRESHOLDS)
        main([str(run_file), "--thresholds", str(floors), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert payload["checks"]
