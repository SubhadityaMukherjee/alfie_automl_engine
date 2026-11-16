"""Unit tests for CO2 emissions tracking (app.core.emissions).

The real codecarbon tracker is replaced by a fake via
``app.core.emissions._load_tracker``; these tests exercise the wrapper
semantics (settings gate, degradation, CSV summarization, payload fields).
"""

import csv
from pathlib import Path

import pytest

from app.core.emissions import (
    EMISSIONS_CSV_FILENAME,
    TrainingEmissionsTracker,
    copy_emissions_csv,
    emissions_extra_fields,
    summarize_emissions_csv,
    summarize_training_emissions,
)

_CSV_HEADER = [
    "timestamp",
    "project_name",
    "run_id",
    "experiment_id",
    "duration",
    "emissions",
    "emissions_rate",
    "cpu_power",
    "gpu_power",
    "ram_power",
    "cpu_energy",
    "gpu_energy",
    "ram_energy",
    "energy_consumed",
    "water_consumed",
    "country_name",
    "country_iso_code",
]


def _write_emissions_csv(path: Path, rows: list[dict]) -> None:
    """Write a codecarbon-shaped emissions.csv with the given rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" for key in _CSV_HEADER} | row)


def _row(**overrides) -> dict:
    base = {
        "duration": "1.5",
        "emissions": "3.2e-07",
        "emissions_rate": "2.1e-07",
        "cpu_power": "12.5",
        "gpu_power": "",
        "ram_power": "3.0",
        "cpu_energy": "4.2e-06",
        "gpu_energy": "",
        "ram_energy": "1.0e-06",
        "energy_consumed": "5.2e-06",
        "country_name": "Denmark",
        "country_iso_code": "DNK",
    }
    return base | overrides


# ---------------------------------------------------------------------------
# _summarize_row (via summarize_emissions_csv)
# ---------------------------------------------------------------------------


def test_summarize_emissions_csv_maps_columns(tmp_path: Path):
    path = tmp_path / EMISSIONS_CSV_FILENAME
    _write_emissions_csv(path, [_row()])

    summary = summarize_emissions_csv(path)
    assert summary == {
        "duration_seconds": 1.5,
        "emissions_kg": 3.2e-07,
        "emissions_rate_kg_per_second": 2.1e-07,
        "cpu_power_watts": 12.5,
        "ram_power_watts": 3.0,
        "cpu_energy_kwh": 4.2e-06,
        "ram_energy_kwh": 1.0e-06,
        "energy_consumed_kwh": 5.2e-06,
        "country_name": "Denmark",
        "country_iso_code": "DNK",
    }


def test_summarize_emissions_csv_returns_last_row(tmp_path: Path):
    path = tmp_path / EMISSIONS_CSV_FILENAME
    # codecarbon appends intermediate rows for long runs; the last one wins.
    _write_emissions_csv(
        path,
        [
            _row(duration="0.5", emissions="1e-07"),
            _row(duration="2.0", emissions="4e-07"),
        ],
    )

    summary = summarize_emissions_csv(path)
    assert summary is not None
    assert summary["duration_seconds"] == 2.0
    assert summary["emissions_kg"] == 4e-07


@pytest.mark.parametrize("content", [None, ""])
def test_summarize_emissions_csv_degrades_to_none(tmp_path: Path, content):
    path = tmp_path / EMISSIONS_CSV_FILENAME
    if content is None:
        pass  # file absent
    else:
        path.write_text(content)

    assert summarize_emissions_csv(path) is None


def test_summarize_row_skips_unparseable_and_non_finite(tmp_path: Path):
    path = tmp_path / EMISSIONS_CSV_FILENAME
    _write_emissions_csv(
        path,
        [_row(duration="not-a-number", emissions="nan", cpu_power="inf")],
    )

    summary = summarize_emissions_csv(path)
    assert summary is not None
    assert "duration_seconds" not in summary
    assert "emissions_kg" not in summary
    assert "cpu_power_watts" not in summary
    # Text fields survive.
    assert summary["country_iso_code"] == "DNK"


# ---------------------------------------------------------------------------
# summarize_training_emissions (settings gate + read-back)
# ---------------------------------------------------------------------------


def test_summarize_training_emissions_disabled_returns_none(tmp_path: Path):
    # The root conftest disables tracking via env; the CSV must be ignored.
    _write_emissions_csv(tmp_path / EMISSIONS_CSV_FILENAME, [_row()])
    assert summarize_training_emissions(tmp_path) is None


def test_summarize_training_emissions_enabled_reads_csv(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EMISSIONS_TRACKING_ENABLED", "true")
    _write_emissions_csv(tmp_path / EMISSIONS_CSV_FILENAME, [_row()])
    summary = summarize_training_emissions(tmp_path)
    assert summary is not None
    assert summary["emissions_kg"] == 3.2e-07


def test_summarize_training_emissions_missing_csv(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EMISSIONS_TRACKING_ENABLED", "true")
    assert summarize_training_emissions(tmp_path) is None


# ---------------------------------------------------------------------------
# emissions_extra_fields
# ---------------------------------------------------------------------------


def test_emissions_extra_fields_flattens_summary():
    summary = {
        "emissions_kg": 3.2e-07,
        "energy_consumed_kwh": 5.2e-06,
        "duration_seconds": 1.5,
        "country_name": "Denmark",
        "country_iso_code": "DNK",
        # Fields below must NOT reach the upload form.
        "emissions_csv": "/tmp/emissions.csv",
    }
    fields = emissions_extra_fields(summary)
    assert fields == {
        "emissions_kg": 3.2e-07,
        "energy_consumed_kwh": 5.2e-06,
        "duration_seconds": 1.5,
        "country_name": "Denmark",
        "country_iso_code": "DNK",
    }


@pytest.mark.parametrize("summary", [None, {}, {"emissions_kg": None}])
def test_emissions_extra_fields_empty_for_missing_data(summary):
    assert emissions_extra_fields(summary) == {}


# ---------------------------------------------------------------------------
# copy_emissions_csv
# ---------------------------------------------------------------------------


def test_copy_emissions_csv_copies_into_model_dir(tmp_path: Path):
    source = tmp_path / "emissions.csv"
    _write_emissions_csv(source, [_row()])
    model_dir = tmp_path / "model"

    copy_emissions_csv(source, model_dir)

    assert (model_dir / EMISSIONS_CSV_FILENAME).exists()


def test_copy_emissions_csv_missing_source_is_silent(tmp_path: Path):
    model_dir = tmp_path / "model"
    copy_emissions_csv(tmp_path / "nope.csv", model_dir)
    assert not model_dir.exists()


# ---------------------------------------------------------------------------
# TrainingEmissionsTracker
# ---------------------------------------------------------------------------


class _FakeCodecarbonTracker:
    """Stand-in for codecarbon's tracker: records calls, writes one CSV row."""

    def __init__(self, project_name=None, output_dir=None, output_file=None, **_):
        self.project_name = project_name
        self.output_dir = output_dir
        self.output_file = output_file
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1
        _write_emissions_csv(
            Path(self.output_dir) / self.output_file,
            [_row(duration="3.0", emissions="9e-07")],
        )
        return 9e-07


@pytest.fixture
def fake_tracker(monkeypatch):
    monkeypatch.setattr(
        "app.core.emissions._load_tracker",
        lambda **kwargs: _FakeCodecarbonTracker(**kwargs),
    )
    monkeypatch.setenv("EMISSIONS_TRACKING_ENABLED", "true")
    return _FakeCodecarbonTracker


def test_tracker_disabled_is_a_noop(tmp_path: Path):
    # Root conftest disables tracking via env; nothing may start.
    tracker = TrainingEmissionsTracker("alfie-automl-test", tmp_path)
    with tracker:
        pass
    assert tracker.summary is None
    assert not (tmp_path / EMISSIONS_CSV_FILENAME).exists()


def test_tracker_wraps_training_and_summarizes(tmp_path: Path, fake_tracker):
    tracker = TrainingEmissionsTracker("alfie-automl-test", tmp_path)
    with tracker as t:
        ran = True  # training would happen here

    assert ran
    assert t.summary is not None
    assert t.summary["duration_seconds"] == 3.0
    assert t.summary["emissions_kg"] == 9e-07
    # CSV persists in the output dir for packaging into the model zip.
    assert (tmp_path / EMISSIONS_CSV_FILENAME).exists()


def test_tracker_stops_on_exception(tmp_path: Path, fake_tracker):
    tracker = TrainingEmissionsTracker("alfie-automl-test", tmp_path)
    with pytest.raises(RuntimeError, match="training failed"):
        with tracker:
            raise RuntimeError("training failed")

    # Even after a failure the tracker stopped and produced what it could.
    assert tracker.summary is not None
    assert tracker.summary["emissions_kg"] == 9e-07


def test_tracker_start_failure_degrades_gracefully(tmp_path: Path, monkeypatch):
    def broken_loader(**kwargs):
        raise RuntimeError("no codecarbon available")

    monkeypatch.setattr("app.core.emissions._load_tracker", broken_loader)
    monkeypatch.setenv("EMISSIONS_TRACKING_ENABLED", "true")

    tracker = TrainingEmissionsTracker("alfie-automl-test", tmp_path)
    with tracker:
        pass
    assert tracker.summary is None


# ---------------------------------------------------------------------------
# Location (country) configuration
# ---------------------------------------------------------------------------


def test_default_country_is_netherlands(monkeypatch):
    """Emissions default to the Netherlands carbon intensity (ALFIE hosting)."""
    monkeypatch.delenv("EMISSIONS_COUNTRY_ISO_CODE", raising=False)
    from app.core.config import get_settings

    assert get_settings().emissions_country_iso_code == "NLD"


def test_tracker_passes_configured_country_to_tracker(tmp_path: Path, monkeypatch):
    """The configured country code reaches codecarbon (empty = autodetect)."""
    received: dict = {}

    def spy_loader(**kwargs):
        received.update(kwargs)
        return _FakeCodecarbonTracker(**kwargs)

    monkeypatch.setattr("app.core.emissions._load_tracker", spy_loader)
    monkeypatch.setenv("EMISSIONS_TRACKING_ENABLED", "true")

    # Default: NLD.
    monkeypatch.delenv("EMISSIONS_COUNTRY_ISO_CODE", raising=False)
    with TrainingEmissionsTracker("alfie-automl-test", tmp_path):
        pass
    assert received["country_iso_code"] == "NLD"

    # Override.
    received.clear()
    monkeypatch.setenv("EMISSIONS_COUNTRY_ISO_CODE", "DNK")
    with TrainingEmissionsTracker("alfie-automl-test", tmp_path):
        pass
    assert received["country_iso_code"] == "DNK"

    # Empty string = autodetect (nothing forwarded).
    received.clear()
    monkeypatch.setenv("EMISSIONS_COUNTRY_ISO_CODE", "")
    with TrainingEmissionsTracker("alfie-automl-test", tmp_path):
        pass
    assert "country_iso_code" not in received


# ---------------------------------------------------------------------------
# coerce_emissions_summary
# ---------------------------------------------------------------------------


def test_coerce_emissions_summary_keeps_dicts():
    from app.core.emissions import coerce_emissions_summary

    summary = {"emissions_kg": 1e-06}
    assert coerce_emissions_summary(summary) is summary


def test_coerce_emissions_summary_drops_non_dicts():
    from app.core.emissions import coerce_emissions_summary

    assert coerce_emissions_summary(None) is None
    assert coerce_emissions_summary("nonsense") is None
    assert coerce_emissions_summary(1e-06) is None
    assert coerce_emissions_summary(object()) is None
