"""CO2 emissions tracking for training runs (codecarbon).

Wraps ``codecarbon.EmissionsTracker`` so every modality's training records
its energy use and CO2eq emissions:

* ``run_optuna_search`` tracks the whole Optuna search (all trials) for the
  generic engine (vision / audio / text tasks).
* ``AutoGluonTrainer.train`` tracks tabular training.

The tracker appends a row to ``emissions.csv`` in the training output
directory — that folder is packaged into the uploaded model zip, so the CSV
reaches AutoDW alongside the model. A JSON-safe summary dict built from the
final CSV row flows into the API response (``emissions`` field) and onto the
AutoDW upload form (``build_upload_payload(..., emissions=...)``).

Tracking is best-effort by design: a missing codecarbon install, a failed
measurement, or an unreadable CSV degrades to ``None`` and never fails a
training run or an upload.
"""

from __future__ import annotations

import csv
import logging
import math
import shutil
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

EMISSIONS_CSV_FILENAME = "emissions.csv"

# (emissions.csv column, summary key) for the numeric fields we surface.
# Units follow codecarbon: emissions in kg CO2eq, power in watts, energy
# in kWh, duration in seconds.
_NUMERIC_SUMMARY_FIELDS: tuple[tuple[str, str], ...] = (
    ("duration", "duration_seconds"),
    ("emissions", "emissions_kg"),
    ("emissions_rate", "emissions_rate_kg_per_second"),
    ("cpu_power", "cpu_power_watts"),
    ("gpu_power", "gpu_power_watts"),
    ("ram_power", "ram_power_watts"),
    ("cpu_energy", "cpu_energy_kwh"),
    ("gpu_energy", "gpu_energy_kwh"),
    ("ram_energy", "ram_energy_kwh"),
    ("energy_consumed", "energy_consumed_kwh"),
)

# Textual fields kept verbatim in the summary.
_TEXT_SUMMARY_FIELDS: tuple[tuple[str, str], ...] = (
    ("country_name", "country_name"),
    ("country_iso_code", "country_iso_code"),
)


def _summarize_row(row: dict[str, str]) -> dict[str, Any]:
    """Turn one ``emissions.csv`` row into a JSON-safe summary dict.

    Missing, unparseable, and non-finite numeric fields are omitted rather
    than reported as zero.
    """
    summary: dict[str, Any] = {}
    for column, key in _NUMERIC_SUMMARY_FIELDS:
        raw = row.get(column)
        if raw is None or raw == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        # Trim float noise; keeps tiny emissions values (1e-07 range) intact.
        summary[key] = float(f"{value:.12g}")
    for column, key in _TEXT_SUMMARY_FIELDS:
        raw = row.get(column)
        if raw:
            summary[key] = str(raw)
    return summary


def summarize_emissions_csv(path: Path) -> dict[str, Any] | None:
    """Summarize the most recent row of a codecarbon ``emissions.csv``.

    The tracker appends one row per run (plus intermediate measurements for
    long runs, where values are cumulative), so the last row describes the
    run that just finished. Returns ``None`` when the file is absent,
    unreadable, or empty — callers must treat that as "no data", not zero.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as e:
        logger.warning("Could not read emissions CSV %s: %s", path, e)
        return None
    if not rows:
        return None
    return _summarize_row(rows[-1])


def summarize_training_emissions(output_dir: Path) -> dict[str, Any] | None:
    """Read back the emissions summary a training run wrote into ``output_dir``.

    Settings-gated: returns ``None`` when tracking is disabled. Orchestrators
    call this after training to attach emissions to the response and upload
    payload; a missing CSV (tracking failed or was disabled mid-run) yields
    ``None``.
    """
    if not get_settings().emissions_tracking_enabled:
        return None
    return summarize_emissions_csv(Path(output_dir) / EMISSIONS_CSV_FILENAME)


def coerce_emissions_summary(value: Any) -> dict[str, Any] | None:
    """Coerce an arbitrary value into a JSON-safe emissions summary or None.

    Orchestrators call this on whatever the training layer returned under
    ``emissions`` before putting it into a response or upload payload, so a
    non-dict (missing key, wrong type) can never leak into the JSON output.
    """
    return value if isinstance(value, dict) else None


def emissions_extra_fields(summary: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten an emissions summary onto AutoDW upload form fields.

    Numeric fields are forwarded as floats, ``country_name`` /
    ``country_iso_code`` as strings; everything else is dropped. ``None`` or
    an empty summary produces an empty dict, so uploads without emissions
    data are unchanged.
    """
    if not summary:
        return {}
    numeric_keys = tuple(key for _, key in _NUMERIC_SUMMARY_FIELDS)
    fields: dict[str, Any] = {}
    for key in numeric_keys:
        value = summary.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            fields[key] = float(value)
    for _, key in _TEXT_SUMMARY_FIELDS:
        value = summary.get(key)
        if value:
            fields[key] = str(value)
    return fields


def copy_emissions_csv(source_csv: Path, dest_dir: Path) -> None:
    """Copy ``emissions.csv`` into ``dest_dir`` (best-effort, never raises).

    Used to ship the CSV inside the model folder that gets zipped and
    uploaded; a missing source or copy failure is logged and skipped.
    """
    source_csv = Path(source_csv)
    if not source_csv.exists():
        logger.debug("No emissions CSV at %s; nothing to copy", source_csv)
        return
    try:
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_csv, Path(dest_dir) / source_csv.name)
        logger.debug("Copied emissions CSV %s -> %s", source_csv, dest_dir)
    except Exception as e:
        logger.warning("Failed to copy emissions CSV into model dir: %s", e)


def _load_tracker(**kwargs: Any) -> Any:
    """Instantiate the codecarbon tracker (isolated for tests).

    With a country code configured (the default) we use the offline tracker:
    the carbon intensity is looked up from that country's energy mix, which
    is deterministic and needs no network. Without one, the online tracker
    autodetects the location from the server's IP.
    """
    if kwargs.get("country_iso_code"):
        from codecarbon import OfflineEmissionsTracker

        return OfflineEmissionsTracker(**kwargs)

    from codecarbon import EmissionsTracker

    return EmissionsTracker(**kwargs)


class TrainingEmissionsTracker:
    """Best-effort codecarbon wrapper around one training run.

    Usable as a context manager::

        tracker = TrainingEmissionsTracker("alfie-automl-image_classification", workdir)
        with tracker:
            ... train ...
        summary = tracker.summary  # dict | None

    ``start()`` is a no-op when tracking is disabled or codecarbon is not
    importable; ``stop()`` never raises. After ``stop()`` (including via the
    context manager), ``summary`` holds the JSON-safe dict built from the
    final row of the written ``emissions.csv`` — or ``None`` when nothing
    could be measured.
    """

    def __init__(
        self,
        project_name: str,
        output_dir: Path,
        output_file: str = EMISSIONS_CSV_FILENAME,
    ) -> None:
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_file = output_file
        self.csv_path: Path = self.output_dir / output_file
        self.summary: dict[str, Any] | None = None
        self._tracker: Any = None

    def start(self) -> None:
        """Start background measurement; a no-op on any failure."""
        settings = get_settings()
        if not settings.emissions_tracking_enabled:
            logger.debug(
                "Emissions tracking disabled; not tracking %s", self.project_name
            )
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, Any] = {
                "project_name": self.project_name,
                "output_dir": str(self.output_dir),
                "output_file": self.output_file,
                "log_level": "warning",
            }
            # Country configured (default: NLD) -> offline tracker with a
            # deterministic carbon intensity for that country; empty -> the
            # online tracker autodetects the location via IP lookup.
            if settings.emissions_country_iso_code:
                kwargs["country_iso_code"] = settings.emissions_country_iso_code
            self._tracker = _load_tracker(**kwargs)
            self._tracker.start()
            logger.info(
                "Emissions tracking started for %s (CSV: %s)",
                self.project_name,
                self.csv_path,
            )
        except Exception as e:
            logger.warning(
                "Failed to start emissions tracking for %s: %s", self.project_name, e
            )
            self._tracker = None

    def stop(self) -> dict[str, Any] | None:
        """Stop measurement and build the summary from the CSV; never raises."""
        if self._tracker is not None:
            try:
                self._tracker.stop()
            except Exception as e:
                logger.warning("Emissions tracker did not stop cleanly: %s", e)
            self._tracker = None
        try:
            self.summary = summarize_emissions_csv(self.csv_path)
        except Exception as e:
            logger.warning("Failed to summarize emissions data: %s", e)
            self.summary = None
        return self.summary

    def __enter__(self) -> TrainingEmissionsTracker:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Stop on every path — including exceptions — so the measurement
        # thread is always cleaned up and the CSV row is flushed.
        self.stop()
