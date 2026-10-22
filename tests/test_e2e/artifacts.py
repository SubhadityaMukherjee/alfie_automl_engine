"""Builders for tiny end-to-end dataset ZIPs + artifact inspection helpers.

The ZIP layouts mirror what AutoDW serves for folder uploads (see
``app.core.dataset_extraction``): a labels CSV at the root plus an ``images/``
media folder — the folder is named ``images`` for every modality by packaging
convention. Vision/multimodal images live in per-class subfolders (string
labels make the torch datasets resolve ``<media_root>/<label>/<file>``),
while audio clips sit flat (the audio dataset resolves paths directly).

Everything is built in-memory; sizes are deliberately tiny so whole-pipeline
tests stay fast.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

N_PER_CLASS = 6  # rows per class -> 12 rows total (enough for stratified splits)

# --- text corpus ------------------------------------------------------------

POSITIVE_WORDS = ("great", "lovely", "bright", "calm", "delight")
NEGATIVE_WORDS = ("grim", "dull", "dark", "harsh", "bleak")


def text_rows() -> list[dict[str, str]]:
    """Build a balanced pos/neg text dataset using only known vocab words."""
    rows: list[dict[str, str]] = []
    for i in range(N_PER_CLASS):
        pos = " ".join(POSITIVE_WORDS[: (i % len(POSITIVE_WORDS)) + 2])
        neg = " ".join(NEGATIVE_WORDS[: (i % len(NEGATIVE_WORDS)) + 2])
        rows.append({"text": pos, "label": "pos"})
        rows.append({"text": neg, "label": "neg"})
    return rows


def text_vocab() -> dict[str, int]:
    """WordLevel vocab covering the corpus plus the special tokens."""
    words = list(POSITIVE_WORDS) + list(NEGATIVE_WORDS)
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for word in sorted(set(words)):
        vocab[word] = len(vocab)
    return vocab


# --- ZIP helpers -------------------------------------------------------------


def make_zip_bytes(entries: dict[str, Any]) -> bytes:
    """Build an in-memory ZIP from {archive name: bytes or writer callable}.

    A callable value receives an output ``BytesIO`` to write into (used for
    images/WAVs produced by PIL/scipy).
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries.items():
            if callable(payload):
                out = io.BytesIO()
                payload(out)
                zf.writestr(name, out.getvalue())
            else:
                zf.writestr(name, payload)
    return buf.getvalue()


def _tiny_png_bytes(seed: int) -> bytes:
    """Deterministic small PNG with a class/index-dependent colour."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _tiny_wav_bytes(seed: int, frequency_hz: float) -> bytes:
    """Deterministic short sine WAV (16 kHz mono, int16)."""
    from scipy.io import wavfile

    sample_rate = 16_000
    duration_s = 0.4
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * frequency_hz * t)
    waveform += 0.01 * rng.standard_normal(t.shape)
    pcm = (waveform * 32767).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, pcm)
    return buf.getvalue()


# --- per-modality dataset ZIPs -----------------------------------------------


def vision_zip_bytes() -> bytes:
    """Image classification ZIP: labels.csv + images/<class>/<file>.png."""
    entries: dict[str, Any] = {}
    rows = []
    for i in range(N_PER_CLASS):
        for cls in ("cat", "dog"):
            fname = f"{cls}_{i:02d}.png"
            entries[f"images/{cls}/{fname}"] = _tiny_png_bytes(
                seed=hash((cls, i)) % 2**31
            )
            rows.append({"filename": fname, "label": cls})
    entries["labels.csv"] = pd.DataFrame(rows).to_csv(index=False)
    return make_zip_bytes(entries)


def multimodal_zip_bytes() -> bytes:
    """Multimodal ZIP: metadata.csv with numeric + categorical aux columns."""
    entries: dict[str, Any] = {}
    rows = []
    for i in range(N_PER_CLASS):
        for cls in ("cat", "dog"):
            fname = f"{cls}_{i:02d}.png"
            entries[f"images/{cls}/{fname}"] = _tiny_png_bytes(
                seed=hash(("mm", cls, i)) % 2**31
            )
            rows.append(
                {
                    "filename": fname,
                    "label": cls,
                    "weight": round(2.0 + i + (0 if cls == "cat" else 5.0), 2),
                    "color": "black" if cls == "cat" else "brown",
                }
            )
    entries["metadata.csv"] = pd.DataFrame(rows).to_csv(index=False)
    return make_zip_bytes(entries)


def audio_zip_bytes() -> bytes:
    """Audio classification ZIP: labels.csv + flat images/<file>.wav clips."""
    entries: dict[str, Any] = {}
    rows = []
    for i in range(N_PER_CLASS):
        for cls, freq in (("beep", 660.0), ("hum", 110.0)):
            fname = f"{cls}_{i:02d}.wav"
            entries[f"images/{fname}"] = _tiny_wav_bytes(
                seed=hash((cls, i)) % 2**31, frequency_hz=freq
            )
            rows.append({"filename": fname, "label": cls})
    entries["labels.csv"] = pd.DataFrame(rows).to_csv(index=False)
    return make_zip_bytes(entries)


def text_zip_bytes() -> bytes:
    """Text classification ZIP: labels.csv + an (empty) images/ folder.

    Text datasets carry no media, but the shared extractor requires the
    packaging-convention ``images/`` folder to exist inside the archive.
    """
    entries: dict[str, Any] = {
        "labels.csv": pd.DataFrame(text_rows()).to_csv(index=False),
        "images/.keep": "",
    }
    return make_zip_bytes(entries)


def vision_zip_with_missing_image_bytes() -> bytes:
    """Vision ZIP whose CSV references one image that is absent on disk."""
    entries: dict[str, Any] = {}
    rows = []
    for i in range(N_PER_CLASS):
        for cls in ("cat", "dog"):
            fname = f"{cls}_{i:02d}.png"
            rows.append({"filename": fname, "label": cls})
            if not (cls == "dog" and i == 0):  # omit one referenced file
                entries[f"images/{cls}/{fname}"] = _tiny_png_bytes(
                    seed=hash(("missing", cls, i)) % 2**31
                )
    entries["labels.csv"] = pd.DataFrame(rows).to_csv(index=False)
    return make_zip_bytes(entries)


# --- artifact inspection -----------------------------------------------------


def _as_zipfile_source(zip_like):
    """Accept a path or raw ZIP bytes and return something zipfile can open."""
    if isinstance(zip_like, (bytes, bytearray)):
        return io.BytesIO(zip_like)
    return zip_like


def zip_entry_names(zip_like) -> set[str]:
    """Return the set of member names inside a ZIP (path or raw bytes)."""
    with zipfile.ZipFile(_as_zipfile_source(zip_like)) as zf:
        return set(zf.namelist())


def zip_member_json(zip_like, member: str) -> dict:
    """Read a JSON member from a ZIP (path or raw bytes)."""
    with zipfile.ZipFile(_as_zipfile_source(zip_like)) as zf:
        return json.loads(zf.read(member))
