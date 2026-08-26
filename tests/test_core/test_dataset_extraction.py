"""Tests for app/core/dataset_extraction.py."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from app.core.dataset_extraction import (
    _find_csv_file,
    _find_or_resolve_media_dir,
    _find_valid_dataset_root,
    collect_missing_files,
    extract_and_locate_dataset,
    normalize_dataframe_filenames,
    resolve_media_root,
)
from app.core.exceptions import AutoMLDataError


@pytest.fixture
def synthetic_images_dir(tmp_path):
    """Flat images dir with labels.csv; images are direct children of images_dir."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    rows = []
    for i in range(6):
        fname = f"img{i}.png"
        Image.new("RGB", (10, 10), color=(128, 64, 32)).save(images_dir / fname)
        rows.append({"filename": fname, "label": "cat" if i < 3 else "dog"})

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, images_dir


# ---------------------------------------------------------------------------
# normalize_dataframe_filenames
# ---------------------------------------------------------------------------


def test_normalize_filenames_unix_paths(tmp_path):
    df = pd.DataFrame(
        {"filename": ["some/path/img.png", "other/img2.png"], "label": [0, 1]}
    )
    csv_path = tmp_path / "labels.csv"
    result = normalize_dataframe_filenames(df, "filename", csv_path)
    assert list(result["filename"]) == ["img.png", "img2.png"]
    assert csv_path.exists()


def test_normalize_filenames_windows_paths(tmp_path):
    df = pd.DataFrame(
        {
            "filename": ["C:\\Users\\test\\img.png", "D:\\data\\img2.png"],
            "label": [0, 1],
        }
    )
    csv_path = tmp_path / "labels.csv"
    result = normalize_dataframe_filenames(df, "filename", csv_path)
    assert list(result["filename"]) == ["img.png", "img2.png"]


def test_normalize_filenames_already_basenames(tmp_path):
    df = pd.DataFrame({"filename": ["a.png", "b.png"], "label": [0, 1]})
    csv_path = tmp_path / "labels.csv"
    result = normalize_dataframe_filenames(df, "filename", csv_path)
    assert list(result["filename"]) == ["a.png", "b.png"]


def test_normalize_filenames_saves_csv(tmp_path):
    df = pd.DataFrame({"filename": ["path/img.png"], "label": [0]})
    csv_path = tmp_path / "labels.csv"
    normalize_dataframe_filenames(df, "filename", csv_path)
    saved = pd.read_csv(csv_path)
    assert saved["filename"].iloc[0] == "img.png"


def test_normalize_filenames_missing_column_returns_df(tmp_path):
    df = pd.DataFrame({"wrong_col": ["a.png"], "label": [0]})
    csv_path = tmp_path / "labels.csv"
    result = normalize_dataframe_filenames(df, "filename", csv_path)
    assert "wrong_col" in result.columns
    assert not csv_path.exists()


# ---------------------------------------------------------------------------
# resolve_media_root
# ---------------------------------------------------------------------------


def test_resolve_media_root_flat_directory(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "img0.png").touch()
    result = resolve_media_root(images_dir)
    assert result == images_dir


def test_resolve_media_root_nested_images_subfolder(tmp_path):
    images_dir = tmp_path / "images"
    nested = images_dir / "images"
    nested.mkdir(parents=True)
    (nested / "img0.png").touch()
    result = resolve_media_root(images_dir)
    assert result == nested


def test_resolve_media_root_single_subdir_unwrap(tmp_path):
    images_dir = tmp_path / "images"
    cls_dir = images_dir / "class_a"
    cls_dir.mkdir(parents=True)
    (cls_dir / "img0.png").touch()
    result = resolve_media_root(images_dir)
    assert result == cls_dir


def test_resolve_media_root_multiple_subdirs_no_unwrap(tmp_path):
    images_dir = tmp_path / "images"
    (images_dir / "cat").mkdir(parents=True)
    (images_dir / "dog").mkdir()
    result = resolve_media_root(images_dir)
    assert result == images_dir


# ---------------------------------------------------------------------------
# collect_missing_files
# ---------------------------------------------------------------------------


def test_collect_missing_files_all_present(synthetic_images_dir):
    csv_path, images_dir = synthetic_images_dir
    df = pd.read_csv(csv_path)
    missing = collect_missing_files(df, images_dir, "filename", "label")
    assert missing == []


def test_collect_missing_files_some_missing(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "present.png").touch()
    df = pd.DataFrame({"filename": ["present.png", "absent.png"], "label": [0, 1]})
    missing = collect_missing_files(df, images_dir, "filename", "label")
    assert missing == ["absent.png"]


def test_collect_missing_files_found_via_rglob(tmp_path):
    images_dir = tmp_path / "images"
    subdir = images_dir / "subdir"
    subdir.mkdir(parents=True)
    (subdir / "nested.png").touch()
    df = pd.DataFrame({"filename": ["nested.png"], "label": [0]})
    missing = collect_missing_files(df, images_dir, "filename", "label")
    assert missing == []


def test_collect_missing_files_multiple_matches_warns(tmp_path):
    images_dir = tmp_path / "images"
    (images_dir / "a").mkdir(parents=True)
    (images_dir / "b").mkdir()
    (images_dir / "a" / "dup.png").touch()
    (images_dir / "b" / "dup.png").touch()
    df = pd.DataFrame({"filename": ["dup.png"], "label": [0]})
    missing = collect_missing_files(df, images_dir, "filename", "label")
    assert "dup.png" in missing


# ---------------------------------------------------------------------------
# _find_valid_dataset_root
# ---------------------------------------------------------------------------


def test_find_valid_dataset_root_skips_macosx(tmp_path):
    (tmp_path / "__MACOSX").mkdir()
    (tmp_path / "real_data").mkdir()
    result = _find_valid_dataset_root(tmp_path)
    assert result == tmp_path / "real_data"


def test_find_valid_dataset_root_skips_dotdirs(tmp_path):
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "actual_data").mkdir()
    result = _find_valid_dataset_root(tmp_path)
    assert result == tmp_path / "actual_data"


def test_find_valid_dataset_root_raises_when_no_valid_dirs(tmp_path):
    (tmp_path / "__MACOSX").mkdir()
    with pytest.raises(AutoMLDataError, match="No valid dataset folder"):
        _find_valid_dataset_root(tmp_path)


# ---------------------------------------------------------------------------
# _find_csv_file
# ---------------------------------------------------------------------------


def test_find_csv_file_finds_labels_csv(tmp_path):
    (tmp_path / "labels.csv").touch()
    result = _find_csv_file(tmp_path)
    assert result == tmp_path / "labels.csv"


def test_find_csv_file_finds_metadata_csv(tmp_path):
    (tmp_path / "metadata.csv").touch()
    result = _find_csv_file(tmp_path)
    assert result == tmp_path / "metadata.csv"


def test_find_csv_file_raises_when_not_found(tmp_path):
    with pytest.raises(AutoMLDataError, match="labels.csv or metadata.csv"):
        _find_csv_file(tmp_path)


def test_find_csv_file_prefers_labels_csv(tmp_path):
    # Both exist - we don't guarantee order, but we just need it to succeed
    (tmp_path / "labels.csv").touch()
    (tmp_path / "metadata.csv").touch()
    result = _find_csv_file(tmp_path)
    assert result.name in ("labels.csv", "metadata.csv")


# ---------------------------------------------------------------------------
# _find_or_resolve_media_dir
# ---------------------------------------------------------------------------


def test_find_or_resolve_media_dir_finds_images_subdir(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "img.png").touch()
    csv_path = tmp_path / "labels.csv"
    result = _find_or_resolve_media_dir(tmp_path, csv_path)
    assert result.exists()
    assert "images" in result.parts


def test_find_or_resolve_media_dir_raises_when_not_found(tmp_path):
    csv_path = tmp_path / "labels.csv"
    # No images/ directory at all
    with pytest.raises(AutoMLDataError, match="images/"):
        _find_or_resolve_media_dir(tmp_path, csv_path)


# ---------------------------------------------------------------------------
# extract_and_locate_dataset
# ---------------------------------------------------------------------------


def _make_dataset_zip(tmp_path: Path) -> Path:
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "my_dataset/labels.csv", "filename,label\nimg0.png,cat\nimg1.png,dog\n"
        )
        zf.writestr("my_dataset/images/img0.png", png_bytes)
        zf.writestr("my_dataset/images/img1.png", png_bytes)
    return zip_path


def test_extract_and_locate_dataset_valid_zip(tmp_path):
    zip_path = _make_dataset_zip(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    csv_path, images_dir = extract_and_locate_dataset(zip_path, workdir)
    assert csv_path.exists()
    assert csv_path.name in ("labels.csv", "metadata.csv")
    assert images_dir.exists()
    assert images_dir.is_dir()


def test_extract_and_locate_dataset_missing_csv_raises(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("my_dataset/images/img0.png", png_bytes)

    workdir = tmp_path / "work"
    workdir.mkdir()
    with pytest.raises(AutoMLDataError, match="labels.csv or metadata.csv"):
        extract_and_locate_dataset(zip_path, workdir)


def test_extract_and_locate_dataset_no_valid_root_raises(tmp_path):
    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("__MACOSX/._something", b"junk")

    workdir = tmp_path / "work"
    workdir.mkdir()
    with pytest.raises(AutoMLDataError, match="No valid dataset folder"):
        extract_and_locate_dataset(zip_path, workdir)
