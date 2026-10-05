"""Tests for app.core.service_helpers.metadata_indicates_zip_download."""

from typing import cast

from app.core.service_helpers import metadata_indicates_zip_download


def test_file_type_zip():
    assert metadata_indicates_zip_download({"file_type": "zip"}) is True


def test_file_type_zip_normalization():
    assert metadata_indicates_zip_download({"file_type": " ZIP "}) is True


def test_original_filename_zip():
    assert (
        metadata_indicates_zip_download(
            {"file_type": "csv, jpg", "original_filename": "Dataset.ZIP"}
        )
        is True
    )


def test_is_folder():
    assert metadata_indicates_zip_download({"file_type": "jpg, csv", "is_folder": True})


def test_contents_type_only_is_not_zip():
    assert (
        metadata_indicates_zip_download(
            {"file_type": "csv, jpg", "original_filename": "data.csv"}
        )
        is False
    )


def test_plain_csv_dataset():
    assert (
        metadata_indicates_zip_download(
            {"file_type": "csv", "original_filename": "train.csv", "is_folder": False}
        )
        is False
    )


def test_non_dict_metadata():
    assert metadata_indicates_zip_download(cast(dict, None)) is False
    assert metadata_indicates_zip_download(cast(dict, "zip")) is False


def test_empty_metadata():
    assert metadata_indicates_zip_download({}) is False


def test_falsy_folder_values():
    assert metadata_indicates_zip_download({"is_folder": False}) is False
    assert metadata_indicates_zip_download({"is_folder": None}) is False
