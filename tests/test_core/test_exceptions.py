"""Tests for app.core.exceptions hierarchy."""

import pytest

from app.core.exceptions import (
    AutoDWDownloadError,
    AutoDWUploadError,
    AutoMLChatError,
    AutoMLDataError,
    AutoMLDeploymentError,
    AutoMLError,
    AutoMLImageError,
    AutoMLLeaderboardError,
    AutoMLRuntimeError,
    AutoMLSerializationError,
    AutoMLTrainingError,
    AutoMLValidationError,
)

AutoMLConfigError = AutoMLValidationError  # config errors share the same base


def test_automl_error_is_base():
    assert issubclass(AutoMLError, Exception)


def test_validation_error_is_value_error():
    assert issubclass(AutoMLValidationError, ValueError)
    assert issubclass(AutoMLValidationError, AutoMLError)


def test_runtime_error_is_runtime_error():
    assert issubclass(AutoMLRuntimeError, RuntimeError)
    assert issubclass(AutoMLRuntimeError, AutoMLError)


@pytest.mark.parametrize(
    "exc_cls",
    [
        AutoMLValidationError,
        AutoMLDataError,
        AutoMLConfigError,
        AutoMLRuntimeError,
        AutoMLTrainingError,
        AutoMLDeploymentError,
        AutoMLLeaderboardError,
        AutoDWDownloadError,
        AutoDWUploadError,
        AutoMLSerializationError,
        AutoMLImageError,
        AutoMLChatError,
    ],
)
def test_all_exceptions_inherit_from_automl_error(exc_cls):
    assert issubclass(exc_cls, AutoMLError)


@pytest.mark.parametrize(
    "exc_cls",
    [
        AutoMLDataError,
        AutoMLConfigError,
    ],
)
def test_validation_subclasses_are_value_errors(exc_cls):
    assert issubclass(exc_cls, ValueError)


@pytest.mark.parametrize(
    "exc_cls",
    [
        AutoMLTrainingError,
        AutoMLDeploymentError,
        AutoMLLeaderboardError,
        AutoDWDownloadError,
        AutoDWUploadError,
        AutoMLSerializationError,
        AutoMLImageError,
        AutoMLChatError,
    ],
)
def test_runtime_subclasses_are_runtime_errors(exc_cls):
    assert issubclass(exc_cls, RuntimeError)
