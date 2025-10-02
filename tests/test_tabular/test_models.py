import pytest
from pathlib import Path
from app.tabular_automl.models import TabularTask, TabularSupervisedClassificationTask, TabularSupervisedRegressionTask, TabularSupervisedTimeSeriesTask

@pytest.mark.parametrize(
    "task_cls",
    [
        TabularSupervisedClassificationTask,
        TabularSupervisedRegressionTask,
    ],
)
def test_task_type_classification_and_regression(task_cls):
    task = task_cls(
        target_feature="test",
        time_stamp_col=None,
        train_file_path=Path("train.csv"),
        test_file_path=None,
    )
    assert isinstance(task.task_type, str)


def test_task_type_time_series():
    task = TabularSupervisedTimeSeriesTask(
        target_feature="test",
        time_stamp_col="timestamp",
        train_file_path=Path("train.csv"),
        test_file_path=None,
    )
    assert isinstance(task.task_type, str)


@pytest.mark.parametrize("field,value", [
    ("target_feature", "target"),
    ("time_stamp_col", None),
    ("train_file_path", Path("train.csv")),
    ("test_file_path", None),
])
def test_tabular_task_fields(field, value):
    task = TabularTask(
        target_feature="target",
        time_stamp_col=None,
        train_file_path=Path("train.csv"),
        test_file_path=None,
    )
    # ensure the field exists and has expected type
    assert getattr(task, field) == value

