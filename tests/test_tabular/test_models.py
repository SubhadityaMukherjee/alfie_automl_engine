import pytest
from app.tabular_automl.models import TabularTask, TabularSupervisedClassificationTask, TabularSupervisedRegressionTask, TabularSupervisedTimeSeriesTask

@pytest.mark.parametrize("classname", [TabularTask, TabularSupervisedClassificationTask, TabularSupervisedRegressionTask, TabularSupervisedTimeSeriesTask])
def test_task_type(classname:TabularTask):
    assert classname.task_type is not None



