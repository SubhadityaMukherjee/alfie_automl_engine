from pathlib import Path
from typing import Dict, List, Optional, Union

from src.llm_task.tasks import LLMProcessingTask
from src.tabular_task.pipeline import AutoGluonTabularPipeline
from src.tabular_task.tasks import (TabularSupervisedClassificationTask,
                                    TabularSupervisedRegressionTask,
                                    TabularSupervisedTimeSeriesTask)

class PipelineManager:
    @staticmethod
    def create_pipeline(
        task_type: str, file_info: dict
    ) -> Optional[AutoGluonTabularPipeline]:
        try:
            import src.tabular_task.tasks as tabular_task_modules

            task_class = getattr(tabular_task_modules, task_type)

            if task_class in [
                TabularSupervisedClassificationTask,
                TabularSupervisedRegressionTask,
                TabularSupervisedTimeSeriesTask,
            ]:
                task = task_class(
                    target_feature=file_info["target_col"],
                    train_file_path=Path(file_info["train"]),
                    test_file_path=Path(file_info["test"]),
                    time_stamp_col=file_info["time_stamp_col"],
                )
                pipeline = AutoGluonTabularPipeline(task, save_path="autogluon_output")
                return pipeline
            elif task_class == LLMProcessingTask:
                # Handle LLM specific pipeline creation
                pass

        except Exception as e:
            print(f"Error creating pipeline: {e}")
            return None