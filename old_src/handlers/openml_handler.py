from typing import Optional, Union

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from old_src.tasks import (
    TabularSupervisedClassificationTask,
    TabularSupervisedRegressionTask,
)


class OpenMLDatasetHandler:
    """
    Handles OpenML task specific features
    """

    def __init__(self, dataset_id: int):
        self.dataset = openml.datasets.get_dataset(dataset_id)
        self.task_type = None

    def get_target_col_type(self, dataset, target_col_name):
        """
        Get the type of the target column based on it's values
        """
        try:
            if dataset.features:
                return next(
                    (
                        feature.data_type
                        for feature in dataset.features.values()
                        if feature.name == target_col_name
                    ),
                    None,
                )
        except Exception as e:
            print(f"Error getting target column type: {e}")
            return None

    def get_task_type(
        self,
    ) -> Optional[
        Union[TabularSupervisedClassificationTask, TabularSupervisedRegressionTask]
    ]:
        try:
            target_col_name = self.dataset.default_target_attribute
            target_col_type = self.get_target_col_type(self.dataset, target_col_name)

            if target_col_type:
                if target_col_type in ["nominal", "string", "categorical"]:
                    self.task_type = TabularSupervisedClassificationTask
                    # try:
                    #     self.class_labels = self.dataset.get_data()[0][target_col_name].unique()
                    # except Exception as e:
                    #     return "No class labels found"

                elif target_col_type == "numeric":
                    # evaluation_measure = "mean_absolute_error"
                    task_type = openml.tasks.TaskType.SUPERVISED_REGRESSION
                    self.task_type = TabularSupervisedRegressionTask
                    # self.class_labels = []
                else:
                    return None

        except Exception as e:
            print(f"Error getting task type: {e}")
            return None

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        X, y, _, _ = self.dataset.get_data(target=self.dataset.default_target_attribute)

        # Convert categorical columns to numerical
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)

        # Encode target if it's categorical
        if y.dtype == "object" or y.dtype.name == "category":
            y = LabelEncoder().fit_transform(y)

        X = X.to_numpy(dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, stratify=y
        )

        train_df = pd.DataFrame(X_train)
        train_df["target"] = y_train

        test_df = pd.DataFrame(X_test)
        test_df["target"] = y_test

        return train_df, test_df
