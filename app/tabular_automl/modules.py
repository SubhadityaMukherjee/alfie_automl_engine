from typing import Optional

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor


class AutoMLTrainer:
    def __init__(self, save_model_path):
        self.save_model_path = save_model_path

    def train(
        self,
        train_file: str,
        test_file: Optional[str],
        target_column: str,
        time_limit: int,
    ) -> pd.DataFrame | str:
        """Trains a tabular automl pipeline on the data, splits if splits dont exist, returns a leaderboard of best models"""

        train_df = pd.read_csv(train_file)
        final_train_df, final_test_df = self._train_test_split(test_file, train_df)

        train_dataset = TabularDataset(final_train_df)
        test_dataset = TabularDataset(final_test_df)

        predictor = TabularPredictor(
            label=target_column, path=self.save_model_path
        ).fit(train_data=train_dataset, time_limit=time_limit)

        try:
            leaderboard = predictor.leaderboard(test_dataset)
        except Exception as e:
            return str(e)
        return leaderboard

    def _train_test_split(self, test_file, train_df):
        """Split data if test file doesnt exist"""
        if test_file is None:
            final_train_df = train_df.sample(frac=0.8)
            final_test_df = train_df.drop(final_train_df.index)
        else:
            final_train_df = train_df
            final_test_df = pd.read_csv(test_file)
        return final_train_df,final_test_df
