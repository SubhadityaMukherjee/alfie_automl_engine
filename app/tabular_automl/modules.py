from typing import Optional

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor


class AutoMLTrainer:
    def __init__(self, save_model_path):
        self.save_model_path = save_model_path

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        target_column: str,
        time_limit: int,
    ) -> pd.DataFrame | str:
        """Trains a tabular AutoML pipeline on provided DataFrames. If test_df is None, performs an 80/20 split.

        Returns a leaderboard of best models or error string.
        """

        final_train_df, final_test_df = self.train_test_split(
            test_df=test_df, train_df=train_df
        )

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

    def train_test_split(self, test_df: Optional[pd.DataFrame], train_df: pd.DataFrame):
        """Split data if test_df doesn't exist, otherwise return provided splits."""
        if test_df is None:
            final_train_df = train_df.sample(frac=0.8, random_state=42)
            final_test_df = train_df.drop(index=final_train_df.index.tolist())
        else:
            final_train_df = train_df
            final_test_df = test_df
        return final_train_df, final_test_df
