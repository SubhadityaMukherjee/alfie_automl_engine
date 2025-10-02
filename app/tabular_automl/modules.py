import os
from typing import Optional

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE: float = float(
    os.getenv("DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE", 0.8)
)


class AutoMLTrainer:
    """Wrapper around AutoGluon Tabular training routines."""

    def __init__(
        self,
        save_model_path,
        DatasetClass=TabularDataset,
        PredictorClass=TabularPredictor,
    ):
        self.save_model_path = save_model_path
        self.DatasetClass = DatasetClass
        self.PredictorClass = PredictorClass

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        target_column: str,
        time_limit: int,
    ) -> pd.DataFrame | str:
        """Train AutoGluon Tabular and return leaderboard or error."""
        final_train_df, final_test_df = self.train_test_split(
            test_df=test_df, train_df=train_df
        )

        train_dataset = self.DatasetClass(final_train_df)
        test_dataset = self.DatasetClass(final_test_df)

        predictor = self.PredictorClass(
            label=target_column, path=self.save_model_path
        ).fit(train_data=train_dataset, time_limit=time_limit)

        try:
            return predictor.leaderboard(test_dataset)
        except Exception as e:
            return str(e)

    def train_test_split(self, test_df: Optional[pd.DataFrame], train_df: pd.DataFrame):
        if test_df is None:
            final_train_df = train_df.sample(
                frac=DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE, random_state=42
            )
            final_test_df = train_df.drop(index=final_train_df.index.tolist())
        else:
            final_train_df = train_df
            final_test_df = test_df
        return final_train_df, final_test_df
