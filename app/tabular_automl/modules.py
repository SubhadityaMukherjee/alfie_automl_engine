import logging
import os
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE: float = float(
    os.getenv("DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE", 0.8)
)


class AutoMLTrainer:
    """Wrapper around AutoGluon Tabular training routines."""

    def __init__(
        self,
        save_model_path: Path,
        DatasetClass=TabularDataset,
        PredictorClass=TabularPredictor,
    ):
        self.save_model_path: Path = Path(save_model_path)
        self.DatasetClass = DatasetClass
        self.PredictorClass = PredictorClass
        logger.debug(f"Automl trainer, model path {self.save_model_path}")

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame | None,
        target_column: str,
        time_limit: int,
    ) -> tuple[pd.DataFrame | str, TabularPredictor]:
        """Train AutoGluon Tabular and return leaderboard or error."""
        final_train_df, final_test_df = self.train_test_split(
            test_df=test_df, train_df=train_df
        )

        train_dataset = self.DatasetClass(final_train_df)
        test_dataset = self.DatasetClass(final_test_df)

        predictor = self.PredictorClass(
            label=target_column, path=self.save_model_path
        ).fit(train_data=train_dataset, time_limit=time_limit)

        save_path_clone_opt = self.save_model_path/ '-clone-opt'
        path_clone_opt = predictor.clone_for_deployment(path=str(save_path_clone_opt))
        predictor_clone_opt = self.PredictorClass.load(path=path_clone_opt)

        try:
            return predictor.leaderboard(test_dataset), predictor_clone_opt
        except Exception as e:
            logger.error(f"AutoML trainer failed {e}")
            return str(e)

    def train_test_split(self, test_df: pd.DataFrame | None, train_df: pd.DataFrame|None = None):
        if test_df is None:
            logger.debug(f"Test dataset not found, creating split")
            final_train_df = train_df.sample(
                frac=DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE, random_state=42
            )
            final_test_df = train_df.drop(index=final_train_df.index.tolist())
        else:
            logger.debug(f"Test dataset found")
            final_train_df = train_df
            final_test_df = test_df
        return final_train_df, final_test_df
