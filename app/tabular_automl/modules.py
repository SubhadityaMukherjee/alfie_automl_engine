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

DEFAULT_TIME_LIMIT: int = int(os.getenv("DEFAULT_TIME_LIMIT", 100))


class AutoMLTrainer:
    """Wrapper around AutoGluon Tabular training routines."""

    def __init__(
        self,
        save_model_path: Path,
        DatasetClass=TabularDataset,
        PredictorClass=TabularPredictor,
    ):
        if save_model_path == "":
            raise ValueError("save_model_path cannot be None or empty")

        self.save_model_path: Path = Path(save_model_path)

        if self.save_model_path.exists() and not self.save_model_path.is_dir():
            raise ValueError(
                f"save_model_path must be a directory, got: {self.save_model_path}"
            )

        self.DatasetClass = DatasetClass
        self.PredictorClass = PredictorClass
        logger.debug(f"Automl trainer, model path {self.save_model_path}")

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame | None,
        target_column: str,
        time_limit: int,
    ) -> tuple[pd.DataFrame, TabularPredictor]:
        """Train AutoGluon Tabular and return leaderboard or error."""

        if train_df is None or train_df.empty:
            raise ValueError("train_df cannot be None or empty")

        if not target_column or not isinstance(target_column, str):
            raise ValueError("target_column must be a non-empty string")

        if target_column not in train_df.columns:
            raise ValueError(
                f"target_column '{target_column}' not found in train_df columns: {train_df.columns.tolist()}"
            )

        if test_df is not None and target_column not in test_df.columns:
            raise ValueError(
                f"target_column '{target_column}' not found in test_df columns: {test_df.columns.tolist()}"
            )

        if not isinstance(time_limit, int) or time_limit <= 0:
            raise ValueError(f"time_limit must be a positive integer, got {time_limit}")

        try:
            final_train_df, final_test_df = self.train_test_split(
                test_df=test_df, train_df=train_df
            )
        except Exception as e:
            logger.error(f"Failed to split train/test data: {e}")
            raise ValueError(f"Train/test split failed: {e}") from e

        try:
            train_dataset = self.DatasetClass(final_train_df)
            test_dataset = self.DatasetClass(final_test_df)
        except Exception as e:
            logger.error(f"Failed to create TabularDataset: {e}")
            raise ValueError(f"Dataset creation failed: {e}") from e

        try:
            predictor = self.PredictorClass(
                label=target_column, path=str(self.save_model_path)
            ).fit(train_data=train_dataset, time_limit=time_limit)
        except Exception as e:
            logger.error(f"AutoGluon training failed: {e}")
            raise RuntimeError(f"Model training failed: {e}") from e

        try:
            save_path_clone_opt = self.save_model_path / "-clone-opt"
            path_clone_opt = predictor.clone_for_deployment(
                path=str(save_path_clone_opt)
            )
            predictor_clone_opt = self.PredictorClass.load(path=str(path_clone_opt))
        except Exception as e:
            logger.error(f"Failed to clone model for deployment: {e}")
            raise RuntimeError(f"Model cloning failed: {e}") from e

        try:
            leaderboard = predictor.leaderboard(test_dataset)
            return leaderboard, predictor_clone_opt
        except Exception as e:
            logger.error(f"Failed to generate leaderboard: {e}")
            raise RuntimeError(f"Leaderboard generation failed: {e}") from e

    def train_test_split(
        self, test_df: pd.DataFrame | None, train_df: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        if train_df is None:
            raise ValueError("train_df cannot be None")

        if train_df.empty:
            raise ValueError("train_df cannot be empty")

        if (
            DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE <= 0
            or DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE >= 1
        ):
            raise ValueError(
                f"DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE must be between 0 and 1 (exclusive), got {DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE}"
            )

        if test_df is None:
            logger.debug("Test dataset not found, creating split")
            try:
                final_train_df = train_df.sample(
                    frac=DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE, random_state=42
                )
                final_test_df = train_df.drop(index=final_train_df.index.tolist())
            except Exception as e:
                logger.error(f"Failed to create train/test split: {e}")
                raise ValueError(f"Train/test split failed: {e}") from e
        else:
            logger.debug("Test dataset found")

            if test_df.empty:
                raise ValueError("test_df cannot be empty")

            final_train_df = train_df
            final_test_df = test_df

        if final_train_df.empty:
            raise ValueError("Final training DataFrame is empty after split")

        if final_test_df.empty:
            raise ValueError("Final test DataFrame is empty after split")

        return final_train_df, final_test_df
