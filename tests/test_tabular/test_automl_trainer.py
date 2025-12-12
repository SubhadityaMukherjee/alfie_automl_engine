import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from app.tabular_automl.modules import AutoMLTrainer


@pytest.fixture
def trainer_class():
    tmpdir = tempfile.mkdtemp()
    trainer = AutoMLTrainer(save_model_path=tmpdir)
    yield trainer
    shutil.rmtree(tmpdir)  # cleanup after test


def test_check_if_save_model_path_exists(trainer_class: AutoMLTrainer):
    assert type(str(trainer_class.save_model_path)) == str


@pytest.fixture
def small_df():
    return pd.DataFrame.from_dict(
        {"feature": [1, 2, 3, 4, 5], "target": [1, 1, 0, 0, 1]}
    )


@pytest.mark.parametrize(
    "test_df, expected_train_shape, expected_test_shape",
    [
        (None, (4, 2), (1, 2)),
        ("same", (5, 2), (5, 2)),
    ],
)
def test_train_test_split(
    trainer_class, small_df, test_df, expected_train_shape, expected_test_shape
):
    if test_df == "same":
        test_df = small_df
    final_train_df, final_test_df = trainer_class.train_test_split(
        train_df=small_df, test_df=test_df
    )
    assert isinstance(final_train_df, pd.DataFrame)
    assert isinstance(final_test_df, pd.DataFrame)
    assert final_train_df.shape == expected_train_shape
    assert final_test_df.shape == expected_test_shape


def test_train_leaderboard_works(trainer_class: AutoMLTrainer, small_df: pd.DataFrame):
    leaderboard, _ = trainer_class.train(
        train_df=small_df, test_df=None, target_column="target", time_limit=20
    )
    for key in ["model", "eval_metric", "score_val", "score_test", "fit_time"]:
        assert key in leaderboard


def test_train_leaderboard_exception(
    trainer_class: AutoMLTrainer, small_df: pd.DataFrame
):
    with pytest.raises(KeyError, match="missing"):
        trainer_class.train(
            train_df=small_df, test_df=None, target_column="wrong_target", time_limit=2
        )
