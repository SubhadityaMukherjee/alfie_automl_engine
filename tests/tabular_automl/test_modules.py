import tempfile

import pandas as pd
import pytest

from app.tabular_automl.modules import AutoMLTrainer

tempdir = tempfile.mkdtemp()
trainer = AutoMLTrainer(tempdir)


@pytest.fixture
def train_df():
    return pd.DataFrame.from_records(
        [
            [1, 2, 3],
            [1, 2, 3],
            [3, 4, 5],
            [1, 2, 3],
            [3, 4, 5],
            [1, 2, 3],
            [3, 4, 5],
            [1, 2, 3],
            [3, 4, 5],
            [1, 2, 3],
            [3, 4, 5],
        ]
    )


@pytest.fixture
def test_df():
    return pd.DataFrame.from_records(
        [
            [1, 2, 3],
            [1, 2, 3],
            [3, 4, 5],
        ]
    )


class TestAutoMLTrainer:
    def test_train_test_split_given_no_test(self, train_df):
        final_train_df, final_test_df = trainer.train_test_split(
            train_df=train_df, test_df=None
        )

        # Validate sizes (80/20 split on 11 rows -> 8 train, 3 test)
        assert len(final_train_df) + len(final_test_df) == len(train_df)

        # Validate disjoint indices and full coverage
        assert final_train_df.index.intersection(final_test_df.index).empty
        reconstructed = pd.concat([final_train_df, final_test_df]).sort_index()
        assert reconstructed.reset_index(drop=True).equals(
            train_df.sort_index().reset_index(drop=True)
        )

    def test_train_test_split_given_test(self, train_df, test_df):
        final_train_df, final_test_df = trainer.train_test_split(
            train_df=train_df, test_df=test_df
        )

        # Validate sizes are preserved
        assert len(final_train_df) == len(train_df)
        assert len(final_test_df) == len(test_df)

        # Validate DataFrames are returned unchanged
        assert final_test_df.equals(test_df)
        assert final_train_df.equals(train_df)
