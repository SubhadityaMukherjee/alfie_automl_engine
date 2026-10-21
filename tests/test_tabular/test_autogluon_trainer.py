import shutil
import tempfile

import pandas as pd
import pytest

from app.core.exceptions import AutoMLConfigError, AutoMLDataError
from app.ml_engine.trainer import AutoGluonTrainer, _compute_tabular_test_metrics


class _FakeTabularPredictor:
    """Minimal AutoGluon predictor stand-in for metric-suite tests."""

    def __init__(self, problem_type, proba=None, pred=None, **attrs):
        self.problem_type = problem_type
        self._proba = proba
        self._pred = pred
        for key, value in attrs.items():
            setattr(self, key, value)

    def predict_proba(self, data):
        if self._proba is None:
            raise RuntimeError("no probabilities available")
        return self._proba

    def predict(self, data):
        if self._pred is None:
            raise RuntimeError("no predictions available")
        return self._pred


@pytest.fixture
def trainer_class():
    tmpdir = tempfile.mkdtemp()
    trainer = AutoGluonTrainer(save_model_path=tmpdir)
    yield trainer
    shutil.rmtree(tmpdir)  # cleanup after test


def test_check_if_save_model_path_exists(trainer_class: AutoGluonTrainer):
    assert isinstance(str(trainer_class.save_model_path), str)


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


@pytest.mark.parametrize(
    "train_df,test_df,expected_error", [(None, None, AutoMLDataError)]
)
def test_train_test_split_raises_exceptions(
    trainer_class, train_df, test_df, expected_error
):
    with pytest.raises(expected_error):
        trainer_class.train_test_split(train_df=train_df, test_df=test_df)


@pytest.mark.parametrize(
    "time_limit,expected_error",
    [
        ("10", AutoMLConfigError),
        ("target", AutoMLConfigError),
        (0, AutoMLConfigError),
        (-1, AutoMLConfigError),
    ],
)
@pytest.mark.slow
def test_train_invalid_time(
    trainer_class,
    small_df,
    time_limit,
    expected_error,
):
    with pytest.raises(expected_error):
        trainer_class.train(
            train_df=small_df,
            test_df=small_df,
            target_column="target",
            time_limit=time_limit,
        )


@pytest.mark.parametrize(
    "train_df,test_df,target_column,time_limit,expected_error",
    [
        (None, None, "target", 10, AutoMLDataError),
        (None, None, None, 10, AutoMLDataError),
    ],
)
@pytest.mark.slow
def test_train_invalid_inputs(
    trainer_class,
    train_df,
    test_df,
    target_column,
    time_limit,
    expected_error,
):
    with pytest.raises(expected_error):
        trainer_class.train(
            train_df=train_df,
            test_df=test_df,
            target_column=target_column,
            time_limit=time_limit,
        )


@pytest.mark.parametrize(
    "target_column,expected_error",
    [
        (None, AutoMLDataError),
        ("no_target", AutoMLDataError),
    ],
)
@pytest.mark.slow
def test_train_invalid_target_column(
    trainer_class,
    small_df,
    target_column,
    expected_error,
):
    with pytest.raises(expected_error):
        trainer_class.train(
            train_df=small_df,
            test_df=small_df,
            target_column=target_column,
            time_limit=10,
        )


@pytest.mark.slow
def test_train_leaderboard_works(
    trainer_class: AutoGluonTrainer, small_df: pd.DataFrame
):
    leaderboard, _, test_metrics = trainer_class.train(
        train_df=small_df, test_df=None, target_column="target", time_limit=20
    )
    for key in ["model", "eval_metric", "score_val", "score_test", "fit_time"]:
        assert key in leaderboard
    # Binary target -> the test-set metric suite is computed.
    assert "accuracy" in test_metrics
    assert 0.0 <= test_metrics["accuracy"] <= 1.0


def test_train_leaderboard_exception(
    trainer_class: AutoGluonTrainer, small_df: pd.DataFrame
):
    with pytest.raises(AutoMLDataError):
        trainer_class.train(
            train_df=small_df, test_df=None, target_column="wrong_target", time_limit=2
        )


# ---------------------------------------------------------------------------
# _compute_tabular_test_metrics
# ---------------------------------------------------------------------------


def test_tabular_test_metrics_binary():
    test_df = pd.DataFrame({"target": [0, 1, 1]})
    proba = pd.DataFrame({"0": [0.9, 0.3, 0.2], "1": [0.1, 0.7, 0.8]})
    predictor = _FakeTabularPredictor(
        "binary", proba=proba, positive_class=1, class_labels=["0", "1"]
    )

    metrics = _compute_tabular_test_metrics(predictor, test_df, "target")

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert "f1_macro" in metrics
    assert "matthews_corrcoef" in metrics


def test_tabular_test_metrics_binary_reorders_positive_class_last():
    # Positive class is column 0 (not last): the helper must swap the columns
    # so ROC-AUC uses P(positive), and accuracy must still line up.
    test_df = pd.DataFrame({"target": [0, 1]})
    proba = pd.DataFrame({"0": [0.9, 0.7], "1": [0.1, 0.3]})
    predictor = _FakeTabularPredictor(
        "binary", proba=proba, positive_class=0, class_labels=["0", "1"]
    )

    metrics = _compute_tabular_test_metrics(predictor, test_df, "target")

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_tabular_test_metrics_regression():
    import math

    test_df = pd.DataFrame({"target": [1.0, 2.0, 3.0]})
    predictor = _FakeTabularPredictor("regression", pred=pd.Series([1.0, 2.0, 5.0]))

    metrics = _compute_tabular_test_metrics(predictor, test_df, "target")

    assert metrics["r2"] == pytest.approx(-1.0)
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(math.sqrt(4 / 3))
    assert "mape" in metrics
    assert "median_absolute_error" in metrics
    assert "explained_variance" in metrics


def test_tabular_test_metrics_skips_unknown_problem_type():
    test_df = pd.DataFrame({"target": [1.0, 2.0]})
    predictor = _FakeTabularPredictor("quantile", pred=pd.Series([1.0, 2.0]))

    assert _compute_tabular_test_metrics(predictor, test_df, "target") == {}


def test_tabular_test_metrics_survives_predictor_errors():
    test_df = pd.DataFrame({"target": [0, 1]})
    predictor = _FakeTabularPredictor("binary", proba=None)

    assert _compute_tabular_test_metrics(predictor, test_df, "target") == {}


def test_tabular_test_metrics_excludes_out_of_vocabulary_labels():
    test_df = pd.DataFrame({"target": [0, 1, "unknown"]})
    proba = pd.DataFrame({"0": [0.9, 0.2, 0.5], "1": [0.1, 0.8, 0.5]})
    predictor = _FakeTabularPredictor(
        "binary", proba=proba, positive_class=1, class_labels=["0", "1"]
    )

    metrics = _compute_tabular_test_metrics(predictor, test_df, "target")

    assert metrics["accuracy"] == pytest.approx(1.0)
