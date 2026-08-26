"""Tests for app/ml_engine/model.py."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from app.ml_engine.model import ClassificationModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_hf_model(num_classes: int = 2):
    """Return a mock HF model with a classifier head and a forward that returns logits."""
    hf_model = MagicMock(spec=nn.Module)

    # parameters() — backbone params (frozen) + classifier params (trainable)
    backbone_param = nn.Parameter(torch.zeros(4, 4), requires_grad=True)
    classifier_param = nn.Parameter(torch.zeros(2, 4), requires_grad=True)

    # Make .parameters() return both params
    hf_model.parameters.return_value = iter([backbone_param, classifier_param])

    # classifier sub-module whose parameters() returns just the classifier param
    classifier_mock = MagicMock()
    classifier_mock.parameters.return_value = iter([classifier_param])
    hf_model.classifier = classifier_mock

    # forward call returns an object with .logits
    logits = torch.randn(1, num_classes)
    hf_model.return_value = MagicMock(logits=logits)

    return hf_model


@pytest.fixture
def mock_hf_model():
    return _make_mock_hf_model(num_classes=2)


@pytest.fixture
def classification_model(mock_hf_model):
    with patch(
        "app.ml_engine.model.AutoModelForImageClassification.from_pretrained",
        return_value=mock_hf_model,
    ):
        model = ClassificationModel(
            model_id="google/vit-base-patch16-224",
            num_classes=2,
            freeze_backbone=True,
            id2label={0: "cat", 1: "dog"},
            label2id={"cat": 0, "dog": 1},
        )
    return model


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_is_nn_module(classification_model):
    assert isinstance(classification_model, nn.Module)


def test_from_pretrained_called_with_correct_model_id():
    model_id = "google/efficientnet-b0"
    with patch(
        "app.ml_engine.model.AutoModelForImageClassification.from_pretrained",
    ) as mock_pretrained:
        mock_pretrained.return_value = _make_mock_hf_model(num_classes=3)
        ClassificationModel(model_id=model_id, num_classes=3)

    call_kwargs = mock_pretrained.call_args
    assert call_kwargs[0][0] == model_id


def test_from_pretrained_receives_num_labels():
    with patch(
        "app.ml_engine.model.AutoModelForImageClassification.from_pretrained",
    ) as mock_pretrained:
        mock_pretrained.return_value = _make_mock_hf_model(num_classes=5)
        ClassificationModel(num_classes=5)

    _, kwargs = mock_pretrained.call_args
    assert kwargs["num_labels"] == 5


def test_default_id2label_built_when_not_provided():
    with patch(
        "app.ml_engine.model.AutoModelForImageClassification.from_pretrained",
    ) as mock_pretrained:
        mock_pretrained.return_value = _make_mock_hf_model(num_classes=3)
        ClassificationModel(num_classes=3)

    _, kwargs = mock_pretrained.call_args
    assert kwargs["id2label"] == {0: "0", 1: "1", 2: "2"}


def test_default_label2id_built_when_not_provided():
    with patch(
        "app.ml_engine.model.AutoModelForImageClassification.from_pretrained",
    ) as mock_pretrained:
        mock_pretrained.return_value = _make_mock_hf_model(num_classes=3)
        ClassificationModel(num_classes=3)

    _, kwargs = mock_pretrained.call_args
    assert kwargs["label2id"] == {"0": 0, "1": 1, "2": 2}


def test_custom_label_maps_forwarded_to_pretrained():
    id2label = {0: "cat", 1: "dog"}
    label2id = {"cat": 0, "dog": 1}
    with patch(
        "app.ml_engine.model.AutoModelForImageClassification.from_pretrained",
    ) as mock_pretrained:
        mock_pretrained.return_value = _make_mock_hf_model(num_classes=2)
        ClassificationModel(num_classes=2, id2label=id2label, label2id=label2id)

    _, kwargs = mock_pretrained.call_args
    assert kwargs["id2label"] == id2label
    assert kwargs["label2id"] == label2id


def test_ignore_mismatched_sizes_is_true():
    with patch(
        "app.ml_engine.model.AutoModelForImageClassification.from_pretrained",
    ) as mock_pretrained:
        mock_pretrained.return_value = _make_mock_hf_model()
        ClassificationModel()

    _, kwargs = mock_pretrained.call_args
    assert kwargs["ignore_mismatched_sizes"] is True


# ---------------------------------------------------------------------------
# Backbone freezing
# ---------------------------------------------------------------------------


def test_freeze_backbone_freezes_all_params_except_classifier():
    """When freeze_backbone=True all backbone params get requires_grad=False."""
    backbone_param = nn.Parameter(torch.zeros(4, 4), requires_grad=True)
    classifier_param = nn.Parameter(torch.zeros(2, 4), requires_grad=True)

    hf_model = MagicMock(spec=nn.Module)
    hf_model.parameters.return_value = iter([backbone_param, classifier_param])

    classifier_mock = MagicMock()
    classifier_mock.parameters.return_value = iter([classifier_param])
    hf_model.classifier = classifier_mock
    hf_model.return_value = MagicMock(logits=torch.randn(1, 2))

    with patch(
        "app.ml_engine.model.AutoModelForImageClassification.from_pretrained",
        return_value=hf_model,
    ):
        ClassificationModel(freeze_backbone=True)

    # All params should be frozen initially, then classifier unfrozen
    assert not backbone_param.requires_grad
    assert classifier_param.requires_grad


def test_no_freeze_when_freeze_backbone_false():
    """When freeze_backbone=False parameters() is never called for freezing."""
    backbone_param = nn.Parameter(torch.zeros(4, 4), requires_grad=True)

    hf_model = MagicMock(spec=nn.Module)
    hf_model.parameters.return_value = iter([backbone_param])
    hf_model.return_value = MagicMock(logits=torch.randn(1, 2))

    with patch(
        "app.ml_engine.model.AutoModelForImageClassification.from_pretrained",
        return_value=hf_model,
    ):
        ClassificationModel(freeze_backbone=False)

    # param should remain trainable
    assert backbone_param.requires_grad


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def test_forward_returns_logits_tensor(classification_model):
    pixel_values = torch.randn(2, 3, 224, 224)
    output = classification_model(pixel_values)
    assert isinstance(output, torch.Tensor)


def test_forward_calls_inner_model_with_pixel_values(classification_model):
    pixel_values = torch.randn(2, 3, 224, 224)
    classification_model(pixel_values)
    classification_model.model.assert_called_once_with(pixel_values)


def test_forward_output_shape_matches_num_classes():
    num_classes = 4
    logits = torch.randn(2, num_classes)
    hf_model = MagicMock(spec=nn.Module)
    hf_model.parameters.return_value = iter([])
    hf_model.return_value = MagicMock(logits=logits)

    with patch(
        "app.ml_engine.model.AutoModelForImageClassification.from_pretrained",
        return_value=hf_model,
    ):
        model = ClassificationModel(num_classes=num_classes, freeze_backbone=False)

    pixel_values = torch.randn(2, 3, 224, 224)
    output = model(pixel_values)
    assert output.shape == (2, num_classes)
