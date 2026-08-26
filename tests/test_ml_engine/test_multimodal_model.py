"""Tests for MultimodalClassificationModel."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from app.ml_engine.model import MultimodalClassificationModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_hf_config(vision_dim=768):
    config = MagicMock()
    config.hidden_size = vision_dim
    return config


def _make_mock_hf_model(vision_dim=768, num_classes=2):
    hf_model = MagicMock(spec=nn.Module)

    backbone_param = nn.Parameter(torch.zeros(vision_dim, vision_dim))
    classifier_param = nn.Parameter(torch.zeros(num_classes, vision_dim))

    hf_model.parameters.return_value = iter([backbone_param, classifier_param])

    classifier_mock = MagicMock()
    classifier_mock.in_features = vision_dim
    classifier_mock.parameters.return_value = iter([classifier_param])
    hf_model.classifier = classifier_mock

    hf_model.config = _make_mock_hf_config(vision_dim)

    def _forward(pixel_values):
        batch = pixel_values.shape[0]
        return MagicMock(logits=torch.randn(batch, vision_dim))

    hf_model.side_effect = _forward
    hf_model.return_value = MagicMock(logits=torch.randn(1, vision_dim))

    return hf_model


def _build_model(
    aux_feature_dim=3,
    num_classes=2,
    freeze_backbone=True,
    vision_dim=768,
):
    with (
        patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=_make_mock_hf_config(vision_dim),
        ),
        patch(
            "app.ml_engine.model.AutoModelForImageClassification.from_pretrained",
            return_value=_make_mock_hf_model(vision_dim, num_classes),
        ),
    ):
        model = MultimodalClassificationModel(
            model_id="google/vit-base-patch16-224",
            num_classes=num_classes,
            aux_feature_dim=aux_feature_dim,
            freeze_backbone=freeze_backbone,
            id2label={0: "cat", 1: "dog"},
            label2id={"cat": 0, "dog": 1},
        )
    return model


@pytest.fixture
def multimodal_model():
    return _build_model(aux_feature_dim=3, num_classes=2)


@pytest.fixture
def model_no_aux():
    return _build_model(aux_feature_dim=0, num_classes=2)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_is_nn_module(multimodal_model):
    assert isinstance(multimodal_model, nn.Module)


def test_aux_feature_dim_stored(multimodal_model):
    assert multimodal_model.aux_feature_dim == 3


def test_tabular_mlp_is_sequential_when_aux_gt_zero(multimodal_model):
    assert isinstance(multimodal_model.tabular_mlp, nn.Sequential)


def test_tabular_mlp_is_identity_when_aux_zero(model_no_aux):
    assert isinstance(model_no_aux.tabular_mlp, nn.Identity)


def test_fusion_head_is_sequential(multimodal_model):
    assert isinstance(multimodal_model.fusion_head, nn.Sequential)


def test_from_pretrained_called_with_model_id():
    model_id = "google/efficientnet-b0"
    with (
        patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=_make_mock_hf_config(768),
        ),
        patch(
            "app.ml_engine.model.AutoModelForImageClassification.from_pretrained",
        ) as mock_pretrained,
    ):
        mock_pretrained.return_value = _make_mock_hf_model(768, 2)
        MultimodalClassificationModel(model_id=model_id, num_classes=2)

    call_kwargs = mock_pretrained.call_args
    assert call_kwargs[0][0] == model_id


def test_ignore_mismatched_sizes_is_true():
    with (
        patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=_make_mock_hf_config(768),
        ),
        patch(
            "app.ml_engine.model.AutoModelForImageClassification.from_pretrained",
        ) as mock_pretrained,
    ):
        mock_pretrained.return_value = _make_mock_hf_model(768, 2)
        MultimodalClassificationModel()

    _, kwargs = mock_pretrained.call_args
    assert kwargs["ignore_mismatched_sizes"] is True


# ---------------------------------------------------------------------------
# Backbone freezing
# ---------------------------------------------------------------------------


def test_freeze_backbone_freezes_params_except_classifier():
    backbone_param = nn.Parameter(torch.zeros(4, 4), requires_grad=True)
    classifier_param = nn.Parameter(torch.zeros(2, 4), requires_grad=True)

    hf_model = MagicMock(spec=nn.Module)
    hf_model.parameters.return_value = iter([backbone_param, classifier_param])

    classifier_mock = MagicMock()
    classifier_mock.in_features = 4
    classifier_mock.parameters.return_value = iter([classifier_param])
    hf_model.classifier = classifier_mock
    hf_model.config = _make_mock_hf_config(4)
    hf_model.return_value = MagicMock(logits=torch.randn(1, 4))

    with (
        patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=_make_mock_hf_config(4),
        ),
        patch(
            "app.ml_engine.model.AutoModelForImageClassification.from_pretrained",
            return_value=hf_model,
        ),
    ):
        MultimodalClassificationModel(
            num_classes=2, aux_feature_dim=2, freeze_backbone=True
        )

    assert not backbone_param.requires_grad
    assert classifier_param.requires_grad


def test_no_freeze_when_freeze_backbone_false():
    backbone_param = nn.Parameter(torch.zeros(4, 4), requires_grad=True)

    hf_model = MagicMock(spec=nn.Module)
    hf_model.parameters.return_value = iter([backbone_param])
    hf_model.config = _make_mock_hf_config(4)
    hf_model.return_value = MagicMock(logits=torch.randn(1, 4))

    with (
        patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=_make_mock_hf_config(4),
        ),
        patch(
            "app.ml_engine.model.AutoModelForImageClassification.from_pretrained",
            return_value=hf_model,
        ),
    ):
        MultimodalClassificationModel(
            num_classes=2, aux_feature_dim=0, freeze_backbone=False
        )

    assert backbone_param.requires_grad


# ---------------------------------------------------------------------------
# _get_vision_embed_dim
# ---------------------------------------------------------------------------


def test_get_vision_embed_dim_from_classifier():
    model = _build_model(vision_dim=512)
    assert model._get_vision_embed_dim() == 512


def test_get_vision_embed_dim_from_fc():
    model = _build_model(vision_dim=1024)
    del model.backbone.classifier
    fc_mock = MagicMock()
    fc_mock.in_features = 1024
    model.backbone.fc = fc_mock
    assert model._get_vision_embed_dim() == 1024


def test_get_vision_embed_dim_from_config_hidden_size():
    model = _build_model(vision_dim=256)
    del model.backbone.classifier
    assert model._get_vision_embed_dim() == 256


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def test_forward_returns_logits_tensor(multimodal_model):
    pixel_values = torch.randn(2, 3, 224, 224)
    aux_features = torch.randn(2, 3)
    output = multimodal_model(pixel_values, aux_features)
    assert isinstance(output, torch.Tensor)


def test_forward_output_shape_matches_num_classes_with_aux(multimodal_model):
    pixel_values = torch.randn(2, 3, 224, 224)
    aux_features = torch.randn(2, 3)
    output = multimodal_model(pixel_values, aux_features)
    assert output.shape == (2, 2)


def test_forward_without_aux_features(model_no_aux):
    pixel_values = torch.randn(2, 3, 224, 224)
    output = model_no_aux(pixel_values)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 2)


def test_forward_with_aux_none_uses_vision_only():
    model = _build_model(aux_feature_dim=0, num_classes=2)
    pixel_values = torch.randn(2, 3, 224, 224)
    output = model(pixel_values, aux_features=None)
    assert output.shape == (2, 2)


def test_forward_custom_num_classes():
    model = _build_model(aux_feature_dim=4, num_classes=5, vision_dim=768)
    pixel_values = torch.randn(2, 3, 224, 224)
    aux_features = torch.randn(2, 4)
    output = model(pixel_values, aux_features)
    assert output.shape == (2, 5)
