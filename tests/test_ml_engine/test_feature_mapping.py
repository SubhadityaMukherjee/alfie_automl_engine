"""Tests for app/ml_engine/feature_mapping.py."""

from app.ml_engine.feature_mapping import extract_feature_mapping


class _FakeTextDatamodule:
    """Stand-in for a text-task datamodule post-setup."""

    def __init__(self):
        self.hf_model_id = "distilbert-base-uncased"
        self.id2label = {0: "neg", 1: "pos"}
        self.label2id = {"neg": 0, "pos": 1}

        class _FakeTokenizer:
            def get_vocab(self):
                return {"hello": 0, "world": 1, "[PAD]": 2}

        self.tokenizer = _FakeTokenizer()


class _FakeScaler:
    n_features_in_ = 2
    mean_ = __import__("numpy").array([0.1, 0.2])
    scale_ = __import__("numpy").array([1.0, 2.0])
    var_ = __import__("numpy").array([1.0, 4.0])


class _FakeEncoder:
    categories_ = [
        __import__("numpy").array(["a", "b", "c"]),
        __import__("numpy").array(["x", "y"]),
    ]


class _FakeMultimodalDatamodule:
    """Stand-in for a fitted MultimodalClassificationDataModule."""

    def __init__(self):
        self.hf_model_id = "google/vit-base-patch16-224"
        self.id2label = {0: "cat", 1: "dog"}
        self.label2id = {"cat": 0, "dog": 1}
        self.auxiliary_columns = ["age", "city"]
        self.numeric_cols = ["age"]
        self.categorical_cols = ["city"]
        self.aux_feature_dim = 2
        self.scaler = _FakeScaler()
        self.encoder = _FakeEncoder()


class _FakeImageDatamodule:
    """Pure-image datamodule: only label maps should be extracted."""

    def __init__(self):
        self.hf_model_id = "google/vit-base-patch16-224"
        self.id2label = {0: "cat", 1: "dog"}
        self.label2id = {"cat": 0, "dog": 1}


class _FakeMultimodalModel:
    def _get_vision_embed_dim(self):
        return 768


def test_extract_feature_mapping_text_task_includes_vocab():
    dm = _FakeTextDatamodule()
    out = extract_feature_mapping(dm, "text_classification")
    assert out["task_type"] == "text_classification"
    assert out["label_map"]["id2label"] == {"0": "neg", "1": "pos"}
    assert out["tokenizer"]["hf_model_id"] == "distilbert-base-uncased"
    assert out["tokenizer"]["vocab"]["hello"] == 0
    assert "[PAD]" in out["tokenizer"]["vocab"]
    assert "auxiliary_features" not in out


def test_extract_feature_mapping_multimodal_includes_preprocessing():
    dm = _FakeMultimodalDatamodule()
    model = _FakeMultimodalModel()
    out = extract_feature_mapping(dm, "image_classification_multimodal", model=model)
    aux = out["auxiliary_features"]
    assert aux["auxiliary_columns"] == ["age", "city"]
    assert aux["numeric_columns"] == ["age"]
    assert aux["categorical_columns"] == ["city"]
    assert aux["aux_feature_dim"] == 2
    assert aux["scaler"]["mean"] == [0.1, 0.2]
    assert aux["scaler"]["scale"] == [1.0, 2.0]
    assert aux["ordinal_encoder"]["categories"] == [
        ["a", "b", "c"],
        ["x", "y"],
    ]
    assert aux["vision_embed_dim"] == 768
    assert "tokenizer" not in out


def test_extract_feature_mapping_image_only_returns_label_map_only():
    dm = _FakeImageDatamodule()
    out = extract_feature_mapping(dm, "image_classification")
    assert out["task_type"] == "image_classification"
    assert out["label_map"]["id2label"] == {"0": "cat", "1": "dog"}
    assert "tokenizer" not in out
    assert "auxiliary_features" not in out


def test_extract_feature_mapping_label_map_empty_when_missing():
    class _Bare:
        pass

    out = extract_feature_mapping(_Bare(), "image_classification")
    assert out["label_map"] == {}


def test_extract_feature_mapping_resilient_to_failing_model():
    dm = _FakeMultimodalDatamodule()

    class _Boom:
        def _get_vision_embed_dim(self):
            raise RuntimeError("boom")

    out = extract_feature_mapping(dm, "image_classification_multimodal", model=_Boom())
    # vision_embed_dim skipped, but the rest of the section is still there
    assert "vision_embed_dim" not in out["auxiliary_features"]
    assert out["auxiliary_features"]["scaler"]["mean"] == [0.1, 0.2]
