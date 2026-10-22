"""Shared fixtures for the end-to-end modality tests.

These tests exercise the whole request path of the non-tabular services:
HTTP form → router → orchestrator → shared ZIP extraction → per-modality
validation → consolidated ML engine (real Optuna search + Lightning Fabric
training on tiny local transformers) → trial-artifact packaging → AutoDW
upload payload.

Exactly two boundaries are stubbed:

* the AutoDW network calls (metadata fetch, dataset download, model upload).
  The download stub writes a *real* ZIP dataset, so extraction and validation
  run for real; the upload stub records what would have been sent.
* Hugging Face hub loading. Every ``from_pretrained`` entry point the ML
  engine uses is replaced with a tiny, locally-built stand-in (fresh random
  weights per construction), so no network access is needed and training
  stays fast.

Training-related fixtures force CPU and zero dataloader workers so runs are
deterministic and cheap on any machine.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    ASTConfig,
    ASTFeatureExtractor,
    ASTForAudioClassification,
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
    ViTConfig,
    ViTForImageClassification,
    ViTImageProcessor,
)
from tokenizers import Tokenizer, models, pre_tokenizers

from tests.test_e2e.artifacts import text_vocab

# Loader-only kwargs forwarded by transformers' from_pretrained that tiny
# config/model constructors must ignore.
_LOADER_ONLY_KWARGS = (
    "ignore_mismatched_sizes",
    "_fast_init",
    "low_cpu_mem_usage",
    "device_map",
)

_TINY_VIT = {
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "image_size": 32,
    "patch_size": 16,
    "num_channels": 3,
}

_TINY_BERT = {
    "vocab_size": len(text_vocab()),
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 64,
}

_TINY_AST = {
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "num_mel_bins": 32,
    "max_length": 50,
}


def _clean_kwargs(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k not in _LOADER_ONLY_KWARGS}


def _tiny_vit_config(**kwargs) -> ViTConfig:
    return ViTConfig(**_TINY_VIT, **_clean_kwargs(kwargs))


def _tiny_bert_config(**kwargs) -> BertConfig:
    return BertConfig(**_TINY_BERT, **_clean_kwargs(kwargs))


def _tiny_ast_config(**kwargs) -> ASTConfig:
    return ASTConfig(**_TINY_AST, **_clean_kwargs(kwargs))


def _tiny_fast_tokenizer() -> PreTrainedTokenizerFast:
    """Whitespace WordLevel tokenizer covering the e2e text corpus."""
    tk = Tokenizer(models.WordLevel(vocab=text_vocab(), unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="[UNK]",
        pad_token="[PAD]",
        model_max_length=64,
    )


# ---------------------------------------------------------------------------
# Autouse environment fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cpu_training(monkeypatch):
    """Force CPU training and inline (zero-worker) dataloaders.

    Keeps the runs deterministic and cheap regardless of host hardware (no
    MPS/CUDA, no dataloader worker processes) and matches what CI gets.
    """
    monkeypatch.setenv("AUTO_NUM_WORKERS_MAX", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


@pytest.fixture(autouse=True)
def tiny_hf_hub(monkeypatch):
    """Replace every Hugging Face hub entry point the ML engine uses.

    Patching the ``transformers`` classes themselves covers both module-level
    imports (``app.ml_engine.model``, ``app.ml_engine.datamodule``) and
    call-time imports (e.g. ``AutoConfig`` inside the multimodal model).
    Each construction yields fresh random weights — identity/quality of the
    model is irrelevant, only that real training code runs.
    """
    import transformers

    def _fake_auto_config(pretrained_model_name_or_path=None, **kwargs):
        return _tiny_vit_config(**kwargs)

    def _fake_image_cls_model(
        pretrained_model_name_or_path=None, config=None, **kwargs
    ):
        return ViTForImageClassification(config or _tiny_vit_config(**kwargs))

    def _fake_seq_cls_model(pretrained_model_name_or_path=None, config=None, **kwargs):
        return BertForSequenceClassification(config or _tiny_bert_config(**kwargs))

    def _fake_audio_cls_model(
        pretrained_model_name_or_path=None, config=None, **kwargs
    ):
        return ASTForAudioClassification(config or _tiny_ast_config(**kwargs))

    def _fake_image_processor(pretrained_model_name_or_path=None, **kwargs):
        return ViTImageProcessor(size={"height": 32, "width": 32})

    def _fake_feature_extractor(pretrained_model_name_or_path=None, **kwargs):
        return ASTFeatureExtractor(sampling_rate=16000, num_mel_bins=32, max_length=50)

    def _fake_tokenizer(pretrained_model_name_or_path=None, **kwargs):
        return _tiny_fast_tokenizer()

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _fake_auto_config)
    monkeypatch.setattr(
        transformers.AutoModelForImageClassification,
        "from_pretrained",
        _fake_image_cls_model,
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        _fake_seq_cls_model,
    )
    monkeypatch.setattr(
        transformers.AutoModelForAudioClassification,
        "from_pretrained",
        _fake_audio_cls_model,
    )
    monkeypatch.setattr(
        transformers.AutoImageProcessor, "from_pretrained", _fake_image_processor
    )
    monkeypatch.setattr(
        transformers.AutoFeatureExtractor,
        "from_pretrained",
        _fake_feature_extractor,
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _fake_tokenizer)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zip_metadata() -> dict:
    """AutoDW metadata for a folder-backed (ZIP download) dataset."""
    return {
        "file_type": "zip",
        "original_filename": "dataset.zip",
        "version": "v1",
        "custom_metadata": {},
    }


# Endpoint package name per URL service segment.
_SERVICE_MODULES = {
    "vision": "vision_automl",
    "audio": "audio_automl",
    "text": "text_automl",
}


@pytest.fixture
def stub_autodw(monkeypatch):
    """Patch the AutoDW network boundary in a service orchestrator's namespace.

    ``fetch_dataset_metadata`` returns the given metadata; ``download_dataset``
    writes a real ZIP so extraction/validation run for real; ``upload_model``
    records its arguments and reports the given HTTP status. Everything else
    (URL resolution, extraction, validation, training, serialization, upload
    payload building) runs for real.

    Returns an installer that yields the recorded calls.
    """

    def _install(
        service: str,
        zip_bytes: bytes,
        metadata: dict,
        upload_status: int = 200,
    ) -> dict:
        calls: dict = {}
        orchestrator = importlib.import_module(
            f"app.{_SERVICE_MODULES[service]}.orchestrator"
        )

        monkeypatch.setattr(
            orchestrator,
            "fetch_dataset_metadata",
            lambda *args, **kwargs: metadata,
        )

        def _fake_download(
            download_url: str, workdir: Path, original_filename: str
        ) -> Path:
            dest = Path(workdir) / original_filename
            dest.write_bytes(zip_bytes)
            calls["download_url"] = download_url
            return dest

        monkeypatch.setattr(orchestrator, "download_dataset", _fake_download)

        def _fake_upload(
            upload_url: str, zip_path: Path, payload: dict, task_id: str | None
        ) -> SimpleNamespace:
            calls.update(
                upload_url=upload_url,
                # The orchestrator deletes its workspace after the request, so
                # snapshot the archive content the way a real upload would.
                zip_bytes=zip_path.read_bytes(),
                payload=payload,
                task_id=task_id,
            )
            # AutoDW reports the created model version in its JSON body.
            return SimpleNamespace(
                status_code=upload_status,
                text="stub ok",
                json=lambda: {"version": "v1"},
            )

        monkeypatch.setattr(orchestrator, "upload_model", _fake_upload)
        return calls

    return _install


@pytest.fixture
def vision_zip() -> bytes:
    from tests.test_e2e.artifacts import vision_zip_bytes

    return vision_zip_bytes()


@pytest.fixture
def multimodal_zip() -> bytes:
    from tests.test_e2e.artifacts import multimodal_zip_bytes

    return multimodal_zip_bytes()


@pytest.fixture
def audio_zip() -> bytes:
    from tests.test_e2e.artifacts import audio_zip_bytes

    return audio_zip_bytes()


@pytest.fixture
def text_zip() -> bytes:
    from tests.test_e2e.artifacts import text_zip_bytes

    return text_zip_bytes()


@pytest.fixture
def broken_vision_zip() -> bytes:
    from tests.test_e2e.artifacts import vision_zip_with_missing_image_bytes

    return vision_zip_with_missing_image_bytes()
