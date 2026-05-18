from pathlib import Path
from typing import Any, Callable

import optuna
from optuna.distributions import CategoricalChoiceType
from torch import nn

from app.vision_automl.ml_engine import (
    EarlyStopping,
    FabricTrainer,
    MultimodalClassificationDataModule,
    MultimodalClassificationModel,
)
from app.vision_automl.ml_engine.datamodule import (
    AudioClassificationDataModule,
    CausalLMDataModule,
    ImageClassificationDataModule,
    ImageSegmentationDataModule,
    KeypointDetectionDataModule,
    MaskedLMDataModule,
    ObjectDetectionDataModule,
    QuestionAnsweringDataModule,
    Seq2SeqLMDataModule,
    SequenceClassificationDataModule,
    VideoClassificationDataModule,
)
from app.vision_automl.ml_engine.model import (
    AudioClassificationModel,
    CausalLMModel,
    ImageClassificationModel,
    ImageSegmentationModel,
    KeypointDetectionModel,
    MaskedLMModel,
    ObjectDetectionModel,
    QuestionAnsweringModel,
    Seq2SeqLMModel,
    SequenceClassificationModel,
    VideoClassificationModel,
)

# ---------------------------------------------------------------------------
# Per-task Optuna objective functions
# ---------------------------------------------------------------------------


def _optuna_objective_base(
    trial: optuna.Trial,
    *,
    model_size: str,
    config: dict[str, Any],
    datamodule_class: type,
    dm_kwargs: dict[str, Any],
    model_class: type[nn.Module],
    build_model_kwargs: Callable[[Any, str], dict[str, Any]],
    model_computes_loss: bool = False,
    loss_fn: nn.Module | None = None,
) -> float:
    models: list[str] = config[f"{model_size}_models"]
    model_id: CategoricalChoiceType = trial.suggest_categorical("model_id", models)
    lr: float = trial.suggest_float("lr", config["lr_low"], config["lr_high"], log=True)
    batch_size: CategoricalChoiceType = trial.suggest_categorical(
        "batch_size", config["batch_sizes"]
    )
    weight_decay: float = trial.suggest_float(
        "weight_decay",
        config["weight_decay_low"],
        config["weight_decay_high"],
        log=True,
    )

    datamodule = datamodule_class(
        **dm_kwargs,
        batch_size=batch_size,
        hf_model_id=model_id,
    )

    trainer_kwargs: dict[str, Any] = {
        "datamodule": datamodule,
        "model_class": model_class,
        "model_kwargs": build_model_kwargs(datamodule, model_id),
        "optimizer_kwargs": {"lr": lr, "weight_decay": weight_decay},
        "epochs": config["max_epochs"],
        "callbacks": [EarlyStopping(patience=config["early_stopping_patience"])],
        "model_computes_loss": model_computes_loss,
    }
    if loss_fn is not None:
        trainer_kwargs["loss_fn"] = loss_fn

    trainer = FabricTrainer(**trainer_kwargs)
    test_loss: float
    test_loss, _ = trainer.fit(trial=trial)
    return test_loss


def optuna_objective_image_classification(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
) -> float:
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        config=config,
        datamodule_class=ImageClassificationDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "root_dir": images_dir,
            "img_col": filename_column,
            "label_col": label_column,
        },
        model_class=ImageClassificationModel,
        build_model_kwargs=lambda dm, mid: {
            "model_id": mid,
            "num_classes": dm.num_classes,
            "id2label": dm.id2label,
            "label2id": dm.label2id,
        },
        model_computes_loss=False,
        loss_fn=nn.CrossEntropyLoss(),
    )


def optuna_objective_image_classification_multimodal(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    auxiliary_columns: list[str],
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
) -> float:
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        config=config,
        datamodule_class=MultimodalClassificationDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "root_dir": images_dir,
            "img_col": filename_column,
            "label_col": label_column,
            "auxiliary_columns": auxiliary_columns,
        },
        model_class=MultimodalClassificationModel,
        build_model_kwargs=lambda dm, mid: {
            "model_id": mid,
            "num_classes": dm.num_classes,
            "aux_feature_dim": dm.aux_feature_dim,
            "id2label": dm.id2label,
            "label2id": dm.label2id,
        },
        model_computes_loss=False,
        loss_fn=nn.CrossEntropyLoss(),
    )


def optuna_objective_image_segmentation(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
) -> float:
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        config=config,
        datamodule_class=ImageSegmentationDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "root_dir": images_dir,
            "img_col": filename_column,
            "label_col": label_column,
        },
        model_class=ImageSegmentationModel,
        build_model_kwargs=lambda dm, mid: {
            "model_id": mid,
            "num_classes": dm.num_classes,
            "id2label": dm.id2label,
            "label2id": dm.label2id,
        },
        model_computes_loss=True,
    )


def optuna_objective_object_detection(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
) -> float:
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        config=config,
        datamodule_class=ObjectDetectionDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "root_dir": images_dir,
            "img_col": filename_column,
        },
        model_class=ObjectDetectionModel,
        build_model_kwargs=lambda dm, mid: {
            "model_id": mid,
            "num_classes": dm.num_classes,
        },
        model_computes_loss=True,
    )


def optuna_objective_video_classification(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
) -> float:
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        config=config,
        datamodule_class=VideoClassificationDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "root_dir": images_dir,
            "video_col": filename_column,
            "label_col": label_column,
        },
        model_class=VideoClassificationModel,
        build_model_kwargs=lambda dm, mid: {
            "model_id": mid,
            "num_classes": dm.num_classes,
            "id2label": dm.id2label,
            "label2id": dm.label2id,
        },
        model_computes_loss=False,
        loss_fn=nn.CrossEntropyLoss(),
    )


def optuna_objective_keypoint_detection(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
) -> float:
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        config=config,
        datamodule_class=KeypointDetectionDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "root_dir": images_dir,
            "img_col": filename_column,
            "label_col": label_column,
        },
        model_class=KeypointDetectionModel,
        build_model_kwargs=lambda dm, mid: {"model_id": mid},
        model_computes_loss=True,
    )


def optuna_objective_audio_classification(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    audio_dir: Path,
    filename_column: str,
    label_column: str,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
) -> float:
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        config=config,
        datamodule_class=AudioClassificationDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "root_dir": audio_dir,
            "audio_col": filename_column,
            "label_col": label_column,
        },
        model_class=AudioClassificationModel,
        build_model_kwargs=lambda dm, mid: {
            "model_id": mid,
            "num_classes": dm.num_classes,
            "id2label": dm.id2label,
            "label2id": dm.label2id,
        },
        model_computes_loss=False,
        loss_fn=nn.CrossEntropyLoss(),
    )


def optuna_objective_text_classification(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    text_column: str,
    label_column: str,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    **_kwargs: Any,
) -> float:
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        config=config,
        datamodule_class=SequenceClassificationDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "text_col": text_column,
            "label_col": label_column,
        },
        model_class=SequenceClassificationModel,
        build_model_kwargs=lambda dm, mid: {
            "model_id": mid,
            "num_classes": dm.num_classes,
            "id2label": dm.id2label,
            "label2id": dm.label2id,
        },
        model_computes_loss=False,
        loss_fn=nn.CrossEntropyLoss(),
    )


def optuna_objective_question_answering(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    **_kwargs: Any,
) -> float:
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        config=config,
        datamodule_class=QuestionAnsweringDataModule,
        dm_kwargs={"csv_file": csv_path},
        model_class=QuestionAnsweringModel,
        build_model_kwargs=lambda dm, mid: {"model_id": mid},
        model_computes_loss=True,
    )


def optuna_objective_causal_lm(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    **_kwargs: Any,
) -> float:
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        config=config,
        datamodule_class=CausalLMDataModule,
        dm_kwargs={"csv_file": csv_path},
        model_class=CausalLMModel,
        build_model_kwargs=lambda dm, mid: {"model_id": mid},
        model_computes_loss=True,
    )


def optuna_objective_seq2seq_lm(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    **_kwargs: Any,
) -> float:
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        config=config,
        datamodule_class=Seq2SeqLMDataModule,
        dm_kwargs={"csv_file": csv_path},
        model_class=Seq2SeqLMModel,
        build_model_kwargs=lambda dm, mid: {"model_id": mid},
        model_computes_loss=True,
    )


def optuna_objective_masked_lm(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    **_kwargs: Any,
) -> float:
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        config=config,
        datamodule_class=MaskedLMDataModule,
        dm_kwargs={"csv_file": csv_path},
        model_class=MaskedLMModel,
        build_model_kwargs=lambda dm, mid: {"model_id": mid},
        model_computes_loss=True,
    )


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

OBJECTIVE_REGISTRY: dict[str, Callable] = {
    "image_classification": optuna_objective_image_classification,
    "image_classification_multimodal": optuna_objective_image_classification_multimodal,
    "image_segmentation": optuna_objective_image_segmentation,
    "object_detection": optuna_objective_object_detection,
    "video_classification": optuna_objective_video_classification,
    "keypoint_detection": optuna_objective_keypoint_detection,
    "audio_classification": optuna_objective_audio_classification,
    "text_classification": optuna_objective_text_classification,
    "question_answering": optuna_objective_question_answering,
    "causal_lm": optuna_objective_causal_lm,
    "seq2seq_lm": optuna_objective_seq2seq_lm,
    "masked_lm": optuna_objective_masked_lm,
}
optuna_objective = optuna_objective_image_classification
