import torch
from torch import nn
from transformers import (
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForImageSegmentation,
    AutoModelForKeypointDetection,
    AutoModelForMaskedLM,
    AutoModelForObjectDetection,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForVideoClassification,
)


class ImageClassificationModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForImageClassification."""

    def __init__(
        self,
        model_id: str = "google/vit-base-patch16-224",
        num_classes: int = 2,
        freeze_backbone: bool = True,
        id2label: dict | None = None,
        label2id: dict | None = None,
    ):
        super().__init__()
        config_kwargs = {
            "num_labels": num_classes,
            "id2label": id2label or {i: str(i) for i in range(num_classes)},
            "label2id": label2id or {str(i): i for i in range(num_classes)},
        }
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True,
            **config_kwargs,
        )
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            if hasattr(self.model, "classifier"):
                for param in self.model.classifier.parameters():
                    param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values).logits


# Backward-compatibility alias
ClassificationModel = ImageClassificationModel


class ImageSegmentationModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForImageSegmentation."""

    def __init__(
        self,
        model_id: str,
        num_classes: int = 2,
        id2label: dict | None = None,
        label2id: dict | None = None,
    ):
        super().__init__()
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True,
            num_labels=num_classes,
            id2label=id2label or {i: str(i) for i in range(num_classes)},
            label2id=label2id or {str(i): i for i in range(num_classes)},
        )

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None):
        """Returns loss (scalar) when labels provided, else logits."""
        output = self.model(pixel_values=pixel_values, labels=labels)
        return output.loss if labels is not None else output.logits


class ObjectDetectionModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForObjectDetection."""

    def __init__(self, model_id: str, num_classes: int = 2):
        super().__init__()
        self.model = AutoModelForObjectDetection.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True,
            num_labels=num_classes,
        )

    def forward(self, pixel_values: torch.Tensor, labels=None):
        """Returns loss when labels provided (list of dicts), else raw output."""
        output = self.model(pixel_values=pixel_values, labels=labels)
        return output.loss if labels is not None else output


class VideoClassificationModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForVideoClassification."""

    def __init__(
        self,
        model_id: str,
        num_classes: int = 2,
        id2label: dict | None = None,
        label2id: dict | None = None,
    ):
        super().__init__()
        self.model = AutoModelForVideoClassification.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True,
            num_labels=num_classes,
            id2label=id2label or {i: str(i) for i in range(num_classes)},
            label2id=label2id or {str(i): i for i in range(num_classes)},
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


class KeypointDetectionModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForKeypointDetection."""

    def __init__(self, model_id: str):
        super().__init__()
        self.model = AutoModelForKeypointDetection.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True,
        )

    def forward(self, pixel_values: torch.Tensor, labels=None):
        """Returns loss when labels provided, else raw output."""
        output = self.model(pixel_values=pixel_values, labels=labels)
        return output.loss if labels is not None else output


class AudioClassificationModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForAudioClassification."""

    def __init__(
        self,
        model_id: str,
        num_classes: int = 2,
        id2label: dict | None = None,
        label2id: dict | None = None,
    ):
        super().__init__()
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True,
            num_labels=num_classes,
            id2label=id2label or {i: str(i) for i in range(num_classes)},
            label2id=label2id or {str(i): i for i in range(num_classes)},
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.model(input_values=input_values).logits


class SequenceClassificationModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForSequenceClassification."""

    def __init__(
        self,
        model_id: str,
        num_classes: int = 2,
        id2label: dict | None = None,
        label2id: dict | None = None,
    ):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True,
            num_labels=num_classes,
            id2label=id2label or {i: str(i) for i in range(num_classes)},
            label2id=label2id or {str(i): i for i in range(num_classes)},
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class QuestionAnsweringModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForQuestionAnswering."""

    def __init__(self, model_id: str):
        super().__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
    ):
        """Returns loss scalar when start/end positions provided, else raw output."""
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        if start_positions is not None and end_positions is not None:
            return output.loss
        return output


class CausalLMModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForCausalLM."""

    def __init__(self, model_id: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Always returns the scalar language modelling loss."""
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss


class Seq2SeqLMModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForSeq2SeqLM."""

    def __init__(self, model_id: str):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns the scalar seq2seq loss."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        ).loss


class MaskedLMModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForMaskedLM."""

    def __init__(self, model_id: str):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns the scalar masked language modelling loss."""
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "image_classification": ImageClassificationModel,
    "image_segmentation": ImageSegmentationModel,
    "object_detection": ObjectDetectionModel,
    "video_classification": VideoClassificationModel,
    "keypoint_detection": KeypointDetectionModel,
    "audio_classification": AudioClassificationModel,
    "text_classification": SequenceClassificationModel,
    "question_answering": QuestionAnsweringModel,
    "causal_lm": CausalLMModel,
    "seq2seq_lm": Seq2SeqLMModel,
    "masked_lm": MaskedLMModel,
}
