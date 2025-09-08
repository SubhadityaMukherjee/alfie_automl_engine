from torch import nn
from transformers import AutoModelForImageClassification


class ClassificationModel(nn.Module):
    """Thin wrapper over HF image classification model for logits output."""
    def __init__(
        self,
        model_id: str = "google/vit-base-patch16-224",
        num_classes: int = 2,
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

    def forward(self, pixel_values):
        """Forward pass returning raw classification logits."""
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits
