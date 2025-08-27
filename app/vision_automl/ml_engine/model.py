from torch import nn
from timm import create_model as create_model_timm


class ClassificationModel(nn.Module):
    def __init__(self, model_name: str = "resnet34", num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.model = create_model_timm(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


