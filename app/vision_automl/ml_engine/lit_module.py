from typing import Callable, Optional

import lightning as L
import torch


class LitClassification(L.LightningModule):
    def __init__(self, model_fn: Callable[[], torch.nn.Module], lr: float = 0.001, loss_fn: Optional[torch.nn.Module] = None):
        """
        model_fn: callable returning a torch.nn.Module (e.g. lambda: resnet34(num_classes=nc))
        lr: learning rate
        loss_fn: optional custom loss function
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model_fn", "loss_fn"])

        self.model = model_fn()
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.CrossEntropyLoss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


