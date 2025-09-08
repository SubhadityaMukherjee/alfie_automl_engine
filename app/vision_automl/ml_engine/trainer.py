import time
from typing import Any, Dict, List, Optional

import lightning as L
import torch
from torch import nn, optim
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, monitor: str = "val_loss", patience: int = 3, min_delta: float = 0.0):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def on_epoch_end(self, trainer: "FabricTrainer", epoch: int, logs: Dict[str, float]):
        current = logs.get(self.monitor)
        if current is None:
            return
        if current < self.best - self.min_delta:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            trainer.fabric.print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                trainer.fabric.print("Early stopping triggered!")
                trainer.epochs = epoch + 1


class FabricTrainer:
    def __init__(
        self,
        datamodule,
        model_class,
        model_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class=optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        lr: float = 0.001,
        epochs: int = 1,
        time_limit: Optional[float] = None,
        device: str = "auto",
        callbacks: Optional[List[Any]] = None,
        input_dtype: torch.dtype = torch.float32,
        target_dtype: torch.dtype = torch.long,
    ):
        self.datamodule = datamodule
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {"lr": lr}
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.time_limit = time_limit
        self.device = device
        self.callbacks = callbacks or []
        self.input_dtype = input_dtype
        self.target_dtype = target_dtype

        self.fabric = L.Fabric(devices=self.device)
        self._setup_model_optimizer()

    def _setup_model_optimizer(self):
        self.model = self.model_class(**self.model_kwargs)
        self.optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        self.model, self.optimizer, self.train_loader, self.val_loader = self.fabric.setup(
            self.model, self.optimizer, train_loader, val_loader
        )
        self.test_loader = self.datamodule.test_dataloader()

    def _move_batch(self, batch):
        if isinstance(batch, dict):
            pixel_values = batch["pixel_values"].to(self.fabric.device, dtype=self.input_dtype)
            labels = batch["labels"].to(self.fabric.device, dtype=self.target_dtype)
            return {"pixel_values": pixel_values, "labels": labels}
        imgs, labels = batch
        imgs = imgs.to(self.fabric.device, dtype=self.input_dtype)
        labels = labels.to(self.fabric.device, dtype=self.target_dtype)
        return {"pixel_values": imgs, "labels": labels}

    def _check_time_limit(self, start_time: float) -> bool:
        if self.time_limit and (time.time() - start_time) > self.time_limit:
            self.fabric.print("Time limit reached. Stopping training.")
            return True
        return False

    def train_epoch(self, epoch: int, start_time: float) -> float:
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            if self._check_time_limit(start_time):
                return running_loss / max(1, len(self.train_loader))
            moved = self._move_batch(batch)
            self.optimizer.zero_grad()
            outputs = self.model(moved["pixel_values"])
            loss = self.loss_fn(outputs, moved["labels"])
            self.fabric.backward(loss)
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

    def validate(self, start_time: float):
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                if self._check_time_limit(start_time):
                    break
                moved = self._move_batch(batch)
                outputs = self.model(moved["pixel_values"])
                loss = self.loss_fn(outputs, moved["labels"])
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == moved["labels"]).sum().item()
                total += moved["labels"].size(0)
        avg_loss = val_loss / max(1, len(self.val_loader))
        acc = correct / max(1, total)
        # self.fabric.print(f"Val Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}")
        return avg_loss, acc

    def test(self):
        self.model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                moved = self._move_batch(batch)
                outputs = self.model(moved["pixel_values"])
                loss = self.loss_fn(outputs, moved["labels"])
                test_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == moved["labels"]).sum().item()
                total += moved["labels"].size(0)
        avg_loss = test_loss / len(self.test_loader)
        acc = correct / total
        # self.fabric.print(f"\nTest Loss: {avg_loss:.4f}, Test Acc: {acc:.4f}")
        return avg_loss, acc

    def fit(self):
        start_time = time.time()
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch, start_time)
            val_loss, val_acc = self.validate(start_time)
            logs = {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)
            if self._check_time_limit(start_time):
                break
        return self.test()


