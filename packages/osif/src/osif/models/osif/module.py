"""Residue Prediction Lightning Module."""

from typing import Any

import pytorch_lightning as pl
import torch
from torch import optim
from torch_geometric.data import Batch, Data

from osif.models.osif.model import OSIFModel
from osif.models.utils.focal_loss import FocalLoss


class OSIFLigtningModule(pl.LightningModule):
    """OSIF Lightning Module."""

    def __init__(
        self,
        lr: float = 0.0001,  # noqa: ARG002
        weight_decay: float = 2e-8,  # noqa: ARG002
        lr_decay: float = 0.5,  # noqa: ARG002
        lr_patience: float = 3,  # noqa: ARG002
        gamma: float = 2,
    ):
        """OSIF Lightning Module.

        Args:
            lr (float, optional): learning rate. Defaults to 0.001.
            weight_decay (float, optional): weight decay for the optimizer. Defaults to 2e-3.
            lr_decay (float, optional): learning rate decay factor. Defaults to 0.5.
            lr_patience (float, optional): learning rate patience (in epochs). Defaults to 3.
            gamma (float, optional): gamma value for the focal loss. Defaults to 2.

        """
        super().__init__()
        self.save_hyperparameters()
        self.model = OSIFModel()

        self.loss_fn = FocalLoss(gamma=gamma)
        self.validation_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, batch: Data | Batch) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(batch)

    def training_step(self, batch: Batch, batch_nb: int) -> torch.Tensor:  # noqa: ARG002
        """A single training step."""
        logits = self.forward(batch)

        loss = self.loss_fn(logits, batch.y[batch.query_mask])
        acc = (logits.max(dim=-1).indices == batch.y[batch.query_mask]).float().mean()
        batch_size = batch.query_mask.int().sum()
        self.log("train_acc", acc, on_step=True, prog_bar=True, batch_size=batch_size)
        self.log("train_loss", loss, on_step=True, prog_bar=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch: Batch, batch_nb: int) -> None:  # noqa: ARG002
        """A single validation step."""
        logits = self.forward(batch)

        loss = self.loss_fn(logits, batch.y[batch.query_mask])
        batch_size = batch.query_mask.int().sum()
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        pred_classes = logits.max(dim=-1).indices.cpu()
        true_classes = batch.y[batch.query_mask].cpu()

        self.validation_outputs.append((pred_classes, true_classes))

    def on_validation_epoch_end(self) -> None:
        """Write validation metrics to the log on the epoch end."""
        pred_classes = torch.cat([a[0] for a in self.validation_outputs])
        true_classes = torch.cat([a[1] for a in self.validation_outputs])

        class_acc = (pred_classes == true_classes).float().mean()
        self.log("val_acc", class_acc, on_epoch=True)
        self.validation_outputs = []

    def configure_optimizers(
        self,
    ) -> Any:
        """Configure the optimizer and scheduler."""
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.hparams["lr_decay"],
                    patience=self.hparams["lr_patience"],
                    min_lr=1e-7,
                ),
                "monitor": "val_loss_epoch",
            },
        }

    def save_model(self, path: str) -> None:
        """Save the model to the disk."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Load the model from the disk."""
        self.model.load_state_dict(torch.load(path))
