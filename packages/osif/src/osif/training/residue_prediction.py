"""Running training of the residue prediction model."""

import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, RadiusGraph, RandomRotate

from osif.data.datasets.protein.atoms import ProteinAtomsDataset
from osif.data.transforms.masking.residue import MaskCentralResidue
from osif.data.transforms.masking.span import MaskResidueSpan
from osif.data.transforms.random_apply import RandomApplyTransform
from osif.data.transforms.virtual_nodes import AddIntermediateVirtualNodes
from osif.models.osif.module import OSIFLigtningModule


def train_residue_predictor(
    experiment_name: str,
    train_dataset_root: str,
    val_dataset_root: str,
    log_dir: str = "./data/logs",
    weight_dir: str = "./data/weights",
    neighborhood_size: tuple[float, float] = (7.1, 14.1),
    edge_length: float = 4.1,
    batch_size: int = 2,
    num_epochs: int = 100,
    num_workers: int = 2,
    fast_dev_run: bool = False,
    seed: int = 1337,
    checkpoint_path: str | None = None,
    lr: float = 1e-4,
) -> OSIFLigtningModule:
    """Run training of the pocket score model."""
    pl.seed_everything(seed)
    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")
    model = (
        OSIFLigtningModule(lr=lr)
        if checkpoint_path is None
        else OSIFLigtningModule.load_from_checkpoint(checkpoint_path)
    )

    train_ds = ProteinAtomsDataset(
        train_dataset_root,
        transform=Compose(
            [
                RandomApplyTransform(
                    MaskResidueSpan(atoms_to_leave=("N", "CA", "C", "O", "CB", "VCB")),
                    p=0.5,
                ),
                MaskCentralResidue(atoms_to_leave=("N", "CA", "C", "O", "CB", "VCB")),
                RandomRotate(180, axis=0),
                RandomRotate(180, axis=1),
                RandomRotate(180, axis=2),
                AddIntermediateVirtualNodes(edge_length=edge_length, merge_cutoff=edge_length - 0.2),
                RadiusGraph(r=edge_length, max_num_neighbors=64),
            ]
        ),
        central_atom_names=("CB", "VCB"),
        min_neighborhood_size=neighborhood_size[0],
        max_neighborhood_size=neighborhood_size[1],
        preprocess_workers=num_workers,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_dl = DataLoader(
        ProteinAtomsDataset(
            val_dataset_root,
            transform=Compose(
                [
                    MaskResidueSpan(atoms_to_leave=("N", "CA", "C", "O", "CB", "VCB")),
                    AddIntermediateVirtualNodes(edge_length=edge_length, merge_cutoff=edge_length - 0.2),
                    RadiusGraph(r=edge_length, max_num_neighbors=64),
                ]
            ),
            central_atom_names=("CB", "VCB"),
            min_neighborhood_size=neighborhood_size[0],
            max_neighborhood_size=neighborhood_size[1],
            preprocess_workers=num_workers,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )

    trainer = pl.Trainer(
        devices=1,
        logger=[
            loggers.WandbLogger(project=experiment_name),
            # loggers.TensorBoardLogger(save_dir=log_dir, name=experiment_name),
            loggers.CSVLogger(save_dir=log_dir, name=experiment_name),
        ],
        callbacks=[
            # LearningRateFinder(max_lr=1e-2),
            ModelCheckpoint(
                dirpath=os.path.join(weight_dir, experiment_name),
                save_weights_only=False,
                save_top_k=-1,
                every_n_train_steps=25000,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        max_epochs=num_epochs,
        log_every_n_steps=100,
        precision="bf16-mixed",
        accumulate_grad_batches=3,
        gradient_clip_val=1.0,
        limit_train_batches=25000,
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
    )

    return model
