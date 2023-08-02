import logging
import os
from typing import Any, Dict

import pytorch_lightning as pl
from datasets import Dataset, load_from_disk
from lightning.pytorch.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np

from .data_utils import download_and_save_dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class IXIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: str = "./",
        save_dir: str = "./",
        batch_size: int = 4,
        image_size: int = 256,
        num_workers: int = 1,
        **kwargs
    ) -> None:
        super().__init__()
        self.path = path
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )
        self.dims = None

    def prepare_data(self) -> None:
        download_and_save_dataset(self.path, self.save_dir)
        logger.info("Training data has been successfully downloaded")

    def setup(self, stage: str = None) -> None:
        # Load training and validation data
        if stage == "fit" or stage is None:
            self.train_ds = load_from_disk(os.path.join(self.save_dir, "train"))
            self.val_ds = load_from_disk(os.path.join(self.save_dir, "validation"))

            # Apply preprocessing
            self.train_ds.set_transform(self.preprocess)
            self.val_ds.set_transform(self.preprocess)

    def preprocess(self, examples: Dataset) -> Dict[str, list[Any]]:
        examples = [self.transform(image) for image in examples["image"]]
        return {"images": examples}

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
        )
