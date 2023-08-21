import argparse
import logging
import os

import numpy
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb

from ..etl.ixi_data_module import IXIDataModule
from ..model.model import DiffusionModel
from ..utils import get_device, load_yaml_config, set_seeds, update_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="Batch size (default: 1)")
    parser.add_argument("--max_epochs", type=int, help="Number of epochs (default: 10)")
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of subprocesses to use for data loading (default: 0)",
    )
    parser.add_argument("--lr", type=float, help="Learning rate (default: 1e-3)")
    parser.add_argument(
        "--pretrained",
        type=bool,
        help="Whether to used a pretrained model (default: True)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = update_config(load_yaml_config("masked_diffusion/model/config.yml"), vars(args))

    set_seeds()
    wandb.init(project="masked-diffusion-mri", config=locals())

    config["device"] = device = get_device()
    logger.info(f"The device set to {device}")

    # Get the data
    data = IXIDataModule(**config["data"]["ixi"], **config["model"])

    # Initialize RePaint model, Trainer and WandbLogger
    repaint = DiffusionModel(config)

    wandb_logger = WandbLogger(**config["logging"])
    trainer = Trainer(accelerator=device, logger=wandb_logger, **config["trainer"])

    # Training loop
    logger.info("Training...")
    trainer.fit(repaint, data)
    logger.info("✅ Training complete!")

    torch.save(repaint.unet.state_dict(), os.path.join(wandb.run.dir, "model.pt"))


if __name__ == "__main__":
    main()
