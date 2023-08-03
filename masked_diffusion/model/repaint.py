import logging
from typing import Any, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.utils
import wandb
from diffusers import DDPMScheduler, RePaintPipeline
from torch import optim
from torch.nn import functional as F

from ..model.guided_diffusion import dist_util
from ..model.guided_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from ..utils import check_pretrained_weights, count_parameters

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RePaintDiffusion(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        # Initialize UNet and DDPM scheduler
        config["model"].update(model_and_diffusion_defaults())
        self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(config["model"], model_and_diffusion_defaults())
        )
        self.scheduler = DDPMScheduler(**self.config["scheduler"])

        # Download pretrained model weights
        self.set_weights()
        logger.info("Model initialized with pretrained weights")

        # Set precision
        if self.config["model"]["use_fp16"]:
            self.model.convert_to_fp16()

        # Add wrapper
        self.model = UNetWrapper(self.model)

        # Create a pipeline
        self.pipe = RePaintPipeline(unet=self.model, scheduler=self.scheduler).to(config["device"])
        logger.info("RePaint pipeline created")

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the clean images from the batch
        clean_images = batch["images"]
        bs = self.config["model"]["batch_size"]

        # Sample noise and add it to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)

        # Get a timestep for each image in a batch
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (bs,),
            device=clean_images.device,
        ).long()

        # Forward diffusion process
        noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)

        # Predict the noise
        noise_pred = self.model(noisy_images, timesteps).to(clean_images.device)

        # Average over channels
        noise_pred = torch.mean(noise_pred, dim=1, keepdim=True)

        return noise_pred, noise

    def _common_step(self, batch, batch_idx) -> torch.Tensor:
        # Calculate the loss
        noise_pred, noise = self.forward(batch)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._common_step(batch, batch_idx)
        self.log("valid_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        # Generate a sample from the model
        sample_np = self.sample()

        # Create a grid and log to wandb
        grid = torchvision.utils.make_grid(sample_np, nrow=4)
        wandb.log({"image_grid": wandb.Image(grid.permute(1, 2, 0).cpu().numpy())})

        logger.info("Sampling complete")

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=self.config["model"]["lr"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def sample(self) -> torch.Tensor:
        model_config = self.config["model"]
        self.model.eval()

        logger.info("Sampling...")
        sample_fn = (
            self.diffusion.p_sample_loop
            if not model_config["use_ddim"]
            else self.diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            self.model,
            (
                model_config["num_samples"],
                1,
                model_config["image_size"],
                model_config["image_size"],
            ),
            clip_denoised=model_config["clip_denoised"],
            progress=True,
            model_kwargs={},
        )

        # Normalize to [0, 1] range
        sample = ((sample + 1) / 2).clamp(0, 1)
        return  torch.mean(sample, dim=1, keepdim=True)

    def set_weights(self) -> None:
        # Check if weights exist in the folder, if not - download
        check_pretrained_weights(**self.config["weights"])
        self.config["model_path"] = self.config["weights"]["save_dir"]
        state_dict = dist_util.load_state_dict(
            self.config["weights"]["save_dir"], map_location="cpu"
        )

        # Set model weights
        self.model.load_state_dict(state_dict)


class UNetWrapper(nn.Module):
    def __init__(self, unet):
        super(UNetWrapper, self).__init__()
        self.unet = unet
        self.dtype = unet.dtype

    def forward(self, x, t):
        # Create a stack of 3 greyscale images
        x = torch.cat([x] * 3, dim=1)

        # Run through the model layers
        out = self.unet(x, t)
        noise_mu, noise_var = torch.split(out, x.shape[1], dim=1)

        return noise_mu


