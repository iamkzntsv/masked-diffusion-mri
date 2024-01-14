import argparse
import logging
import os

import nibabel as nib
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

from ..etl.custom_dataset import CustomDataset
from ..etl.data_utils import download_and_save_dataset, np_to_nifti
from ..etl.image_utils import (
    get_reference_image,
    get_reverse_transform,
)
from ..model.model import DiffusionModel
from ..model.repaint import RePaintPipeline
from ..utils import get_device, load_yaml_config, update_config
from diffusers import RePaintScheduler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_inference_steps", type=int, help="Num inference steps")
    parser.add_argument("--jump_length", type=int, help="Jump Length")
    parser.add_argument("--jump_n_sample", type=int, help="Resampling rate")
    return parser.parse_args()


def inpaint(args):
    config = update_config(load_yaml_config("masked_diffusion/model/config.yml"), vars(args))

    config["device"] = device = get_device()
    logger.info(f"The device set to {device}")

    dataset = CustomDataset(
        "data/new/processed",
        hist_ref=None,
        image_transform=T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)]),
        mask_transform=T.ToTensor())

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
    diffusion = DiffusionModel(config).to(device)
    pipe = RePaintPipeline(unet=diffusion.unet, scheduler=scheduler)

    reverse_transform = get_reverse_transform()["image"]

    inpainted_images = []
    for i, (image, mask) in tqdm(enumerate(dataloader), total=len(dataloader)):
        logger.info(f"Slice {i + 1}")
        inpainted_image = image

        mask[mask > 0] = 1
        mask_sum = torch.sum(mask != 1)  # invert mask before summation
        if mask_sum > 0:  # only inpaint if there is any tumour tissue
            inpainted_image = pipe(
                image,
                mask,
                **config["repaint"],
                device=device,
            )
            inpainted_image = reverse_transform(inpainted_image)

            original_image = image.clone()
            inpaint_mask = (mask == 0)
            original_image[inpaint_mask] = inpainted_image[inpaint_mask]
            inpainted_image = original_image.clone()

        else:
            inpainted_image = reverse_transform(inpainted_image)
            logger.info("No tumour mask found. Skipping.")

        inpainted_image = np.split(inpainted_image, args.batch_size, axis=0)
        inpainted_images.extend(inpainted_image)

        # wandb.log({"image_grid": wandb.Image(inpainted_image[0])})

    volume = dataset.slice_ext.combine_slices(inpainted_images)
    affine = dataset.slice_ext.affine

    print(f"test: {np.min(volume)}")

    nifti_image = np_to_nifti(volume, affine)
    save_dir = "data/new/processed/inpainted.nii.gz"
    nib.save(nifti_image, save_dir)
    logger.info(f"Inpainted volume saved at SAVE_PATH")


def main():
    # wandb.init(project="masked-diffusion-mri", config=locals())

    args = parse_args()
    inpaint(args)


if __name__ == "__main__":
    main()