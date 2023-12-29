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
from ..etl.data_utils import determine_file_type, download_and_save_dataset, np_to_nifti
from ..etl.image_utils import (
    get_reference_image,
    get_reverse_transform,
)
from ..model.model import DiffusionModel
from ..model.repaint import RePaintPipeline, RePaintScheduler
from ..utils import dir_path, get_device, load_yaml_config, update_config
from diffusers import RePaintScheduler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=dir_path, help="Path to a nifti image")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_inference_steps", type=int, help="Num inference steps")
    parser.add_argument("--jump_length", type=int, help="Jump Length")
    parser.add_argument("--jump_n_sample", type=int, help="Resampling rate")
    return parser.parse_args()


def main():
    args = parse_args()

    input_file_type = determine_file_type(args.path)

    # wandb.init(project="masked-diffusion-mri", config=locals())

    if input_file_type == "nifti":
        inpaint(args)
    else:
        logger.info("Unknown file type.")


def inpaint(args):
    config = update_config(load_yaml_config("masked_diffusion/model/config.yml"), vars(args))

    wandb.init(project="masked-diffusion-mri", config=locals())

    # Get IXI dataset
    download_and_save_dataset(**config["data"]["ixi"])
    ref_dataset = load_from_disk(os.path.join("data/ixi/transformed", "train"))
    hist_ref = get_reference_image(ref_dataset)

    config["device"] = device = get_device()

    dataset = CustomDataset(
        args.path,
        hist_ref,
        image_transform=T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)]),
        mask_transform=T.ToTensor())

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    scheduler = RePaintScheduler()
    diffusion = DiffusionModel(config).to(device)
    scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
    pipe = RePaintPipeline(unet=diffusion.unet, scheduler=scheduler)

    reverse_transform = get_reverse_transform()["image"]

    inpainted_images = []
    for i, (image, mask) in tqdm(enumerate(dataloader), total=len(dataloader)):

        mask[mask > 0] = 1
        mask_sum = torch.sum(mask != 1)  # invert mask before summation

        if mask_sum > 0:  # only inpaint if there is any tumour tissue
            inpainted_image = pipe(
                image,
                mask,
                **config["repaint"],
                device=device,
            )
        else:
            logger.info("No tumour mask found. Skipping.")

            mask_sum = torch.sum(mask != 1)  # invert before summation
            if mask_sum > 0:  # only inpaint if there is any tumour tissue
                inpainted_image = pipe(
                    image,
                    mask,
                    **config["repaint"],
                    device=device,
                )
            else:
                logger.info("No tumour mask found. Skipping.")

            inpainted_image = reverse_transform(inpainted_image)
            inpainted_image = np.split(inpainted_image, args.batch_size, axis=0)
            inpainted_images.extend(inpainted_image)

            from PIL import Image

            wandb.log({"image_grid": wandb.Image(inpainted_image[0])})
            return

        # wandb.log({"image_grid": wandb.Image(inpainted_image[0])})

    volume = dataset.slice_ext.combine_slices(inpainted_images)
    affine = dataset.slice_ext.affine

    nifti_image = np_to_nifti(volume, affine)
    save_path = "inpainted.nii.gz"
    nib.save(nifti_image, save_path)
    logger.info(f"Inpainted volume saved at: {save_path}")


if __name__ == "__main__":
    main()
