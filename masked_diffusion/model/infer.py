import argparse
import logging
import os

import numpy as np
import nibabel as nib
import torch
from datasets import load_from_disk
from diffusers import RePaintScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..etl.custom_dataset import CustomDataset
from ..etl.data_utils import download_and_save_dataset, np_to_nifti
from ..etl.image_utils import (
    get_reference_image,
    get_reverse_transform,
    get_transform,
)
from ..model.model import DiffusionModel
from ..model.repaint import RePaintPipeline
from ..utils import dir_path, get_device, load_yaml_config, update_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    config = update_config(load_yaml_config("masked_diffusion/model/config.yml"), vars(args))

    # Get IXI dataset
    download_and_save_dataset(**config["data"]["ixi"])
    ref_dataset = load_from_disk(os.path.join("data/ixi/transformed", "train"))
    hist_ref = get_reference_image(ref_dataset)

    config["device"] = device = get_device()

    transform, transform_state = get_transform(config["model"]["image_size"])
    dataset = CustomDataset(args.path, hist_ref, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
    diffusion = DiffusionModel(config).to(device)
    pipe = RePaintPipeline(unet=diffusion.unet, scheduler=scheduler)

    original_shape = dataset.slice_ext.original_shape[0]
    reverse_transform = get_reverse_transform(original_shape, transform_state)["image"]

    inpainted_images = []
    for i, (image, mask) in tqdm(enumerate(dataloader), total=len(dataloader)):
        logger.info(f"Slice {i + 1}")
        inpainted_image = image

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

    volume = dataset.slice_ext.combine_slices(inpainted_images)
    affine = dataset.slice_ext.affine

    nifti_image = np_to_nifti(volume, affine)
    save_path = os.path.join(config["save_path"], "inpainted.nii.gz")
    nib.save(nifti_image, save_path)
    logger.info(f"Inpainted volume saved at: {save_path}")


if __name__ == "__main__":
    main()
