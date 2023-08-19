import argparse
import logging
import os

import nibabel as nib
from datasets import load_from_disk
from diffusers import RePaintScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from ..etl.custom_dataset import CustomDataset
from ..etl.data_utils import download_and_save_dataset, np_to_nifti
from ..etl.image_utils import (
    get_reference_image,
    get_reverse_transform,
    get_transform,
)
from ..model.model import DiffusionModel
from ..model.repaint import RePaintPipeline
from ..utils import dir_path, get_device, load_yaml_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=dir_path, help="Path to a nifti image")
    return parser.parse_args()


def main(config):
    # Get IXI dataset
    download_and_save_dataset(**config["data"]["ixi"])
    ref_dataset = load_from_disk(os.path.join("data/ixi/transformed", "train"))
    hist_ref = get_reference_image(ref_dataset)

    # wandb.init(project="masked-diffusion-mri", config=locals())

    config["device"] = device = get_device()

    transform, transform_state = get_transform(config["model"]["image_size"])
    dataset = CustomDataset(args.path, hist_ref, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
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
        inpainted_image = pipe(
            image, mask, num_inference_steps=250, jump_length=10, jump_n_sample=10, device=device
        )
        inpainted_image = reverse_transform(inpainted_image)
        inpainted_images.append(inpainted_image)

    volume = dataset.slice_ext.combine_slices(inpainted_images)
    affine = dataset.slice_ext.affine

    nifti_image = np_to_nifti(volume, affine)
    save_path = os.path.join(config["save_path"], "inpainted.nii.gz")
    nib.save(nifti_image, save_path)

    """
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    file_path = "original_image.jpg"
    image.save(file_path)
    print(f"Image saved at {file_path}")

    # Scale the image values to [0, 255]
    image = (inpainted_im * 255).astype(np.uint8)
    image = Image.fromarray(image)
    file_path = "inpainted_image.jpg"
    image.save(file_path)
    print(f"Image saved at {file_path}")
    """


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config("masked_diffusion/model/config.yml")
    main(config)
