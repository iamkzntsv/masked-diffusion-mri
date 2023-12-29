import logging
import os
import argparse
import nibabel as nib
import numpy as np
from datasets import load_from_disk
from torch.utils.data import DataLoader
from ..etl.custom_dataset import CustomDataset
from ..etl.data_utils import download_and_save_dataset, np_to_nifti
from ..etl.image_utils import (
    get_reference_image,
    get_transform,
)
from ..utils import dir_path, load_yaml_config, update_config
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=dir_path, help="Path to a nifti image")
    parser.add_argument("--save_dir", type=dir_path, help="Save directory")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--offset", type=int, help="Whether to use an offset during center crop")
    return parser.parse_args()


def main():
    args = parse_args()
    config = update_config(load_yaml_config("masked_diffusion/model/config.yml"), vars(args))

    download_and_save_dataset(**config["data"]["ixi"])
    ref_dataset = load_from_disk(os.path.join("data/ixi/transformed", "train"))
    hist_ref = get_reference_image(ref_dataset)

    transform = get_transform(config["model"]["image_size"])

    dataset = CustomDataset(args.path, hist_ref, image_transform=transform["image"], mask_transform=transform["mask"])
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    logger.info(f"Preprocessing nifti volume...")
    processed_im_slices = []
    processed_mask_slices = []
    for i, (image, mask) in tqdm(enumerate(dataloader), total=len(dataloader)):

        processed_im = image.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        processed_im = np.split(processed_im, args.batch_size, axis=0)
        processed_im_slices.extend(processed_im)

        processed_mask = mask.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        processed_mask = np.split(processed_mask, args.batch_size, axis=0)
        processed_mask_slices.extend(processed_mask)

    affine = dataset.slice_ext.affine

    volume = dataset.slice_ext.combine_slices(processed_im_slices)
    nifti_volume = np_to_nifti(volume, affine)
    nib.save(nifti_volume, os.path.join(args.save_dir, "t1.mgz"))

    mask = dataset.slice_ext.combine_slices(processed_mask_slices)
    nifti_mask = np_to_nifti(mask, affine)
    nib.save(nifti_mask, os.path.join(args.save_dir, "mask.mgz"))

    logger.info(f"Processed volumes saved at: {args.save_dir}")


if __name__ == "__main__":
    main()
