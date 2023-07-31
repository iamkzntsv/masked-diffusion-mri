import argparse
import os

import torch
from datasets import load_from_disk

from ..etl.data_utils import download_and_save_dataset, get_reference_image
from ..etl.slice_extractor import SliceExtractor
from ..model.repaint import RePaintDiffusion
from ..utils import dir_path, load_all_mri, load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=dir_path, help="Path to a nifti image")
    return parser.parse_args()


def main(config):
    mri_images = load_all_mri(args.path)

    # Get the dataset
    download_and_save_dataset(**config["data"]["ixi"])
    dataset = load_from_disk(os.path.join("data/ixi/transformed", "train"))
    hist_ref = get_reference_image(dataset)

    repaint = RePaintDiffusion(config)
    repaint.pipe.unet.load_state_dict(torch.load("/Users/iamkzntsv/model.pt"))

    slice_ext = SliceExtractor(hist_ref)
    images, masks = slice_ext.get_slices(mri_images["t1-voided"], mri_images["mask"])


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config("masked_diffusion/model/config.yml")
    main(config)
