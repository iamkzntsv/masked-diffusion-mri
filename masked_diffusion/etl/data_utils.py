import os

import nibabel as nib
import numpy as np
from datasets import load_dataset

FILENAMES = ["t1", "mask"]


def download_and_save_dataset(path, save_dir) -> None:
    # Load the dataset from HuggingFace hub
    dataset = load_dataset(path)

    # Save the dataset locally
    dataset.save_to_disk(save_dir)


def np_to_nifti(arr, affine):
    if affine is None:
        affine = np.eye(4)

    if arr.ndim not in (3, 4):
        raise ValueError("Input array must be 3D or 4D.")

    nifti_img = nib.Nifti1Image(arr, affine)

    return nifti_img


def load_all_mri(dir_path):
    return {filename: load_mri(os.path.join(dir_path, filename + ".mgz")) for filename in FILENAMES}


def load_mri(path):
    image = nib.load(path)
    return image
