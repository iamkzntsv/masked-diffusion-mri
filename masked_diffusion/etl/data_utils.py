import os

import cv2
import numpy as np
from datasets import load_dataset


def download_and_save_dataset(path, save_dir) -> None:
    # Load the dataset from HuggingFace hub
    dataset = load_dataset(path)

    # Save the dataset locally
    dataset.save_to_disk(save_dir)


def get_reference_image(dataset):
    """Reference for histogram matching"""
    return np.array(dataset[42]["image"])
