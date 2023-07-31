import os
import random

import nibabel as nib
import numpy as np
import requests
import torch
import yaml

FILENAMES = ["t1", "t1-voided", "mask"]


def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU


def load_all_mri(dir_path):
    return {filename: load_mri(os.path.join(dir_path, filename + ".mgz")) for filename in FILENAMES}


def load_mri(path):
    image = nib.load(path)
    return image


def get_device():
    return (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_yaml_config(file_path):
    """
    Load yaml configuration file

    :param file_path: Path to the yaml configuration file
    :return: Contents of the configuration file
    """
    with open(file_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    return config


def check_pretrained_weights(url, save_dir):
    # If already exists - do not download again
    if os.path.exists(save_dir):
        print(f"Pretrained weights found at {save_dir}. Skipping download.")
        return
    else:
        print("Downloading pretrained weights...")
        download_weights(url, save_dir)


def download_weights(url, save_dir):
    response = requests.get(url)
    response.raise_for_status()  # raise an exception if the request failed
    with open(save_dir, "wb") as f:
        f.write(response.content)


def update_config(config, args_dict):
    for k, v in config.items():
        for kk, vv in args_dict.items():
            if kk in v:
                if vv is not None:
                    config[k][kk] = vv
    return config


def count_parameters(model):
    params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return params


def dir_path(string):
    """
    https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
    """
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
