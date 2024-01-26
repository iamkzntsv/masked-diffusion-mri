import os
import random
import requests

import numpy as np
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
