import os

import requests


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
