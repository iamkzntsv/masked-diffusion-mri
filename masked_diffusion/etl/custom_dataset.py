import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ..etl.data_utils import load_all_mri
from ..etl.slice_extractor import SliceExtractor


class CustomDataset(Dataset):
    def __init__(self, path, hist_ref, transform=None):
        self.slice_ext = SliceExtractor(hist_ref)
        self.images, self.masks = self.preprocess(path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx].astype(np.uint8))
        mask = Image.fromarray(self.masks[idx].astype(np.uint8))

        if self.transform:
            image = self.transform["image"](image)
            mask = self.transform["mask"](mask)
        return image, mask

    def preprocess(self, path):
        nii_images = load_all_mri(path)
        images, masks = self.slice_ext.extract_slices(nii_images["t1"], nii_images["mask"])
        return images, masks
