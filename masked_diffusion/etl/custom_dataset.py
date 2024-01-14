import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ..etl.data_utils import load_all_mri
from ..etl.slice_extractor import SliceExtractor


class CustomDataset(Dataset):
    def __init__(self, path, hist_ref=None, image_transform=None, mask_transform=None):
        self.slice_ext = SliceExtractor(hist_ref)
        self.images, self.masks = self.preprocess(path)
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        mask = Image.fromarray(self.masks[idx])

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask

    def preprocess(self, path):
        nii_images = load_all_mri(path)
        images, masks = self.slice_ext.extract_slices(nii_images["t1"], nii_images["mask"])
        return images, masks
