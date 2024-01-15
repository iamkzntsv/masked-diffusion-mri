import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from skimage import exposure


def match_image_histogram(source_img, reference_img):
    # Define non-black mask for reference image
    reference_mask = reference_img > 0

    # Define non-black mask for source image
    source_mask = source_img > 0

    # Perform histogram matching
    matched_image = exposure.match_histograms(
        source_img[source_mask], reference_img[reference_mask]
    )

    # Create output image with non-black pixels replaced by matched pixels
    img_eq = np.zeros_like(source_img)
    img_eq[source_mask] = matched_image

    return img_eq


def get_transform(image_size):
    return {
        "image": image_transform(image_size),
        "mask": mask_transform(image_size),
    }


def get_reverse_transform():
    return {"image": reverse_image_transform(), "mask": None}


def image_transform(image_size):
    return T.Compose(
        [
            T.ToTensor(),
            CenterCropWithOffset(),
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode("bicubic")),
        ]
    )


def mask_transform(image_size):
    return T.Compose(
        [
            T.ToTensor(),
            CenterCropWithOffset(),
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode("nearest")),
            InvertMask(),
        ]
    )


def reverse_image_transform():
    return T.Compose(
        [
            Denormalize(0.5, 0.5)
        ]
    )


def get_reference_image(dataset):
    """Reference for histogram matching"""
    return np.array(dataset[42]["image"])


class TransformState:
    def __init__(self):
        self.original_size = None

    def set_original_size(self, size):
        self.original_size = size

    def get_original_size(self):
        return self.original_size


class CenterCropWithOffset:
    def __init__(self, target_shape=(200, 200), offset=15):
        self.target_shape = target_shape
        self.offset = offset

    def __call__(self, image):
        h, w = image.shape[-2], image.shape[-1]
        new_h, new_w = self.target_shape

        top = int((h - new_h) / 2) + self.offset
        left = int((w - new_w) / 2)
        bottom = top + new_h
        right = left + new_w

        cropped_image = image[..., top:bottom, left:right]
        return cropped_image


class InvertMask:
    def __call__(self, mask):
        mask = (mask > 0).float()
        inverted_mask = 1 - mask  # Invert the binary mask
        return inverted_mask


class Denormalize:
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = image * self.std + self.mean
        return image


class ToNumpy:
    def __call__(self, image_tensor):
        image_tensor = torch.mean(image_tensor, dim=1, keepdim=True)
        image = image_tensor.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        image = np.clip(image, 0, 1)
        return image
