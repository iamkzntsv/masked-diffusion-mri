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
    transform_state = TransformState()
    return {
        "image": image_transform(image_size, transform_state),
        "mask": mask_transform(image_size),
    }, transform_state


def get_reverse_transform(original_size, transform_state):
    return {"image": reverse_image_transform(original_size, transform_state), "mask": None}


def image_transform(image_size, transform_state):
    return T.Compose(
        [
            T.ToTensor(),
            CenterCropWithOffset(transform_state),
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode("bilinear")),
            T.Normalize(0.5, 0.5),
        ]
    )


def mask_transform(image_size):
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode("nearest")),
            T.ToTensor(),
            InvertMask(),
        ]
    )


def reverse_image_transform(original_size, transform_state):
    return T.Compose(
        [
            Denormalize(0.5, 0.5),
            ReverseCenterCropWithOffset(transform_state),
            T.Resize((original_size, original_size), interpolation=T.InterpolationMode("bilinear")),
            ToNumpy(),
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
    def __init__(self, state, target_shape=(200, 200), offset=15):
        self.state = state
        self.target_shape = target_shape
        self.offset = offset

    def __call__(self, image):
        h, w = image.shape[-2], image.shape[-1]
        new_h, new_w = self.target_shape

        top = int((h - new_h) / 2) + self.offset
        left = int((w - new_w) / 2)
        bottom = top + new_h
        right = left + new_w

        self.state.set_original_size((h, w))

        cropped_image = image[..., top:bottom, left:right]
        return cropped_image


class ReverseCenterCropWithOffset:
    def __init__(self, state, target_shape=(200, 200), offset=15):
        self.state = state
        self.target_shape = target_shape
        self.offset = offset

    def __call__(self, cropped_image):
        original_h, original_w = self.state.get_original_size()

        pad_top = int((original_h - self.target_shape[0]) / 2) + self.offset
        pad_bottom = original_h - (pad_top + self.target_shape[0])
        pad_left = int((original_w - self.target_shape[1]) / 2)
        pad_right = original_w - (pad_left + self.target_shape[1])

        padded_image = F.pad(cropped_image, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        return padded_image


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
        return image
