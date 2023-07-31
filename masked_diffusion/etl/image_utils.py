import numpy as np
from skimage import exposure


def center_crop(img, size, offset=15):
    h, w = img.shape
    new_h, new_w = size

    top = int((h - new_h) / 2) + offset
    left = int((w - new_w) / 2)
    bottom = top + new_h
    right = left + new_w

    cropped_image = img[top:bottom, left:right]
    return cropped_image


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
