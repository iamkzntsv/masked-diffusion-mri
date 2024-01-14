import cv2
import numpy as np

from ..etl.image_utils import match_image_histogram


class SliceExtractor:
    def __init__(self, hist_ref=None):
        self.hist_ref = hist_ref
        self.target_shape = (200, 200)

    def extract_slices(self, volume, mask=None):
        """
        Extract 2D slices from a 3D volume based on the amounts of brain quantity
        :param volume: nifti image representing MRI volume of a single subject
        :param mask: nifti image representing segmentation mask for corresponding MRI volume
        :return: list of slices
        """
        nx, ny, nz = volume.header.get_data_shape()
        self.original_shape = (nx, ny, nz)
        self.affine = volume.affine

        img_arr = volume.get_fdata()

        mask_arr = mask.get_fdata()
        mask_arr[mask_arr != 0] = 1

        img_slices = []
        mask_slices = []

        for i in range(ny - 1):
            # Get slice
            img = np.squeeze(img_arr[:, i : i + 1, :])
            mask = np.squeeze(mask_arr[:, i : i + 1, :])

            # Rotate
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Equalize
            if self.hist_ref is not None:
                # Normalize
                if np.sum(img) > 0:
                    img = img / np.max(img)

                img = match_image_histogram(img, self.hist_ref)

                img = img.astype(np.uint8)
                mask = mask.astype(np.uint8)

            img_slices.append(img)
            mask_slices.append(mask)

        return img_slices, mask_slices

    def combine_slices(self, img_slices):
        target_shape = self.original_shape
        reconstructed_img = np.zeros(target_shape)

        for i, img in enumerate(img_slices):
            # Reverse the rotation
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            # Assign to volume
            reconstructed_img[:, i, :] = img

        return reconstructed_img
