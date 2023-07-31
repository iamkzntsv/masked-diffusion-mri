import cv2
import numpy as np

from ..etl.image_utils import center_crop, match_image_histogram


class SliceExtractor:
    def __init__(self, hist_ref=None):
        self.hist_ref = hist_ref
        self.target_shape = (200, 200)

    def get_slices(self, volume, mask=None):
        """
        Extract 2D slices from a 3D volume based on the amounts of brain quantity
        :param volume: nifti image representing MRI volume of a single subject
        :param mask: nifti image representing segmentation mask for corresponding MRI volume
        :return: list of slices, where each slice is a numpy array of shape (256, 150)
        """
        nx, ny, nz = volume.header.get_data_shape()

        img_arr = volume.get_fdata()

        # Loop over axial plane
        if mask is not None:
            mask_arr = mask.get_fdata()
            mask_arr[mask_arr != 0] = 1

            img_slices = []
            mask_slices = []

            for i in range(ny - 1):

                # Get slice, rotate
                img = np.squeeze(img_arr[:, i : i + 1, :])
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                mask = np.squeeze(mask_arr[:, i : i + 1, :])
                mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Normalization, equalization and cropping
                if np.sum(img) > 0:
                    img = img / np.max(img)
                img = center_crop(img, self.target_shape)

                if self.hist_ref is not None:
                    img = match_image_histogram(img, self.hist_ref)
                img_slices.append(img)

                mask = center_crop(mask, self.target_shape)
                mask_slices.append(mask)

            return img_slices, mask_slices
