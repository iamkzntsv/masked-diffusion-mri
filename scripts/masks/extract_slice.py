import nibabel as nib
import matplotlib.pyplot as plt
import sys


class Extractor:
    def __init__(self, input_path, output_path, slice_number):
        self.input_path = input_path
        self.output_path = output_path
        self.slice_number = slice_number
        self.img = self.load_image()

    def load_image(self):
        try:
            return nib.load(self.input_path)
        except Exception as e:
            print(f"Error loading file {self.input_path}: {e}")
            sys.exit(1)

    def display_slice(self, title):
        data = self.img.get_fdata()
        plt.imshow(data[:, self.slice_number, :], cmap='gray')
        plt.title(f'Slice {self.slice_number} {title}')
        plt.show()

    def extract_and_save_slice(self):
        data = self.img.get_fdata()
        data[:, :self.slice_number - 1, :] = 0
        data[:, self.slice_number + 1:, :] = 0
        new_img = nib.Nifti1Image(data, self.img.affine)
        self.display_slice('After')

        try:
            nib.save(new_img, self.output_path)
            print(f"File saved successfully at {self.output_path}")
        except Exception as e:
            print(f"Error saving file to {self.output_path}: {e}")


def process_structure(structure, slice_number, base_path):
    left_path = f"{base_path}/convert/left_{structure}.nii.gz"
    right_path = f"{base_path}/convert/right_{structure}.nii.gz"
    combined_path = f"{base_path}/convert/{structure}.nii.gz"

    left_output_path = f"{base_path}/structures/left_{structure}_slice_{slice_number}.nii.gz"
    right_output_path = f"{base_path}/structures/right_{structure}_slice_{slice_number}.nii.gz"
    combined_output_path = f"{base_path}/structures/{structure}_slice_{slice_number}.nii.gz"

    # Process left structure
    left_extractor = Extractor(left_path, left_output_path, slice_number)
    left_extractor.display_slice('Before')
    left_extractor.extract_and_save_slice()

    # Process right structure
    right_extractor = Extractor(right_path, right_output_path, slice_number)
    right_extractor.display_slice('Before')
    right_extractor.extract_and_save_slice()

    # Process combined structure
    combined_extractor = Extractor(combined_path, combined_output_path, slice_number)
    combined_extractor.display_slice('Before')
    combined_extractor.extract_and_save_slice()


def main():
    structures = {
        "amygdala": 162,
        "caudate": 143,
        "hippocampus": 165,
        "lateral_ventricle": 130,
        "pallidum": 153,
        "putamen": 150,
        "thalamus": 147
    }

    base_path = "/{path}/masked-diffusion-mri/experiments/hcp/{subject}/masks"

    for structure, slice_number in structures.items():
        process_structure(structure, slice_number, base_path)


if __name__ == "__main__":
    main()
