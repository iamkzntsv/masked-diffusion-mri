#!/bin/bash

base_dir="."

# Find all .mgz files and convert them
find "$base_dir" -type f -name "*.mgz" | while read -r mgz_file; do
    nii_file="${mgz_file%.mgz}.nii.gz"

    mri_convert "$mgz_file" "$nii_file"

    echo "Converted $mgz_file to $nii_file"
done

echo "Conversion of all .mgz files to .nii.gz completed."
