#!/bin/bash

structures=(
    "lateral_ventricle"
    "thalamus"
    "caudate"
    "putamen"
    "pallidum"
    "hippocampus"
    "amygdala"
)

for structure in "${structures[@]}"; do
    # Process left
    mri_convert "left_${structure}.mgz" "left_${structure}.nii.gz"
    echo "Converted left_${structure}.mgz to left_${structure}.nii.gz"

    # Process right
    mri_convert "right_${structure}.mgz" "right_${structure}.nii.gz"
    echo "Converted right_${structure}.mgz to right_${structure}.nii.gz"

    # Process combined
    mri_convert "${structure}.mgz" "${structure}.nii.gz"
    echo "Converted ${structure}.mgz to ${structure}.nii.gz"
done
