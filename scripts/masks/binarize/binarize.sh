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

# Corresponding left and right match numbers for each structure
left_matches=(4 10 11 12 13 17 18)
right_matches=(43 49 50 51 52 53 54)

input_file="aseg.auto_noCCseg.mgz"

# Loop through each structure
for i in "${!structures[@]}"; do
    structure=${structures[$i]}
    left_match=${left_matches[$i]}
    right_match=${right_matches[$i]}

    # Process left
    left_output="left_${structure}.mgz"
    mri_binarize --i "$input_file" --match "$left_match" --o "$left_output"
    echo "Processed $left_output"

    # Process right
    right_output="right_${structure}.mgz"
    mri_binarize --i "$input_file" --match "$right_match" --o "$right_output"
    echo "Processed $right_output"

    # Process combined
    combined_output="${structure}.mgz"
    mri_binarize --i "$input_file" --match "$left_match" --match "$right_match" --o "$combined_output"
    echo "Processed combined $combined_output"
done

