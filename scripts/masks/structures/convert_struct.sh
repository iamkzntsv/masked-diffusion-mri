#!/bin/bash

base_dir="."

# Find all .nii.gz files
find "$base_dir" -type f -name "*.nii.gz" | while read -r file; do
    # Output file path
    output_dir=$(dirname "$file")
    output_file="${output_dir}/mask.mgz"

    mri_convert "$file" "$output_file"

    echo "Converted $file to $output_file"
done

echo "Conversion completed."
