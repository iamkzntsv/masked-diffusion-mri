#!/bin/bash

base_dir="."

structures=("amygdala" "caudate" "hippocampus" "lateral_ventricle" "pallidum" "putamen" "thalamus")

t1_path="$base_dir/t1.mgz"

if [ ! -f "$t1_path" ]; then
    echo "t1.mgz not found in the current directory."
    exit 1
fi

mkdir -p "$base_dir"

# Create directories for each structure and copy t1.mgz into their workdir
for structure in "${structures[@]}"; do
    # Create directories for left, right, and full structures
    mkdir -p "$base_dir/left_${structure}/workdir"
    mkdir -p "$base_dir/right_${structure}/workdir"
    mkdir -p "$base_dir/${structure}/workdir"

    # Copy t1.mgz to each workdir
    cp "$t1_path" "$base_dir/left_${structure}/workdir/"
    cp "$t1_path" "$base_dir/right_${structure}/workdir/"
    cp "$t1_path" "$base_dir/${structure}/workdir/"

    # Find and copy the specific slice file for left, right, and combined structures
    left_slice_file=$(find . -maxdepth 1 -type f -name "left_${structure}_slice_*.nii.gz")
    right_slice_file=$(find . -maxdepth 1 -type f -name "right_${structure}_slice_*.nii.gz")
    combined_slice_file=$(find . -maxdepth 1 -type f -name "${structure}_slice_*.nii.gz")

    if [ -n "$left_slice_file" ]; then
        cp "$left_slice_file" "$base_dir/left_${structure}/workdir/"
    fi

    if [ -n "$right_slice_file" ]; then
        cp "$right_slice_file" "$base_dir/right_${structure}/workdir/"
    fi

    if [ -n "$combined_slice_file" ]; then
        cp "$combined_slice_file" "$base_dir/${structure}/workdir/"
    fi

    echo "Processed $structure"
done

echo "All directories created and files copied."
