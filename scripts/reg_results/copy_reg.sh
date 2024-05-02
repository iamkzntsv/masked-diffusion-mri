#!/bin/bash

base_dir="."

structures=(
    "amygdala"
    "caudate"
    "hippocampus"
    "lateral_ventricle"
    "pallidum"
    "putamen"
    "thalamus"
)

reg_file="reg.sh"

if [ ! -f "$reg_file" ]; then
    echo "reg.sh not found in the current directory."
    exit 1
fi

for structure in "${structures[@]}"; do
    # Left
    cp "$reg_file" "$base_dir/left_${structure}/workdir/"
    echo "Copied reg.sh to left_${structure}/workdir/"

    # Right
    cp "$reg_file" "$base_dir/right_${structure}/workdir/"
    echo "Copied reg.sh to right_${structure}/workdir/"

    # Combined
    cp "$reg_file" "$base_dir/${structure}/workdir/"
    echo "Copied reg.sh to ${structure}/workdir/"
done

echo "Finished copying reg.sh to all subdirectories."
