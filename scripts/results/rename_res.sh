#!/bin/bash

search_dir="."

find "$search_dir" -type f -name "*.nii" | while read -r file; do
    # If file exists
    if [ -f "$file" ]; then
        new_file="${file%.nii}.nii.gz"

        # Rename the .nii file to .nii.gz
        mv "$file" "$new_file"
        echo "Renamed $file to $new_file"
    fi
done
