#!/bin/bash

search_dir="."

find "$search_dir" -type f -name "*.nii.gz" | while read -r file; do
    rm "$file"
    echo "Deleted $file"
done
