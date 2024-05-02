#!/bin/bash

# Directory to start the search from
search_dir="."

# Find and delete specific files
find "$search_dir" -type f \( -name "outputCPP.nii.gz" -o -name "outputResult.nii.gz" -o -name "transform.nii.gz" \) -exec rm {} \;

echo "All outputCPP, outputResult, transform files have been deleted."
