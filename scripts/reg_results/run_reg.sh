#!/bin/bash

base_dir="/{path}/masked-diffusion-mri-reg/hcp/{subject}/"

# Find 'reg.sh'
find "$base_dir" -type f -name "reg.sh" | while read -r reg_file; do
    # Get the directory of the reg.sh file
    reg_dir=$(dirname "$reg_file")

    # Change to the directory where reg.sh is located
    cd "$reg_dir" || exit

    # Run reg.sh in its directory
    echo "Running reg.sh in $reg_dir"
    chmod +x "$reg_file"
    ./reg.sh

    # Change back to the base
    cd - > /dev/null

    echo "Finished running reg.sh in $reg_dir"
done

echo "All reg.sh scripts have been processed."
