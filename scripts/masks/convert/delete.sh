#!/bin/bash

search_dir="."

find "$search_dir" -type f -name "*.mgz" | while read -r file; do
    rm "$file"
    echo "Deleted $file"
done
