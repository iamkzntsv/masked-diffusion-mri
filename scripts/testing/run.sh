#!/bin/bash

# Base directory
base_dir="/{path}/masked-diffusion-mri/data/hcp/{subject}/structures"
weights_dir="/{path}/masked-diffusion-mri/masked_diffusion/model/pretrained"

# Arguments
preprocess_args="--offset 15"
inpaint_args="--batch_size 1 --num_inference_steps 250 --jump_length 10 --jump_n_sample 10"

structures=(
    "amygdala"
    "caudate"
    "hippocampus"
    "lateral_ventricle"
    "pallidum"
    "putamen"
    "thalamus"
)

#make get_weights SAVE_DIR="--path $weights_dir"

for structure in "${structures[@]}"; do
    # Process left
    left_data_path="$base_dir/left_${structure}/workdir"
    make preprocess_mri DATA_PATH="$left_data_path" SAVE_DIR="$left_data_path" PREPROCESS_ARGS="$preprocess_args"
    make inpaint DATA_PATH="$left_data_path" WEIGHTS_PATH="$weights_dir" SAVE_DIR="$left_data_path" INPAINT_ARGS="$inpaint_args" GPU_ID=0

    # Process right
    right_data_path="$base_dir/right_${structure}/workdir"
    make preprocess_mri DATA_PATH="$right_data_path" SAVE_DIR="$right_data_path" PREPROCESS_ARGS="$preprocess_args"
    make inpaint DATA_PATH="$right_data_path" WEIGHTS_PATH="$weights_dir" SAVE_DIR="$right_data_path" INPAINT_ARGS="$inpaint_args" GPU_ID=0

    # Process combined
    combined_data_path="$base_dir/${structure}/workdir"
    make preprocess_mri DATA_PATH="$combined_data_path" SAVE_DIR="$combined_data_path" PREPROCESS_ARGS="$preprocess_args"
    make inpaint DATA_PATH="$combined_data_path" WEIGHTS_PATH="$weights_dir" SAVE_DIR="$combined_data_path" INPAINT_ARGS="$inpaint_args" GPU_ID=0
done