# Masked Diffusion Models to Predict Morphological Relationships from Brain MRI. 

## Project Structure
```
├── README.md                   # Project description
├── environment.yml             # Conda environment file
├── Makefile                    # File with directives for the `make` tool
├── setup.py                    # Build script for setuptools
├── data
│   ├── interim                 # Interim data
│   ├── raw                     # Raw data
│   └── transformed             # Processed data
├── masked_diffusion            # Project working files
│   ├── etl                     # Scripts for extracting, transforming, and loading data
│       ├── __init__.py
│       ├── custom_dataset.py   # Data processing script
│       ├── data_utils.py       # Data utility functions
│       ├── image_utils.py      # Image utility functions
│       ├── ixi_data_module.py  # Training dataset loading script
│       ├── preprocess_mri.py   # MRI preprocessing script
│       └── slice_extractor.py  # MRI slice extraction script
│   ├── model                   # Scripts for model training and evaluation
│       ├── __init__.py
│       ├── train.py            # Training script
│       ├── model.py            # Diffusion model implementation script
│       ├── inpaint.py          # Inference script
│       ├── repaint.py          # RePaint algorithm script
│       └── config.yml          # Configuration file
│   ├── __init__.py    
│   └── utils.py                # Utility functions       
└── notebooks                   # Jupyter notebooks for exploration and presentation
```

## Download Pre-trained Model
The model parameters can be downloaded from [here](https://drive.google.com/uc?export=download&id=18dmQbZiqBKh6ilC0HsXxM8ikHWDyyUJs). Please put them into `masked_diffusion/model/pretrained`.

## MRI Preprocessing
Prior to applying our trained model to your MRI data, it's crucial to undergo specific preprocessing steps. 
Note: Before running the script make sure to perform skull-stripping and registration using FreeSurfer or a similar MRI processing tool (following [this](https://github.com/iamkzntsv/self-supervised-learning-mri/blob/master/preprocessing.md]) procedure).

```
make preprocess_mri PREPROCESS_ARGS="--path /your_subj_path --save_dir /your_save_dir --batch_size 1 --offset 15"
```
## Tumour Inpainting
To perform MRI image inpainting run the following command on a `.mgz` MRI file obtained from MRI preprocessing step.
```
make inpaint INPAINT_ARGS="--path /mri_path --batch_size 1 --num_inference_steps 250 --jump_length 10 --jump_n_sample 10"
```