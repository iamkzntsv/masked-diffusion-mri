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
│       ├── custom_dataset.py   # Script to get training data from HuggingFace hub
│       ├── data_utils.py       # Script to transform raw BRATS data to processed data
│       ├── image_utils.py      # IXI dataset script
│       ├── ixi_data_module.py  # Training dataset script
│       ├── preprocess_mri.py   # MRI preprocessing script
│       └── slice_extractor.py  # MRI slice extraction script
│   ├── model                   # Scripts for model training and evaluation
│       ├── __init__.py
│       ├── train.py            # Training script
│       ├── model.py            # Diffusion model script
│       ├── inpaint.py          # Inference script
│       ├── repaint.py          # RePaint algorithm script
│       └── config.yml          # Configuration file
│   ├── __init__.py    
│   └── utils.py                # Utility functions       
└── notebooks                   # Jupyter notebooks for exploration and presentation
```

## Tumour Inpainting

## Fine-tuning