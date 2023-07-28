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
├── masked_diffusion            # TO DO
│   ├── etl                     # Scripts for extracting, transforming, and loading data
│       ├── __init__.py
│       ├── get_data.py         # Script to get training data from HuggingFace hub
│       ├── transform_brats.py  # Script to transform raw BRATS data to processed data
│       ├── ixi_dataset.py      # IXI dataset script
│       ├── brats_dataset.py    # BRATS dataset script
│       └── brats_dataloader.py # BRATS dataloader script
│   ├── model                   # Scripts for model training and evaluation
│       ├── __init__.py
│       ├── train.py            # Training script
│       ├── infer.py            # Inference script
│       ├── repaint.py          # RePaint model script
│       └── config.yml          # Configuration file
│   ├── __init__.py    
│   └── utils.py                # Utility functions       
└── notebooks                   # Jupyter notebooks for exploration and presentation
```