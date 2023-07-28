"""
Codebase for masked-diffusion-mri.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)

logging.getLogger(__name__).addHandler(logging.NullHandler())
