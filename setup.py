from setuptools import find_packages, setup

setup(
    name="masked-diffusion-mri",
    version="0.0.1",
    author="iamkzntsv",
    author_email="iamkzntsv.ai@gmail.com",
    description="",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
    ],
)
