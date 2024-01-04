SHELL = /bin/bash

# Default path to Conda environment
ENV_PATH=~/miniconda3/envs/masked-diffusion-mri

# Docker image
DOCKER_TAG=masked-diffusion-mri:latest

# Args
PREPROCESS_ARGS=
TRAIN_ARGS=
INPAINT_ARGS=

.PHONY: build create_env remove_env preprocess_mri train inpaint clean sample

# Default rule
all: build train

build:
	docker build --no-cache . -f Dockerfile -t $(DOCKER_TAG)

create_env:
	conda info --envs | grep $(ENV_PATH) > /dev/null || conda env create -f environment.yml --prefix $(ENV_PATH)

remove_env:
	conda env remove --prefix $(ENV_PATH)


preprocess_mri:
	docker run $(DOCKER_TAG) conda run --no-capture-output -n masked-diffusion-mri python -u -m masked_diffusion.etl.preprocess_mri $(PREPROCESS_ARGS)

train:
	docker run $(DOCKER_TAG) conda run --no-capture-output -n masked-diffusion-mri python -u -m masked_diffusion.model.train $(TRAIN_ARGS)

inpaint:
	docker run $(DOCKER_TAG) conda run --no-capture-output -n masked-diffusion-mri python -u -m masked_diffusion.model.inpaint $(INPAINT_ARGS)

clean: style
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -f .make.*

style:
	black .
	flake8
	isort .

help:
	@echo "Commands":
	@echo: "build					: builds docker image"
	@echo: "create_env				: creates conda environment"
	@echo: "remove_env				: deletes conda environment"
	@echo: "preprocess				: applies necessary transformations to mri volume"
	@echo: "train					: runs the training loop"
	@echo: "inpaint					: runs the inpainting"