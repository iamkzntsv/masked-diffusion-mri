SHELL = /bin/bash

# Docker image
DOCKER_TAG=masked-diffusion-mri:latest

# Args
WEIGHTS_PATH=
DATA_PATH=
SAVE_DIR=
PREPROCESS_ARGS=
INPAINT_ARGS=
TRAIN_ARGS=
GPU_ID=0

.PHONY: get_weights build preprocess_mri train inpaint clean sample

# Default rule
all: get_weights build preprocess_mri inpaint

get_weights:
	python -u -m get_weights $(SAVE_DIR)

build:
	docker build --no-cache . -f Dockerfile -t $(DOCKER_TAG)

preprocess_mri:
	docker run \
    -v $(DATA_PATH):/app/data/new/raw \
    -v $(SAVE_DIR):/app/data/new/processed \
    $(DOCKER_TAG) python -m masked_diffusion.etl.preprocess_mri $(PREPROCESS_ARGS)



inpaint:
	docker run --gpus '"device=$(GPU_ID)"' \
	-v $(DATA_PATH):/app/data/new/processed \
    -v $(SAVE_DIR):/app/data/new/processed \
    -v $(WEIGHTS_PATH):/app/masked_diffusion/model/pretrained \
    -m 8g \
    $(DOCKER_TAG) python -u -m masked_diffusion.model.inpaint $(INPAINT_ARGS)


train:
	docker run $(DOCKER_TAG) python -u -m masked_diffusion.model.train $(TRAIN_ARGS)

clean: style
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -f .make.*

style:
	black .
	flake8
	isort .

help:
	@echo "Commands:"
	@echo "get_weights				: download model weights"
	@echo "build					: builds docker image"
	@echo "preprocess_mri			: applies necessary transformations to mri volume"
	@echo "train					: runs the training loop"
	@echo "inpaint					: runs the inpainting"
