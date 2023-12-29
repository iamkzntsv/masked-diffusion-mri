SHELL = /bin/bash

# Default path to Conda environment
ENV_PATH=~/miniconda3/envs/masked-diffusion-mri

# Args
PREPROCESS_ARGS=
TRAIN_ARGS=
INPAINT_ARGS=

.PHONY: create_environment remove_environment preprocess_mri train inpaint clean sample

# Default rule
all: create_environment train

create_environment:
	conda info --envs | grep $(ENV_PATH) > /dev/null || conda env create -f environment.yml --prefix $(ENV_PATH)

remove_environment:
	conda env remove --prefix $(ENV_PATH)

preprocess_mri:
	conda run --no-capture-output -p $(ENV_PATH) python -u -m masked_diffusion.etl.preprocess_mri $(PREPROCESS_ARGS)

train:
	conda run --no-capture-output -p $(ENV_PATH) python -u -m masked_diffusion.model.train $(TRAIN_ARGS)

inpaint:
	conda run --no-capture-output -p $(ENV_PATH) python -u -m masked_diffusion.model.inpaint $(INPAINT_ARGS)

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
	@echo: "create_environment		: creates conda environment"
	@echo: "remove_environment		: deletes conda environment"
	@echo: "train					: applies necessary transformations to mri volume"
	@echo: "train					: runs the training loop"
	@echo: "inpaint					: runs the inpainting"
