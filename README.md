# Lightweight Medical Image Segmentation

This repository contains code for a medical-image segmentation project
using nnU-Net and knowledge distillation.

## Repository structure

- `pipeline/`: training, knowledge-distillation, prediction, and evaluation scripts.
- `deployment/`: deployment configuration and supporting files.
- `report_assets/`: selected lightweight evaluation summaries and report assets.

## Large artifacts

Medical images, preprocessed data, model checkpoints, and prediction outputs
are not included in this code repository. Keep these in the project's
approved storage location and configure paths in `pipeline/00_settings.py`
as needed.

## Getting started

1. Review `pipeline/README.md`.
2. Set up the required Python environment using the provided requirements files.
3. Configure the data and model paths.
4. Follow the pipeline scripts in the documented order.

## Reproducibility

Check the dataset's official terms and permissions before sharing medical
images or derived data. Results may depend on the dataset version, model
checkpoints, and environment configuration.
