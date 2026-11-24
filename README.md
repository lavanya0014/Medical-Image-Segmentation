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

## Novelty assessment: ~5 M-parameter student

Scripts and results are in `novelty_assessment/`. We trained three 0.4×-width nnU-Net students for 10 epochs on the same CAS2023 split, using one seed per model. The teacher checkpoint is included as a reference.

| Model | Parameters (M) | Dice | IoU |
|---|---:|---:|---:|
| Teacher (reference) | 30.79 | 0.8040 | 0.6743 |
| Scratch student | 4.92 | 0.8144 | 0.6885 |
| C1 output KD | 4.92 | 0.8020 | 0.6716 |
| C2 feature KD | 4.92 | 0.8225 | 0.7002 |

INT8 quantization reduced the model file from 18.79 MB to 5.43 MB.

[Open the live demo](https://lava199-cas2023-demo.static.hf.space/)
