# cfm-uda-pmw-rain

Code for the paper
[*Unsupervised Domain Adaptation For Enhanced Radiometer Image Precipitation Estimation Using Conditional Flow Matching*](https://hal.science/hal-05650358/document) accepted at ICIP2026.

Conditional flow matching for unsupervised domain adaptation of passive microwave (PMW) rain retrieval.

This codebase is built on [TorchCFM](https://github.com/atong01/conditional-flow-matching) — many thanks to its authors.

## Release notes

A few changes were made for the release version, relative to earlier experiments:

- **Ground truth** — switched from [GPM DPR](https://search.earthdata.nasa.gov/search/granules?p=C2179081499-GES_DISC&pg[0][v]=f&pg[0][gsk]=-start_date&q=GPM%20DPR)
  to [GPM CORRA KuKa](https://search.earthdata.nasa.gov/search/granules?p=C2179081553-GES_DISC&pg[0][v]=f&pg[0][gsk]=-start_date),
  as CORRA KuKa is accepted as a better ground truth for rain estimation.
- **Image size** — switched from 128×128 to 32×32, which gives equivalent results at a lower
  compute cost.
- **Data split** — trained on 2020 + 2021, tested on 2019.
- **Stochastic interpolants** — this code also implements some stochastic interpolant variants, selected with the `cfm_used` and `si_gamma_fn`.

## Installation

1. Create and set up the conda environment by following the steps in [install.md](install.md).
2. Download the pretrained checkpoints:

   ```bash
   python download_checkpoints.py
   ```

   This pulls `huggingface_downloads.zip` from the
   [`vicene/cfm-uda-pmw-rain`](https://huggingface.co/datasets/vicene/cfm-uda-pmw-rain) dataset
   repo and extracts it to `./huggingface_downloads/`. The step is skipped if that directory
   already exists.

## Inference

Run inference with a pretrained model:

```bash
./run_inference.sh
```

## Training

Train a model from scratch:

```bash
./run_train.sh
```

Note that the training data is not provided, so to train your own model you need to set up the data pipeline yourself.
