# Installation

## 1. Create the environment

```bash
conda create -n cfm-uda-pmw-rain python=3.10 -y
conda activate cfm-uda-pmw-rain
```

## 2. Install dependencies

```bash
python -m pip install timm
python -m pip install matplotlib hydra-core
python -m pip install lightning lightning-fabric
python -m pip install tensorboard torchdyn pot einops natsort zstandard cartopy pyresample
python -m pip install -U huggingface_hub
```

## 3. Download the checkpoints

```bash
python download_checkpoints
```