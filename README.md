# MAPVNet: A Multi-Agent AI Framework for Multi-Resolution Photovoltaic Panel Detection

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1](https://img.shields.io/badge/pytorch-2.1.2-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Paper:** MAPVNet: A Multi-Agent AI Framework for Multi-Resolution Photovoltaic Panel Detection from Satellite, Aerial, and UAV Imagery  
> **Authors:** Amreen Batool, Yong-Woon Kim, Yung-Cheol Byun  
> **Institution:** Jeju National University, South Korea  
> **Dataset:** [Jiang et al. 2021](https://doi.org/10.5281/zenodo.5171712)

---

## Overview

MAPVNet coordinates **six specialised agents** for multi-resolution PV panel detection:

| Agent | Role | Model |
|-------|------|-------|
| Agent 1 | VLM Orchestrator | Qwen2.5-VL-7B |
| Agent 2 | Resolution & Context Router | EfficientNet-B2 |
| Agent 3a | PV08 Specialist (0.8 m satellite) | SegFormer-B2 |
| Agent 3b | PV03 Specialist (0.3 m aerial) | SegFormer-B4 |
| Agent 3c | PV01 Specialist (0.1 m UAV) | Swin-UNet |
| Agent 4 | Cross-Resolution Fusion | Rasterio + GDAL |
| Agent 5 | Anomaly Detection | PatchCore |
| Agent 6 | Report Generation | — |

## Results

| Model | PV08 IoU | PV03 IoU | PV01 IoU | Mean IoU |
|-------|----------|----------|----------|----------|
| MAPVNet (Ours) | **0.8052** | **0.8569** | **0.9020** | **0.8547** |
| Kleebauer et al. 2023 | 0.8234 | 0.8512 | 0.8198 | 0.8315 |
| SegFormer-B4 (unified) | 0.7812 | 0.8850 | 0.8102 | 0.8255 |

## Installation

```bash
git clone https://github.com/Amreen-source/MAPVNet.git
cd MAPVNet
conda create -n pv_env python=3.10
conda activate pv_env
pip install -r requirements.txt
```

## Dataset

Download the Jiang et al. 2021 dataset from Zenodo:
```bash
wget https://zenodo.org/record/5171712/files/PV_dataset.zip
unzip PV_dataset.zip -d data/
```

Expected structure:
```
data/
├── PV08/
│   ├── images/   # 763 .bmp files
│   └── masks/    # 763 _label.bmp files
├── PV03/
│   ├── images/   # 2308 .bmp files
│   └── masks/    # 2308 _label.bmp files
├── PV01/
│   ├── images/   # 645 .bmp files
│   └── masks/    # 645 _label.bmp files
└── router/
    ├── images/   # 2072 PV03 .bmp files
    └── labels.json
```

## Training

**Step 1 — Train the background context router:**
```bash
python training/train_router.py
```

**Step 2 — Train the three specialist models:**
```bash
python training/train_specialists.py --resolution PV08
python training/train_specialists.py --resolution PV03
python training/train_specialists.py --resolution PV01
```

## Evaluation

**Evaluate specialists (produces real IoU values):**
```bash
python evaluation/eval_specialists.py
```

**Evaluate router:**
```bash
python evaluation/eval_router.py
```

**Run full pipeline on a single image:**
```bash
python pipeline.py --image path/to/image.bmp
```

## Pretrained Checkpoints

| Model | Checkpoint | Val IoU |
|-------|-----------|---------|
| SegFormer-B2 (PV08) | `models/segformer_b2_pv08.pth` | 0.8254 |
| SegFormer-B4 (PV03) | `models/segformer_b4_pv03.pth` | 0.8613 |
| Swin-UNet (PV01) | `models/swinunet_pv01.pth` | 0.9030 |
| EfficientNet-B2 (Router) | `models/efficientnet_b2_router.pth` | 75.48% |

Checkpoints available upon paper acceptance.

## Citation

```bibtex
@article{batool2025mapvnet,
  title   = {MAPVNet: A Multi-Agent AI Framework for Multi-Resolution
             Photovoltaic Panel Detection from Satellite, Aerial, and UAV Imagery},
  author  = {Batool, Amreen and Kim, Yong-Woon and Byun, Yung-Cheol},
  journal = {Applied Energy},
  year    = {2025}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

Supported by the National Research Foundation of Korea (NRF) grant RS-2024-00405278
and the Regional Innovation System Education (RISE) program 2026-RISE-17-001.
