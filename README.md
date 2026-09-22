# MAPVNet: A Resolution-Aware Multi-Agent Framework for Multi-Resolution Photovoltaic Panel Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11.15-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/pytorch-2.5.1-orange.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Paper:** MAPVNet: A Resolution-Aware Multi-Agent Framework for Photovoltaic Panel Detection from Multi-Resolution Remote Sensing Imagery  
> **Authors:** Amreen Batool, Yong-Woon Kim, Yung-Cheol Byun  
> **Institution:** Jeju National University, Republic of Korea  
> **Dataset:** [Jiang et al. 2021 Multi-Resolution PV Dataset](https://doi.org/10.5281/zenodo.5171712)

---

## Overview

MAPVNet is a **resolution-aware multi-agent AI framework** for photovoltaic (PV) panel segmentation across heterogeneous satellite, aerial, and UAV imagery.

Instead of training a single model to handle substantially different spatial resolutions, MAPVNet coordinates dedicated resolution-specific specialists through a multi-agent workflow.

The framework integrates:

- Vision-language-based orchestration
- Resolution-aware routing
- Scene-context classification
- Resolution-specific PV segmentation
- Confidence-aware re-processing
- Geospatial output generation
- UAV anomaly assessment
- Automated operational reporting

---

## MAPVNet Multi-Agent Architecture

MAPVNet coordinates six functional agents together with a confidence-aware processing gate.

| Agent | Role | Model / Tool |
|---|---|---|
| **Agent 1** | VLM Orchestrator | Qwen2.5-VL-7B |
| **Agent 2** | Resolution & Context Router | Deterministic GSD Routing + EfficientNet-B2 |
| **Agent 3a** | PV08 Satellite Specialist | SegFormer-B2 |
| **Agent 3b** | PV03 Aerial Specialist | SegFormer-B4 |
| **Agent 3c** | PV01 UAV Specialist | Swin-UNet |
| **Agent 4** | Geospatial Processing | Rasterio + GDAL |
| **Agent 5** | UAV Anomaly Assessment | PatchCore / Anomalib |
| **Agent 6** | Operational Report Generation | JSON + TXT + Folium + GeoTIFF |

### Resolution-Specific Specialists

| Dataset | Sensor | GSD | Specialist |
|---|---|---:|---|
| **PV08** | Satellite | 0.8 m | SegFormer-B2 |
| **PV03** | Aerial | 0.3 m | SegFormer-B4 |
| **PV01** | UAV | 0.1 m | Swin-UNet |

---

## Dataset

MAPVNet is evaluated using the multi-resolution photovoltaic dataset introduced by **Jiang et al. (2021)**.

Dataset DOI:

https://doi.org/10.5281/zenodo.5171712

The dataset contains **3,716 image-mask pairs** covering three sensing platforms.

| Subset | Sensor | GSD | Total | Train | Validation | Test |
|---|---|---:|---:|---:|---:|---:|
| **PV08** | Gaofen-2 / Beijing-2 Satellite | 0.8 m | 763 | 534 | 114 | 115 |
| **PV03** | Aerial Photography | 0.3 m | 2,308 | 1,615 | 346 | 347 |
| **PV01** | UAV Orthophoto | 0.1 m | 645 | 451 | 96 | 98 |
| **Total** | 3 Platforms | — | **3,716** | **2,600** | **556** | **560** |

The dataset is divided using a fixed **70/15/15 train-validation-test split** with random seed **42**.

Ground-truth masks are converted into binary PV/background masks.

---

## Expected Dataset Structure

```text
data/
├── PV08/
│   ├── images/       # 763 images
│   └── masks/        # 763 masks
│
├── PV03/
│   ├── images/       # 2,308 images
│   └── masks/        # 2,308 masks
│
├── PV01/
│   ├── images/       # 645 images
│   └── masks/        # 645 masks
│
└── router/
## Training

### Step 1 — Train the Background Context Router

```bash
python training/train_router.py
```

### Step 2 — Train the Resolution-Specific Specialist Models

**PV08 — SegFormer-B2**
```bash
python training/train_specialists.py --resolution PV08
```

**PV03 — SegFormer-B4**
```bash
python training/train_specialists.py --resolution PV03
```

**PV01 — Swin-UNet**
```bash
python training/train_specialists.py --resolution PV01
```

### Step 3 — Train the Unified Baseline

```bash
python training/train_unified_baseline.py
```

---

## Evaluation

### Evaluate the Specialist Models

```bash
python evaluation/eval_specialists.py
```

### Evaluate the Context Router

```bash
python evaluation/eval_router.py
```

### Evaluate the Unified Baseline

```bash
python evaluation/eval_unified_baseline.py
```

### Run the Full MAPVNet Pipeline

```bash
python pipeline.py --image path/to/image.bmp
```

---

## Pretrained Checkpoints

| Model | Checkpoint | Best Validation Performance |
|---|---|---:|
| SegFormer-B2 (PV08) | `checkpoints/segformer_b2_pv08.pth` | IoU = **0.8642** |
| SegFormer-B4 (PV03) | `checkpoints/segformer_b4_pv03.pth` | IoU = **0.9082** |
| Swin-UNet (PV01) | `checkpoints/swinunet_pv01.pth` | IoU = **0.9208** |
| EfficientNet-B2 (Context Router) | `checkpoints/efficientnet_b2_router.pth` | Accuracy = **88.20%** |
| Unified SegFormer-B4 | `checkpoints/segformer_b4_unified.pth` | Mean Test IoU = **0.8156** |

The specialist checkpoints correspond to the best validation checkpoints selected during training.

Large pretrained model files are not stored directly in the Git repository. Download links and checkpoint hashes will be provided with the public model release.

---

## Citation

If you use MAPVNet in your research, please cite:

```bibtex
@article{batool2026mapvnet,
  title   = {MAPVNet: A Resolution-Aware Multi-Agent Framework for Photovoltaic Panel Detection from Multi-Resolution Remote Sensing Imagery},
  author  = {Batool, Amreen and Kim, Yong-Woon and Byun, Yung-Cheol},
  journal = {To appear},
  year    = {2026}
}
```

The citation will be updated with the journal name, DOI, volume, issue, and page numbers after publication.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grants funded by the Korean government (MSIT):

- **RS-2024-00405278**
- **RS-2026-25470261**

---

## Contact

For questions regarding MAPVNet, reproducibility, or research collaboration, please open an issue in this repository or contact the corresponding author.
    ├── images/
    └── labels.json
