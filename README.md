# MAPVNet: A Resolution-Aware Multi-Agent Framework for Multi-Resolution Photovoltaic Panel Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1](https://img.shields.io/badge/pytorch-2.5.1-orange.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Paper:** MAPVNet: A Resolution-Aware Multi-Agent Framework for Photovoltaic Panel Detection from Multi-Resolution Remote Sensing Imagery  
> **Authors:** Amreen Batool, Yong-Woon Kim, Yung-Cheol Byun  
> **Institution:** Jeju National University, Republic of Korea  
> **Dataset:** [Jiang et al. 2021](https://doi.org/10.5281/zenodo.5171712)

---

## Overview

MAPVNet is a **resolution-aware multi-agent AI framework** designed for photovoltaic (PV) panel segmentation across heterogeneous satellite, aerial, and UAV remote-sensing imagery.

The framework coordinates six functional agents for visual orchestration, resolution and context-aware routing, resolution-specific segmentation, geospatial processing, UAV anomaly assessment, and automated reporting.

| Agent | Role | Model / Tool |
|---|---|---|
| Agent 1 | VLM Orchestrator | Qwen2.5-VL-7B |
| Agent 2 | Resolution & Context Router | EfficientNet-B2 + deterministic GSD routing |
| Agent 3a | PV08 Specialist — Satellite, 0.8 m GSD | SegFormer-B2 |
| Agent 3b | PV03 Specialist — Aerial, 0.3 m GSD | SegFormer-B4 |
| Agent 3c | PV01 Specialist — UAV, 0.1 m GSD | Swin-UNet |
| Agent 4 | Geospatial Processing | Rasterio + GDAL |
| Agent 5 | UAV Anomaly Assessment | PatchCore / Anomalib |
| Agent 6 | Operational Report Generation | JSON + TXT + GeoTIFF + interactive map |

A confidence gate is applied after specialist segmentation. Predictions below the operational threshold can enter a re-processing loop using CLAHE and test-time augmentation.

---

## Dataset

Experiments use the publicly available **Jiang et al. (2021) multi-resolution photovoltaic dataset**.

The dataset contains **3,716 image-mask pairs** covering three sensing resolutions:

| Subset | Sensor | GSD | Total | Train | Validation | Test |
|---|---|---:|---:|---:|---:|---:|
| PV08 | Gaofen-2 / Beijing-2 satellite | 0.8 m | 763 | 534 | 114 | 115 |
| PV03 | Aerial photography | 0.3 m | 2,308 | 1,615 | 346 | 347 |
| PV01 | UAV orthophoto | 0.1 m | 645 | 451 | 96 | 98 |
| **Total** | **3 platforms** | — | **3,716** | **2,600** | **556** | **560** |

All partitions use a fixed random seed of **42**.

Download the dataset from:

https://doi.org/10.5281/zenodo.5171712

Expected structure:

```text
data/
├── PV08/
│   ├── images/
│   └── masks/
├── PV03/
│   ├── images/
│   └── masks/
└── PV01/
    ├── images/
    └── masks/
