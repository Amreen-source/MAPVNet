"""
config.py — MAPVNet Global Configuration
=========================================
All paths, hyperparameters, and constants in one place.
"""

import os

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(ROOT, "data")
MODEL_ROOT = os.path.join(ROOT, "models")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

os.makedirs(MODEL_ROOT, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset paths ──────────────────────────────────────────────────────
PV08_IMG_DIR = os.path.join(DATA_ROOT, "PV08", "images")
PV08_MSK_DIR = os.path.join(DATA_ROOT, "PV08", "masks")
PV03_IMG_DIR = os.path.join(DATA_ROOT, "PV03", "images")
PV03_MSK_DIR = os.path.join(DATA_ROOT, "PV03", "masks")
PV01_IMG_DIR = os.path.join(DATA_ROOT, "PV01", "images")
PV01_MSK_DIR = os.path.join(DATA_ROOT, "PV01", "masks")
ROUTER_DIR   = os.path.join(DATA_ROOT, "router")
ROUTER_IMGS  = os.path.join(ROUTER_DIR, "images")
ROUTER_LABELS= os.path.join(ROUTER_DIR, "labels.json")

# ── Model checkpoints ──────────────────────────────────────────────────
CKPT_PV08   = os.path.join(MODEL_ROOT, "segformer_b2_pv08.pth")
CKPT_PV03   = os.path.join(MODEL_ROOT, "segformer_b4_pv03.pth")
CKPT_PV01   = os.path.join(MODEL_ROOT, "swinunet_pv01.pth")
CKPT_ROUTER = os.path.join(MODEL_ROOT, "efficientnet_b2_router.pth")
CKPT_UNIFIED= os.path.join(MODEL_ROOT, "segformer_b4_unified.pth")

# ── Data split ─────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO = 0.15 (remainder)
SEED        = 42

# ── Mask binarisation ──────────────────────────────────────────────────
# Raw pixel values in ground truth masks (binarise with pixel > 0)
MASK_VAL_PV08 = 12
MASK_VAL_PV03 = 123
MASK_VAL_PV01 = 212

# ── Image input sizes ──────────────────────────────────────────────────
IMG_SIZE_PV08   = 512   # SegFormer-B2
IMG_SIZE_PV03   = 512   # SegFormer-B4
IMG_SIZE_PV01   = 224   # Swin-UNet (Swin-B positional embedding)
ROUTER_SIZE     = 224   # EfficientNet-B2

# ── ImageNet normalisation ─────────────────────────────────────────────
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# ── Training hyperparameters (shared) ──────────────────────────────────
BATCH_SIZE    = 4
EPOCHS        = 50
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
LR_MIN        = 1e-6
WARMUP_EPOCHS = 5

# ── Router training ────────────────────────────────────────────────────
EPOCHS_ROUTER  = 30
LR_ROUTER      = 1e-4
BATCH_ROUTER   = 32

# ── Background categories ──────────────────────────────────────────────
# PV03 land-use backgrounds (router classification)
PV03_BACKGROUNDS = [
    "cropland",
    "grassland",
    "shrubland",
    "water_surface",
    "saline_alkali",
]

# ── Confidence gate ────────────────────────────────────────────────────
CONF_THRESHOLD  = 0.70
MAX_RETRIES     = 3

# ── Cross-resolution fusion weights ───────────────────────────────────
FUSION_WEIGHT_PV01 = 0.50   # Highest weight — finest resolution
FUSION_WEIGHT_PV03 = 0.35
FUSION_WEIGHT_PV08 = 0.15   # Lowest weight — coarsest resolution
FUSION_THRESHOLD   = 0.50   # Binarisation threshold for fused mask

# ── VLM (Agent 1) ──────────────────────────────────────────────────────
VLM_MODEL_ID  = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_QUANTIZE  = True        # 4-bit NF4 via BitsAndBytes
VLM_MAX_TOKENS= 512

# ── SegFormer decoder hidden size ──────────────────────────────────────
# Must match checkpoint: shape [768, 64] confirms decoder_hidden_size=768
SEGFORMER_DECODER_HIDDEN = 768
SEGFORMER_NUM_LABELS     = 1    # Binary sigmoid output
