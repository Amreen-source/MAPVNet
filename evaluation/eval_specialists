"""
evaluation/eval_specialists.py
================================
Evaluates all three trained specialist models on their test splits.
Produces real IoU, F1, Precision, Recall values for Table 2.

Usage:
    python evaluation/eval_specialists.py
"""

import os, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from config import *


# ── Dataset ────────────────────────────────────────────────────────────
class PVDataset(Dataset):
    def __init__(self, pairs, img_size):
        self.pairs = pairs
        self.itf   = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ])
        self.mtf   = transforms.Compose([
            transforms.Resize((img_size, img_size),
                interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
    def __len__(self):  return len(self.pairs)
    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        return self.itf(Image.open(ip).convert("RGB")), \
               self.mtf(Image.open(mp).convert("L"))


def make_pairs(img_dir, msk_dir):
    pairs = []
    for fn in sorted(os.listdir(img_dir)):
        ip = os.path.join(img_dir, fn)
        b, e = os.path.splitext(fn)
        mp = os.path.join(msk_dir, f"{b}_label{e}")
        if os.path.exists(mp): pairs.append((ip, mp))
    return pairs


def get_test_loader(img_dir, msk_dir, img_size):
    pairs = make_pairs(img_dir, msk_dir)
    n     = len(pairs)
    n_tr  = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_te  = n - n_tr - n_val
    gen   = torch.Generator().manual_seed(SEED)
    _, _, te = random_split(pairs, [n_tr, n_val, n_te], generator=gen)
    print(f"  Test split: {len(te)} samples")
    return DataLoader(PVDataset(list(te), img_size),
                      batch_size=8, shuffle=False, num_workers=4)


# ── Metrics ────────────────────────────────────────────────────────────
def compute_metrics(logits, masks):
    """Binary segmentation metrics from sigmoid logits."""
    pred = (torch.sigmoid(logits) > 0.5).squeeze(1).long()
    true = (masks.squeeze(1) > 0).long()

    tp = ((pred == 1) & (true == 1)).float().sum((1, 2))
    fp = ((pred == 1) & (true == 0)).float().sum((1, 2))
    fn = ((pred == 0) & (true == 1)).float().sum((1, 2))
    tn = ((pred == 0) & (true == 0)).float().sum((1, 2))

    iou  = torch.where(tp + fp + fn > 0, tp / (tp + fp + fn),
                       torch.ones_like(tp))
    prec = torch.where(tp + fp > 0, tp / (tp + fp), torch.ones_like(tp))
    rec  = torch.where(tp + fn > 0, tp / (tp + fn), torch.ones_like(tp))
    f1   = torch.where(prec + rec > 0,
                       2 * prec * rec / (prec + rec), torch.zeros_like(tp))
    return iou.mean().item(), f1.mean().item(), prec.mean().item(), rec.mean().item()


# ── Model loaders ──────────────────────────────────────────────────────
def load_segformer(ckpt_path, backbone):
    depths = {"b2": [3, 4, 6, 3], "b4": [3, 8, 27, 3]}[backbone]
    cfg = SegformerConfig(
        num_encoder_blocks=4, depths=depths,
        hidden_sizes=[64, 128, 320, 512], num_attention_heads=[1, 2, 5, 8],
        decoder_hidden_size=SEGFORMER_DECODER_HIDDEN,
        num_labels=SEGFORMER_NUM_LABELS,
    )
    model = SegformerForSemanticSegmentation(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state: state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    print(f"  Loaded {os.path.basename(ckpt_path)} ✓")
    return model


def load_swinunet(ckpt_path):
    from models.swinunet import SwinUNet
    model = SwinUNet()
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state: state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    print(f"  Loaded {os.path.basename(ckpt_path)} ✓")
    return model


# ── Evaluation loop ────────────────────────────────────────────────────
@torch.no_grad()
def eval_segformer(model, loader, device):
    model.eval().to(device)
    results = []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(pixel_values=imgs).logits
        logits = F.interpolate(logits, size=masks.shape[-2:],
                               mode="bilinear", align_corners=False)
        results.append(compute_metrics(logits, masks))
    return tuple(float(np.mean([r[i] for r in results])) for i in range(4))


@torch.no_grad()
def eval_swinunet(model, loader, device):
    model.eval().to(device)
    results = []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode="bilinear", align_corners=False)
        results.append(compute_metrics(logits, masks))
    return tuple(float(np.mean([r[i] for r in results])) for i in range(4))


# ── Main ───────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    all_results = {}

    # PV08 — SegFormer-B2
    print("── PV08 (SegFormer-B2) ─────────────────────")
    loader = get_test_loader(PV08_IMG_DIR, PV08_MSK_DIR, IMG_SIZE_PV08)
    model  = load_segformer(CKPT_PV08, "b2")
    iou, f1, prec, rec = eval_segformer(model, loader, device)
    all_results["PV08"] = {"IoU": round(iou, 4), "F1": round(f1, 4),
                           "Precision": round(prec, 4), "Recall": round(rec, 4)}
    print(f"  IoU={iou:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    del model; torch.cuda.empty_cache()

    # PV03 — SegFormer-B4
    print("\n── PV03 (SegFormer-B4) ─────────────────────")
    loader = get_test_loader(PV03_IMG_DIR, PV03_MSK_DIR, IMG_SIZE_PV03)
    model  = load_segformer(CKPT_PV03, "b4")
    iou, f1, prec, rec = eval_segformer(model, loader, device)
    all_results["PV03"] = {"IoU": round(iou, 4), "F1": round(f1, 4),
                           "Precision": round(prec, 4), "Recall": round(rec, 4)}
    print(f"  IoU={iou:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    del model; torch.cuda.empty_cache()

    # PV01 — Swin-UNet
    print("\n── PV01 (Swin-UNet) ────────────────────────")
    loader = get_test_loader(PV01_IMG_DIR, PV01_MSK_DIR, IMG_SIZE_PV01)
    model  = load_swinunet(CKPT_PV01)
    iou, f1, prec, rec = eval_swinunet(model, loader, device)
    all_results["PV01"] = {"IoU": round(iou, 4), "F1": round(f1, 4),
                           "Precision": round(prec, 4), "Recall": round(rec, 4)}
    print(f"  IoU={iou:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    del model; torch.cuda.empty_cache()

    # Mean
    mean_iou  = np.mean([v["IoU"]  for v in all_results.values()])
    mean_f1   = np.mean([v["F1"]   for v in all_results.values()])
    mean_prec = np.mean([v["Precision"] for v in all_results.values()])
    mean_rec  = np.mean([v["Recall"]    for v in all_results.values()])
    all_results["Mean"] = {"IoU": round(float(mean_iou),  4),
                           "F1":  round(float(mean_f1),   4),
                           "Precision": round(float(mean_prec), 4),
                           "Recall":    round(float(mean_rec),  4)}

    print("\n══ FINAL RESULTS ══════════════════════════════")
    for k, v in all_results.items():
        print(f"  {k}: IoU={v['IoU']}  F1={v['F1']}  "
              f"Prec={v['Precision']}  Rec={v['Recall']}")

    out_path = os.path.join(OUTPUT_DIR, "specialist_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
