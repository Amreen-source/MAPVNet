"""
training/train_specialists.py
==============================
Trains one of the three MAPVNet specialist segmentation models.

Usage:
    python training/train_specialists.py --resolution PV08
    python training/train_specialists.py --resolution PV03
    python training/train_specialists.py --resolution PV01
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import timm
from config import *


# ── Dataset ────────────────────────────────────────────────────────────
class PVSegDataset(Dataset):
    def __init__(self, pairs, img_size, augment=False):
        self.pairs   = pairs
        self.augment = augment
        self.img_tf  = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ])
        self.aug_tf  = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
        self.msk_tf  = transforms.Compose([
            transforms.Resize((img_size, img_size),
                interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):  return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp).convert("L")
        if self.augment:
            img = self.aug_tf(img)
        return self.img_tf(img), self.msk_tf(msk)


def make_pairs(img_dir, msk_dir):
    pairs = []
    for fn in sorted(os.listdir(img_dir)):
        ip = os.path.join(img_dir, fn)
        base, ext = os.path.splitext(fn)
        mp = os.path.join(msk_dir, f"{base}_label{ext}")
        if os.path.exists(mp):
            pairs.append((ip, mp))
    return pairs


# ── Loss ───────────────────────────────────────────────────────────────
def dice_loss(pred, target, smooth=1.0):
    pred   = torch.sigmoid(pred)
    inter  = (pred * target).sum(dim=(2, 3))
    union  = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - (2 * inter + smooth) / (union + smooth)


def combined_loss(logits, masks):
    masks = (masks > 0).float()
    bce   = F.binary_cross_entropy_with_logits(logits, masks)
    dice  = dice_loss(logits, masks).mean()
    return 0.5 * bce + 0.5 * dice


# ── IoU metric ─────────────────────────────────────────────────────────
@torch.no_grad()
def compute_iou(logits, masks):
    pred = (torch.sigmoid(logits) > 0.5).squeeze(1).long()
    true = (masks.squeeze(1) > 0).long()
    inter = ((pred == 1) & (true == 1)).float().sum((1, 2))
    union = ((pred == 1) | (true == 1)).float().sum((1, 2))
    iou   = torch.where(union > 0, inter / union, torch.ones_like(inter))
    return iou.mean().item()


# ── Build SegFormer ────────────────────────────────────────────────────
def build_segformer(backbone):
    depths = {"b2": [3, 4, 6, 3], "b4": [3, 8, 27, 3]}[backbone]
    cfg = SegformerConfig(
        num_encoder_blocks   = 4,
        depths               = depths,
        hidden_sizes         = [64, 128, 320, 512],
        num_attention_heads  = [1, 2, 5, 8],
        decoder_hidden_size  = SEGFORMER_DECODER_HIDDEN,
        num_labels           = SEGFORMER_NUM_LABELS,
    )
    return SegformerForSemanticSegmentation(cfg)


# ── Build SwinUNet ─────────────────────────────────────────────────────
def build_swinunet():
    from models.swinunet import SwinUNet
    return SwinUNet()


# ── Eval loop ──────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, is_segformer=True):
    model.eval()
    total_loss, iou_list = 0.0, []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        if is_segformer:
            logits = model(pixel_values=imgs).logits
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode="bilinear", align_corners=False)
        else:
            logits = model(imgs)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:],
                                       mode="bilinear", align_corners=False)
        total_loss += combined_loss(logits, masks).item()
        iou_list.append(compute_iou(logits, masks))
    return total_loss / len(loader), float(np.mean(iou_list))


# ── Main ───────────────────────────────────────────────────────────────
def main(resolution):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining {resolution} specialist on {device}")

    # Paths and sizes
    cfg = {
        "PV08": (PV08_IMG_DIR, PV08_MSK_DIR, IMG_SIZE_PV08, "b2", CKPT_PV08),
        "PV03": (PV03_IMG_DIR, PV03_MSK_DIR, IMG_SIZE_PV03, "b4", CKPT_PV03),
        "PV01": (PV01_IMG_DIR, PV01_MSK_DIR, IMG_SIZE_PV01, "swin", CKPT_PV01),
    }
    img_dir, msk_dir, img_size, backbone, ckpt_path = cfg[resolution]

    # Data
    pairs = make_pairs(img_dir, msk_dir)
    n     = len(pairs)
    n_tr  = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_te  = n - n_tr - n_val
    gen   = torch.Generator().manual_seed(SEED)
    tr_s, val_s, te_s = random_split(pairs, [n_tr, n_val, n_te], generator=gen)

    print(f"  Train: {len(tr_s)}  Val: {len(val_s)}  Test: {len(te_s)}")

    tr_loader  = DataLoader(PVSegDataset(list(tr_s),  img_size, augment=True),
                            batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader = DataLoader(PVSegDataset(list(val_s), img_size, augment=False),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model
    is_segformer = backbone != "swin"
    model = build_segformer(backbone) if is_segformer else build_swinunet()
    model = model.to(device)

    # Optimiser + scheduler
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR_MIN)

    # Training
    best_iou = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for imgs, masks in tr_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            if is_segformer:
                logits = model(pixel_values=imgs).logits
                logits = F.interpolate(logits, size=masks.shape[-2:],
                                       mode="bilinear", align_corners=False)
            else:
                logits = model(imgs)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:],
                                           mode="bilinear", align_corners=False)
            loss = combined_loss(logits, masks)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()

        _, val_iou = evaluate(model, val_loader, device, is_segformer)
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"tr_loss={tr_loss/len(tr_loader):.4f} | val_IoU={val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_iou": val_iou, "resolution": resolution},
                       ckpt_path)
            print(f"  → Saved best checkpoint (val_iou={val_iou:.4f})")

    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", choices=["PV08", "PV03", "PV01"],
                        required=True, help="Which specialist to train")
    args = parser.parse_args()
    main(args.resolution)
