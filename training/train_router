"""
training/train_router.py
=========================
Trains the EfficientNet-B2 background context router on PV03 land-use categories.

Usage:
    python training/train_router.py
"""

import os, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import timm
from config import *


CLASS2IDX = {c: i for i, c in enumerate(PV03_BACKGROUNDS)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}


class RouterDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def load_samples():
    with open(ROUTER_LABELS) as f:
        labels = json.load(f)
    samples, skipped = [], 0
    for fn, cls in labels.items():
        cls = cls.strip().lower()
        if cls not in CLASS2IDX:
            skipped += 1; continue
        path = os.path.join(ROUTER_IMGS, fn)
        if not os.path.exists(path):
            skipped += 1; continue
        samples.append((path, CLASS2IDX[cls]))
    print(f"Loaded {len(samples)} samples ({skipped} skipped)")
    return samples


train_tf = transforms.Compose([
    transforms.Resize((ROUTER_SIZE, ROUTER_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])
val_tf = transforms.Compose([
    transforms.Resize((ROUTER_SIZE, ROUTER_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    samples = load_samples()
    n       = len(samples)
    n_tr    = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_te    = n - n_tr - n_val
    gen     = torch.Generator().manual_seed(SEED)
    tr_s, val_s, te_s = random_split(samples, [n_tr, n_val, n_te], generator=gen)
    print(f"Train: {len(tr_s)}  Val: {len(val_s)}  Test: {len(te_s)}")

    tr_loader  = DataLoader(RouterDataset(list(tr_s),  train_tf),
                            batch_size=BATCH_ROUTER, shuffle=True,  num_workers=4)
    val_loader = DataLoader(RouterDataset(list(val_s), val_tf),
                            batch_size=BATCH_ROUTER, shuffle=False, num_workers=4)
    te_loader  = DataLoader(RouterDataset(list(te_s),  val_tf),
                            batch_size=BATCH_ROUTER, shuffle=False, num_workers=4)

    model = timm.create_model("efficientnet_b2", pretrained=True,
                               num_classes=len(PV03_BACKGROUNDS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=LR_ROUTER, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_ROUTER, eta_min=LR_MIN)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS_ROUTER + 1):
        model.train()
        correct, total = 0, 0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward(); optimizer.step()
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += imgs.size(0)
        scheduler.step()

        # Validation
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                vc += (model(imgs).argmax(1) == labels).sum().item()
                vt += imgs.size(0)
        val_acc = vc / vt * 100
        print(f"Epoch {epoch:02d}/{EPOCHS_ROUTER} | val_acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_acc": val_acc, "classes": PV03_BACKGROUNDS},
                       CKPT_ROUTER)
            print(f"  → Saved (val_acc={val_acc:.2f}%)")

    # Test evaluation
    print("\n── Test Evaluation ──────────────────────────────")
    ckpt = torch.load(CKPT_ROUTER, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in te_loader:
            preds = model(imgs.to(device)).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) * 100
    print(f"Overall Test Accuracy: {acc:.2f}%")
    print(classification_report(all_labels, all_preds,
                                target_names=PV03_BACKGROUNDS, digits=3))

    results = {"overall_acc": round(acc, 2), "n_test": len(all_labels),
               "classes": PV03_BACKGROUNDS,
               "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist()}
    with open(os.path.join(OUTPUT_DIR, "router_eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → outputs/router_eval_results.json")


if __name__ == "__main__":
    main()
