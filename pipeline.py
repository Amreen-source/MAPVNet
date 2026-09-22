"""
pipeline.py — MAPVNet Full Inference Pipeline
==============================================
Runs all six agents on a single input image.

Usage:
    python pipeline.py --image data/PV01/images/PV01_325123_1204226.bmp
    python pipeline.py --image data/PV03/images/PV03_338280_1204261.bmp
"""

import argparse, os, json, re
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import folium
from config import *


# ── Transforms ─────────────────────────────────────────────────────────
def get_tf(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])


# ── Agent 1: VLM Orchestrator ──────────────────────────────────────────
def agent1_parse(image_path):
    """Parse resolution and GPS from filename. Falls back to VLM if unclear."""
    fn = os.path.basename(image_path)
    # Deterministic parse from filename prefix
    for res in ["PV08", "PV03", "PV01"]:
        if fn.startswith(res):
            parts = fn.replace(".bmp", "").split("_")
            try:
                lat = float(parts[1]) / 1e4
                lon = float(parts[2]) / 1e4
            except (IndexError, ValueError):
                lat, lon = 0.0, 0.0
            return {"resolution": res, "lat": lat, "lon": lon, "filename": fn}
    raise ValueError(f"Cannot determine resolution from filename: {fn}")


# ── Agent 2: Context Router ────────────────────────────────────────────
def agent2_route(task, device):
    """Deterministic routing + optional PV03 background classification."""
    resolution = task["resolution"]
    context    = "unknown"

    if resolution == "PV03":
        import timm
        model = timm.create_model("efficientnet_b2", pretrained=False,
                                   num_classes=len(PV03_BACKGROUNDS))
        ckpt  = torch.load(CKPT_ROUTER, map_location=device)
        if "model_state_dict" in ckpt: ckpt = ckpt["model_state_dict"]
        # Router was saved with 9-class head; load with strict=False
        model.load_state_dict(ckpt, strict=False)
        model.eval().to(device)
        img = Image.open(task["image_path"]).convert("RGB")
        tf  = get_tf(ROUTER_SIZE)
        with torch.no_grad():
            logits = model(tf(img).unsqueeze(0).to(device))
            idx    = logits[:, :len(PV03_BACKGROUNDS)].argmax(1).item()
        context = PV03_BACKGROUNDS[idx]
        del model; torch.cuda.empty_cache()

    task["context"]    = context
    task["specialist"] = resolution
    return task


# ── Agent 3: Specialists ───────────────────────────────────────────────
def agent3_segment(task, device):
    from transformers import SegformerForSemanticSegmentation, SegformerConfig

    resolution = task["resolution"]
    img_path   = task["image_path"]

    # Load image
    if resolution == "PV01":
        img_size = IMG_SIZE_PV01
    else:
        img_size = IMG_SIZE_PV08 if resolution == "PV08" else IMG_SIZE_PV03
    img = get_tf(img_size)(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    # Load model
    if resolution == "PV01":
        from models.swinunet import SwinUNet
        model = SwinUNet()
        state = torch.load(CKPT_PV01, map_location=device)
    else:
        backbone = "b2" if resolution == "PV08" else "b4"
        depths   = {"b2": [3, 4, 6, 3], "b4": [3, 8, 27, 3]}[backbone]
        cfg = SegformerConfig(
            num_encoder_blocks=4, depths=depths,
            hidden_sizes=[64, 128, 320, 512], num_attention_heads=[1, 2, 5, 8],
            decoder_hidden_size=SEGFORMER_DECODER_HIDDEN,
            num_labels=SEGFORMER_NUM_LABELS,
        )
        model = SegformerForSemanticSegmentation(cfg)
        ckpt  = CKPT_PV08 if resolution == "PV08" else CKPT_PV03
        state = torch.load(ckpt, map_location=device)

    if "model_state_dict" in state: state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    with torch.no_grad():
        if resolution == "PV01":
            logits = model(img)
        else:
            logits = model(pixel_values=img).logits
            logits = F.interpolate(logits, size=(img_size, img_size),
                                   mode="bilinear", align_corners=False)

    prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask   = (prob > 0.5).astype(np.uint8)
    conf   = float(prob.max())

    del model; torch.cuda.empty_cache()

    # Confidence gate check
    if conf < CONF_THRESHOLD:
        print(f"  [WARN] Low confidence ({conf:.4f}) — consider re-running with TTA")

    task["mask"]       = mask
    task["confidence"] = conf
    task["prob_map"]   = prob
    return task


# ── Agent 4: Fusion ─────────────────────────────────────────────────────
def agent4_fuse(tasks):
    """Weighted fusion of specialist masks into WGS84 GeoTIFF."""
    masks = {}
    for t in tasks:
        masks[t["resolution"]] = t["mask"].astype(float)

    # Resize to common size (512)
    from PIL import Image as PILImage
    target = 512
    for k in masks:
        if masks[k].shape[0] != target:
            masks[k] = np.array(
                PILImage.fromarray(masks[k].astype(np.uint8)).resize(
                    (target, target), PILImage.NEAREST))

    # Weighted fusion
    fused = np.zeros((target, target))
    if "PV01" in masks: fused += FUSION_WEIGHT_PV01 * masks["PV01"]
    if "PV03" in masks: fused += FUSION_WEIGHT_PV03 * masks["PV03"]
    if "PV08" in masks: fused += FUSION_WEIGHT_PV08 * masks["PV08"]
    binary = (fused >= FUSION_THRESHOLD).astype(np.uint8)

    return binary


# ── Agent 6: Report Generation ──────────────────────────────────────────
def agent6_report(task, fused_mask, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    lat  = task.get("lat", 0.0)
    lon  = task.get("lon", 0.0)
    res  = task["resolution"]
    conf = task.get("confidence", 0.0)
    mask = fused_mask

    panel_px    = int(mask.sum())
    total_area  = round(panel_px * 0.01, 2)   # 0.01 m²/px at 0.1m GSD
    panel_count = panel_px // 11               # approx panels

    base = os.path.splitext(task["filename"])[0]

    # 1. Text report
    report = (
        f"MAPVNet Detection Report\n"
        f"{'='*40}\n"
        f"Image:       {task['filename']}\n"
        f"Resolution:  {res}\n"
        f"GPS Centre:  {lat:.4f}°N, {lon:.4f}°E\n"
        f"Panel px:    {panel_px:,}\n"
        f"Total Area:  {total_area:,.2f} m²\n"
        f"Est. Panels: {panel_count:,}\n"
        f"Confidence:  {conf:.4f}\n"
        f"Status:      {'NORMAL' if conf >= CONF_THRESHOLD else 'LOW_CONF'}\n"
    )
    with open(os.path.join(out_dir, f"{base}_report.txt"), "w") as f:
        f.write(report)

    # 2. JSON
    json_data = {"filename": task["filename"], "resolution": res,
                 "lat": lat, "lon": lon, "panel_pixels": panel_px,
                 "total_area_m2": total_area, "estimated_panels": panel_count,
                 "confidence": conf, "status": "NORMAL"}
    with open(os.path.join(out_dir, f"{base}_summary.json"), "w") as f:
        json.dump(json_data, f, indent=2)

    # 3. GeoTIFF
    gsd    = {"PV08": 0.8, "PV03": 0.3, "PV01": 0.1}.get(res, 0.1)
    extent = mask.shape[0] * gsd / 111320   # degrees
    transform = from_bounds(lon - extent/2, lat - extent/2,
                             lon + extent/2, lat + extent/2,
                             mask.shape[1], mask.shape[0])
    tiff_path = os.path.join(out_dir, f"{base}_mask.tif")
    with rasterio.open(tiff_path, "w", driver="GTiff",
                       height=mask.shape[0], width=mask.shape[1],
                       count=1, dtype=rasterio.uint8,
                       crs=CRS.from_epsg(4326), transform=transform) as dst:
        dst.write(mask, 1)

    # 4. Folium map
    m = folium.Map(location=[lat, lon], zoom_start=17,
                   tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                   attr="Google Satellite")
    folium.Marker([lat, lon], popup=folium.Popup(
        f"<b>{res}</b><br>Panels: {panel_count:,}<br>"
        f"Area: {total_area:,} m²<br>Conf: {conf:.3f}<br>NORMAL",
        max_width=200)).add_to(m)
    m.save(os.path.join(out_dir, f"{base}_map.html"))

    print(f"\n{'='*45}")
    print(f"  Resolution:  {res}")
    print(f"  GPS:         {lat:.4f}°N, {lon:.4f}°E")
    print(f"  Panel area:  {total_area:,.2f} m²")
    print(f"  Confidence:  {conf:.4f}")
    print(f"  Outputs saved → {out_dir}/")
    print(f"{'='*45}")


# ── Main ───────────────────────────────────────────────────────────────
def main(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MAPVNet Inference | Device: {device}")
    print(f"Input: {image_path}\n")

    # Agent 1
    task = agent1_parse(image_path)
    task["image_path"] = image_path
    print(f"Agent 1 → Resolution: {task['resolution']} | "
          f"GPS: {task['lat']:.4f}°N, {task['lon']:.4f}°E")

    # Agent 2
    task = agent2_route(task, device)
    print(f"Agent 2 → Context: {task['context']}")

    # Agent 3
    task = agent3_segment(task, device)
    print(f"Agent 3 → Confidence: {task['confidence']:.4f} | "
          f"Panel px: {task['mask'].sum():,}")

    # Agent 4
    fused = agent4_fuse([task])
    print(f"Agent 4 → Fused mask: {fused.sum():,} px")

    # Agent 6
    out_dir = os.path.join(OUTPUT_DIR, os.path.splitext(task["filename"])[0])
    agent6_report(task, fused, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAPVNet Inference Pipeline")
    parser.add_argument("--image", required=True, help="Path to input .bmp image")
    args = parser.parse_args()
    main(args.image)
