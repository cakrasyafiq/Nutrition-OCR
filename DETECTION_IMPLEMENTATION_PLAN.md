# Nutrition Table Detection — Implementation Plan

> **Goal:** Add a detection stage before OCR that automatically locates and crops the  
> "Informasi Nilai Gizi" table from raw phone camera photos, then feeds the clean  
> crop into the existing PaddleOCR extraction pipeline.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architecture Overview](#2-architecture-overview)
3. [Approach Comparison](#3-approach-comparison)
4. [Recommended Path](#4-recommended-path)
5. [Approach A — Pretrained YOLO (Start Here)](#5-approach-a--pretrained-yolo-start-here)
6. [Approach B — PaddleOCR PPStructure (Alternative)](#6-approach-b--paddleocr-ppstructure-alternative)
7. [Approach C — Custom-Trained YOLOv8 (If Needed)](#7-approach-c--custom-trained-yolov8-if-needed)
8. [Mobile Deployment Strategy](#8-mobile-deployment-strategy)
9. [File Changes Summary](#9-file-changes-summary)
10. [Verification Checklist](#10-verification-checklist)

---

## 1. Problem Statement

The current OCR pipeline (`app_text.py`) assumes input images are **already cropped**
to the nutrition table. When photos come directly from a phone camera, the image
contains full product packaging, background clutter, hands, etc. — OCR picks up
irrelevant text and accuracy drops significantly.

**Solution:** Insert a detection step that locates the nutrition table bounding box,
crops the image (with padding), and passes only the cropped region to OCR.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FULL PIPELINE                                │
│                                                                     │
│  📷 Phone Camera Photo                                              │
│       │                                                             │
│       ▼                                                             │
│  🔍 YOLO Detector (nutrition-table-yolo, ~6 MB)                     │
│       │                                                             │
│       ├── Table found (conf > 0.3) ──▶ ✂️ Crop + 5% padding         │
│       │                                                             │
│       └── No table found ──▶ ⚠️ Fallback: use full image            │
│                                        │                            │
│                                        ▼                            │
│                              🔧 Preprocess for OCR                  │
│                              (upscale, denoise, binarize)           │
│                                        │                            │
│                                        ▼                            │
│                              📝 PaddleOCR Text Extraction           │
│                              (PP-OCRv5 det + rec)                   │
│                                        │                            │
│                                        ▼                            │
│                              📊 Structured CSV Output               │
└─────────────────────────────────────────────────────────────────────┘
```

### Mobile-Optimized Hybrid Architecture

For mobile app deployment, split the pipeline across device and server:

```
┌──────── ON DEVICE ─────────┐     ┌────────── ON SERVER ──────────┐
│                             │     │                               │
│  📷 Camera Preview          │     │  📥 Receive cropped image     │
│       │                    │     │       │                       │
│       ▼                    │     │       ▼                       │
│  🔍 YOLOv8n (TFLite/CoreML)│     │  🔧 Preprocess (OpenCV)      │
│  ~6 MB, ~30-50ms on CPU    │     │       │                       │
│       │                    │     │       ▼                       │
│       ▼                    │     │  📝 PaddleOCR (full models)   │
│  ✂️ Crop + bounding box UI  │     │       │                       │
│       │                    │     │       ▼                       │
│       ▼                    │     │  📊 Structured JSON/CSV       │
│  📤 Upload cropped image ──┼────▶│                               │
│     (5-10x smaller)        │     │                               │
└─────────────────────────────┘     └───────────────────────────────┘
```

**Benefits of the hybrid approach:**
- Instant visual feedback on-device (bounding box overlay on camera)
- Smaller upload payload (cropped image vs full photo)
- Best OCR accuracy (full PaddleOCR models on server)
- Privacy: only the nutrition table leaves the device

---

## 3. Approach Comparison

| Approach | Effort | Accuracy | Model Size | Mobile-Ready | Best For |
|----------|--------|----------|------------|-------------|----------|
| **A. Pretrained YOLO** | ⭐ Low | Good | ~6 MB | ✅ Yes | Fast MVP, test viability |
| **B. PPStructure** | ⭐⭐ Low | Moderate | ~40-80 MB | ⚠️ Difficult | Staying 100% PaddleOCR |
| **C. Custom YOLO** | ⭐⭐⭐ High | Best | ~6 MB | ✅ Yes | Production with Indonesian labels |

---

## 4. Recommended Path

```
Step 1: Download pretrained model                          → 2 minutes
Step 2: Test on existing 5 test images                     → 5 minutes
Step 3: Test on 10 new raw phone photos                    → 10 minutes
              │
              ├── Works well (>90% detection rate)? → Ship it ✅
              │
              └── Doesn't work? → Go to Approach C:
                                   - Annotate 50-100 Indonesian label images
                                   - Fine-tune from pretrained weights (30 min on Colab T4)
                                   - Ship it ✅
```

**Rule of thumb:** Don't train custom until you've proven the pretrained model fails.

---

## 5. Approach A — Pretrained YOLO (Start Here)

### 5.1 Dependencies

Add to `requirements.txt`:
```
ultralytics
huggingface_hub
```

### 5.2 New File: `detector.py`

```python
"""
Nutrition table detection using YOLOv8n pretrained model.

Downloads the Open Food Facts 'nutrition-table-yolo' model from HuggingFace
on first run, caches it locally in models/ directory.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

MODEL_REPO = "openfoodfacts/nutrition-table-yolo"
MODEL_FILENAME = "best.pt"
LOCAL_MODEL_DIR = Path(__file__).parent / "models"


def get_model() -> YOLO:
    """Download (if needed) and load the YOLO detection model."""
    local_path = LOCAL_MODEL_DIR / MODEL_FILENAME

    if not local_path.exists():
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            local_dir=str(LOCAL_MODEL_DIR),
        )
        local_path = Path(downloaded)
        print(f"[detector] Model downloaded to {local_path}")

    return YOLO(str(local_path))


# ---------------------------------------------------------------------------
# Detection & cropping
# ---------------------------------------------------------------------------

def detect_nutrition_table(
    image_path: Path,
    model: YOLO | None = None,
    confidence: float = 0.3,
) -> list[dict]:
    """Run detection and return list of detections sorted by confidence (desc).

    Each detection dict contains:
        - bbox: (x1, y1, x2, y2) in pixel coordinates
        - confidence: float
        - class_id: int
    """
    if model is None:
        model = get_model()

    results = model(str(image_path), conf=confidence, verbose=False)

    detections = []
    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": conf,
                "class_id": cls,
            })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def crop_detection(
    image_path: Path,
    bbox: tuple[int, int, int, int],
    padding_ratio: float = 0.05,
) -> np.ndarray:
    """Crop the detected region from the image with padding.

    Args:
        image_path: Path to the source image.
        bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.
        padding_ratio: Extra padding as a fraction of box dimensions (0.05 = 5%).

    Returns:
        Cropped image as a numpy array (BGR).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    # Add padding
    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return img[y1:y2, x1:x2]


def detect_and_crop(
    image_path: Path,
    output_dir: Path,
    model: YOLO | None = None,
    confidence: float = 0.3,
    padding_ratio: float = 0.05,
) -> Path:
    """Detect nutrition table, crop, save, and return path to cropped image.

    If no table is detected, returns the original image path as fallback.
    """
    detections = detect_nutrition_table(image_path, model, confidence)

    if not detections:
        print(f"[detector] No nutrition table detected in {image_path.name}, using full image")
        return image_path

    best = detections[0]
    print(f"[detector] Found nutrition table in {image_path.name} "
          f"(confidence: {best['confidence']:.2f}, bbox: {best['bbox']})")

    cropped = crop_detection(image_path, best["bbox"], padding_ratio)

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = output_dir / f"{image_path.stem}_detected_crop.png"
    cv2.imwrite(str(crop_path), cropped)

    return crop_path
```

### 5.3 Integration into `app_text.py`

Add these lines near the top of the script, before OCR runs:

```python
from detector import detect_and_crop

input_image = Path("test_files/test_gizi_5.jpeg")  # raw phone photo
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# NEW: Detect and crop nutrition table first
cropped_image = detect_and_crop(input_image, output_dir)

# Existing pipeline now uses cropped image instead of raw photo
preprocessed_image = preprocess_for_ocr(cropped_image, output_dir)
```

The rest of the pipeline (`run_ocr_extract_nutrition`, `extract_nutrition_wide`, CSV
output) remains **completely unchanged**.

### 5.4 Mobile Export

To export the model for on-device inference:

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")

# Android (TensorFlow Lite)
model.export(format="tflite", imgsz=640)

# iOS (CoreML)
model.export(format="coreml", imgsz=640)

# Cross-platform (ONNX)
model.export(format="onnx", imgsz=640)

# Lightweight (NCNN — good for ARM devices)
model.export(format="ncnn", imgsz=640)
```

---

## 6. Approach B — PaddleOCR PPStructure (Alternative)

Use only if you want to avoid adding `ultralytics` as a dependency.

```python
from paddleocr import PPStructure
import cv2

# Layout detection only (no OCR, no table structure recognition)
engine = PPStructure(table=False, ocr=False, show_log=False)

img = cv2.imread("phone_photo.jpg")
results = engine(img)

for i, region in enumerate(results):
    if region['type'] == 'table':
        x1, y1, x2, y2 = map(int, region['bbox'])
        # Boundary checks
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(f"table_crop_{i}.jpg", crop)
```

**Limitations:**
- Trained on **documents** (papers, forms), not food packaging
- Larger model size (~40-80 MB) compared to YOLOv8n (~6 MB)
- Harder to export for mobile deployment
- May struggle with colored backgrounds and curved packaging surfaces

---

## 7. Approach C — Custom-Trained YOLOv8 (If Needed)

Only pursue this if Approach A fails on Indonesian (BPOM-style) nutrition labels.

### 7.1 Data Collection

- Photograph **50-100 Indonesian products** with your phone
- Capture variety: different lighting, angles, packaging colors, distances
- Include edge cases: curved surfaces, reflective packaging, partial occlusion

### 7.2 Annotation

Use **Roboflow** (free tier: 10k images):
1. Create project → Object Detection
2. Upload images
3. Draw bounding boxes around the "Informasi Nilai Gizi" section
4. Single class: `nutrition_table`
5. Apply augmentations (rotation ±15°, brightness ±20%, blur)
6. Export as **YOLOv8 format** → downloads a ZIP with `data.yaml`

Alternative: **Label Studio** (self-hosted, fully free)

### 7.3 Training (Google Colab T4 — Free)

```python
# In a Colab notebook:
!pip install ultralytics

from ultralytics import YOLO

# Fine-tune FROM the Open Food Facts weights (transfer learning)
# This is much better than training from scratch
model = YOLO("path/to/openfoodfacts_best.pt")

results = model.train(
    data="path/to/data.yaml",
    epochs=50,          # 30-50 is usually enough for fine-tuning
    imgsz=640,
    batch=16,
    patience=10,        # early stopping
    name="nutrition_indo",
)

# Best weights saved to: runs/detect/nutrition_indo/weights/best.pt
```

Expected training time: **~15-30 minutes** on a Colab T4 GPU.

### 7.4 Expected Performance

With fine-tuning from pretrained weights:
- **50 images:** ~85-90% mAP
- **100 images:** ~92-95% mAP
- **200 images:** ~95-98% mAP

### 7.5 Swap Into Pipeline

Simply replace the model path in `detector.py`:

```python
LOCAL_MODEL_DIR = Path(__file__).parent / "models"
MODEL_FILENAME = "best.pt"  # replace with your custom-trained weights
```

---

## 8. Mobile Deployment Strategy

### On-Device Detection Performance (YOLOv8n)

| Device Tier | Inference Time | Format |
|-------------|---------------|--------|
| Flagship (Snapdragon 8 Gen 2+) | ~15-25 ms | TFLite/NCNN |
| Mid-range (Snapdragon 6/7 series) | ~30-50 ms | TFLite/NCNN |
| Budget (Helio G series) | ~80-120 ms | TFLite |
| iPhone 13+ | ~10-20 ms | CoreML |

All times are for a single 640×640 image on CPU. More than fast enough for
real-time camera preview overlay.

### Mobile Framework Options

| Platform | Framework | Detection Model Format |
|----------|-----------|----------------------|
| Android (Kotlin/Java) | TFLite Interpreter | `.tflite` |
| Android (Flutter) | `tflite_flutter` plugin | `.tflite` |
| iOS (Swift) | CoreML / Vision | `.mlmodel` |
| Cross-platform (React Native) | `react-native-fast-tflite` | `.tflite` |

### Server-Side OCR API

For the hybrid architecture, expose the OCR pipeline as a REST API:

```python
# Example with FastAPI (future development)
from fastapi import FastAPI, UploadFile
from app_text import run_ocr_extract_nutrition, preprocess_for_ocr

app = FastAPI()

@app.post("/extract")
async def extract_nutrition(file: UploadFile):
    # Save uploaded crop
    # Run preprocess + OCR
    # Return structured JSON
    ...
```

---

## 9. File Changes Summary

```
Nutrition_OCR/
├── detector.py                 [NEW]  — YOLO detection + cropping module
├── models/
│   └── best.pt                 [NEW]  — auto-downloaded on first run
├── app_text.py                 [MODIFY] — add 3 lines to call detector before OCR
├── requirements.txt            [MODIFY] — add ultralytics, huggingface_hub
├── DETECTION_IMPLEMENTATION_PLAN.md  [NEW] — this file
└── .gitignore                  [MODIFY] — add models/ directory
```

---

## 10. Verification Checklist

- [ ] **Install dependencies:** `pip install ultralytics huggingface_hub`
- [ ] **Test pretrained model** on `test_gizi_1.png` through `test_gizi_5.jpeg`
- [ ] **Verify cropped outputs** saved in `output/*_detected_crop.png`
- [ ] **Compare OCR results** before/after detection stage (should be equal or better)
- [ ] **Test with new raw phone photos** (not pre-cropped) — end-to-end flow
- [ ] **Evaluate if custom training needed** based on detection rate on Indonesian labels
- [ ] *(If needed)* Annotate 50-100 images on Roboflow
- [ ] *(If needed)* Fine-tune on Colab and swap weights
- [ ] *(For mobile)* Export model to TFLite/CoreML and test on target device

---

## Resources

- **Pretrained Model:** https://huggingface.co/openfoodfacts/nutrition-table-yolo
- **Training Dataset:** https://huggingface.co/datasets/openfoodfacts/nutrition-table-detection
- **Ultralytics Docs:** https://docs.ultralytics.com/
- **Roboflow (annotation):** https://roboflow.com/
- **YOLOv8 Export Guide:** https://docs.ultralytics.com/modes/export/
