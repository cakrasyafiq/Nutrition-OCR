"""
Nutrition Table Detector
========================
YOLO-based detection module that locates nutrition fact tables
("Informasi Nilai Gizi") in raw photos before OCR processing.

Uses the Open Food Facts pretrained model from HuggingFace:
    openfoodfacts/nutrition-table-yolo

The model is automatically downloaded on first use and cached
in the ``models/`` directory next to this file.
"""

import cv2
import logging
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

logger = logging.getLogger("nutrition_detector")

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

MODEL_REPO = "openfoodfacts/nutrition-table-yolo"
MODEL_FILENAME = "weights/best.pt"          # path inside the HF repo
BASE_DIR = Path(__file__).resolve().parents[1]
LOCAL_MODEL_DIR = BASE_DIR / "models"
LOCAL_WEIGHT_PATH = LOCAL_MODEL_DIR / "weights" / "best.pt"
LEGACY_WEIGHT_PATH = LOCAL_MODEL_DIR / "best.pt"

_yolo_model: YOLO | None = None


def get_model() -> YOLO:
    """Download (if needed) and lazily load the YOLO detection model."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    # The HF hub stores the file under weights/best.pt by default.
    if LOCAL_WEIGHT_PATH.exists():
        target_path = LOCAL_WEIGHT_PATH
    elif LEGACY_WEIGHT_PATH.exists():
        target_path = LEGACY_WEIGHT_PATH
    else:
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading nutrition-table detection model from %s …", MODEL_REPO)

        downloaded_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            local_dir=str(LOCAL_MODEL_DIR),
        )
        target_path = Path(downloaded_path)
        logger.info("Model cached at %s", target_path)

    _yolo_model = YOLO(str(target_path))
    return _yolo_model


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_nutrition_table(
    image_path: Path,
    model: YOLO | None = None,
    confidence: float = 0.3,
) -> list[dict]:
    """Run YOLO inference and return detections sorted by confidence (desc).

    Each detection dict contains:
        - bbox: (x1, y1, x2, y2) in pixel coordinates
        - confidence: float
        - class_id: int
    """
    if model is None:
        model = get_model()

    results = model(str(image_path), conf=confidence, verbose=False)

    detections: list[dict] = []
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


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def crop_detection(
    image_path: Path,
    bbox: tuple[int, int, int, int],
    padding_ratio: float = 0.05,
) -> np.ndarray:
    """Crop the detected region from *image_path* with padding.

    Parameters
    ----------
    image_path : Path
        Source image file.
    bbox : tuple
        (x1, y1, x2, y2) bounding box in pixel coordinates.
    padding_ratio : float
        Extra padding as a fraction of box dimensions (0.05 = 5 %).

    Returns
    -------
    np.ndarray
        Cropped image (BGR).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return img[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Combined detect-and-crop convenience function
# ---------------------------------------------------------------------------

def detect_and_crop(
    image_path: Path,
    output_dir: Path,
    model: YOLO | None = None,
    confidence: float = 0.3,
    padding_ratio: float = 0.05,
) -> tuple[Path, dict | None]:
    """Detect the nutrition table, crop it, and return the cropped image path.

    Returns
    -------
    tuple[Path, dict | None]
        (cropped_image_path, detection_info)
        If no table is found, *cropped_image_path* is the original path and
        *detection_info* is ``None``.
    """
    detections = detect_nutrition_table(image_path, model, confidence)

    if not detections:
        logger.warning(
            "No nutrition table detected in %s — falling back to full image",
            image_path.name,
        )
        return image_path, None

    best = detections[0]
    logger.info(
        "Detected nutrition table in %s (conf=%.2f, bbox=%s)",
        image_path.name,
        best["confidence"],
        best["bbox"],
    )

    cropped = crop_detection(image_path, best["bbox"], padding_ratio)

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = output_dir / f"{image_path.stem}_detected_crop.png"
    cv2.imwrite(str(crop_path), cropped)

    return crop_path, best
