"""
Nutrition OCR API
=================
FastAPI application exposing nutrition extraction via REST endpoints.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import tempfile
import shutil
import time
import logging

from tools.ocr_engine import extract_nutrition_from_image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nutrition_ocr_api")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Nutrition OCR API",
    description=(
        "Upload an image of a nutrition facts table (Indonesian format) "
        "and receive structured JSON with extracted nutrient values."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class DetectionInfo(BaseModel):
    """Info about the nutrition-table detection stage."""
    detected: bool
    bbox: list[int] | None = None
    detection_confidence: float | None = None


class NutritionResult(BaseModel):
    """Structured extraction result."""
    nutrition: dict[str, str]
    confidence: float
    fields_extracted: int
    source_used: str
    processing_time_ms: float
    detection: DetectionInfo


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/extract", response_model=NutritionResult, tags=["OCR"])
async def extract_nutrition(
    image: UploadFile = File(..., description="Image of a nutrition facts table"),
):
    """Extract nutrition data from an uploaded image.

    Accepts common image formats (PNG, JPEG, BMP, TIFF, WebP).
    Returns structured JSON with nutrient names as keys and their
    extracted values.

    Example response:
    ```json
    {
        "nutrition": {
            "Takaran Saji": "200ml",
            "Energi Total": "180",
            "Protein": "6",
            "Lemak Total": "7"
        },
        "confidence": 0.9521,
        "fields_extracted": 14,
        "source_used": "preprocessed",
        "processing_time_ms": 1234.56
    }
    ```
    """
    # Validate file type
    allowed_types = {
        "image/png", "image/jpeg", "image/jpg", "image/bmp",
        "image/tiff", "image/webp",
    }
    if image.content_type and image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {image.content_type}. "
                   f"Accepted: {', '.join(sorted(allowed_types))}",
        )

    # Save uploaded file to a temp location
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    tmp_dir = Path(tempfile.mkdtemp(prefix="nutrition_upload_"))
    tmp_path = tmp_dir / f"upload{suffix}"

    try:
        with tmp_path.open("wb") as f:
            content = await image.read()
            f.write(content)

        logger.info(f"Processing image: {image.filename} ({len(content)} bytes)")

        start = time.perf_counter()
        result = extract_nutrition_from_image(tmp_path)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return NutritionResult(
            nutrition=result["nutrition"],
            confidence=result["confidence"],
            fields_extracted=result["fields_extracted"],
            source_used=result["source_used"],
            processing_time_ms=round(elapsed_ms, 2),
            detection=DetectionInfo(**result.get("detection", {"detected": False})),
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("OCR extraction failed")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/extract/batch", tags=["OCR"])
async def extract_nutrition_batch(
    images: list[UploadFile] = File(
        ..., description="Multiple images of nutrition facts tables"
    ),
):
    """Extract nutrition data from multiple images in one request.

    Returns a list of results, one per image. If an individual image
    fails, its entry will contain an `error` field instead of nutrition data.
    """
    results = []

    for image in images:
        suffix = Path(image.filename or "upload.png").suffix or ".png"
        tmp_dir = Path(tempfile.mkdtemp(prefix="nutrition_upload_"))
        tmp_path = tmp_dir / f"upload{suffix}"

        try:
            with tmp_path.open("wb") as f:
                content = await image.read()
                f.write(content)

            start = time.perf_counter()
            result = extract_nutrition_from_image(tmp_path)
            elapsed_ms = (time.perf_counter() - start) * 1000

            results.append({
                "filename": image.filename,
                "nutrition": result["nutrition"],
                "confidence": result["confidence"],
                "fields_extracted": result["fields_extracted"],
                "source_used": result["source_used"],
                "processing_time_ms": round(elapsed_ms, 2),
                "detection": result.get("detection", {"detected": False}),
            })

        except Exception as e:
            logger.exception(f"Failed processing {image.filename}")
            results.append({
                "filename": image.filename,
                "error": str(e),
            })
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"results": results, "total": len(results)}
