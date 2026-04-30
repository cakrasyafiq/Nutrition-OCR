from paddleocr import PaddleOCR
from pathlib import Path
import json
import csv
import re
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# OCR setup
# ---------------------------------------------------------------------------
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def preprocess_for_ocr(image_path: Path, output_dir: Path) -> Path:
    """Preprocess input image to improve OCR detection/recognition quality."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Upscale small images so thin characters become easier to detect.
    h, w = image.shape[:2]
    scale = 1.8 if max(h, w) < 1400 else 1.25
    resized = cv2.resize(
        image,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Preserve edges while reducing camera/compression noise.
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Boost local contrast to make faded print more legible.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    # Adaptive threshold works better than global threshold for uneven lighting.
    binarized = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    # Light morphology to connect broken character strokes.
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel, iterations=1)

    preprocessed_path = output_dir / f"{image_path.stem}_preprocessed.png"
    cv2.imwrite(str(preprocessed_path), processed)
    return preprocessed_path

BASE_DIR = Path(__file__).resolve().parents[1]
input_image = BASE_DIR / "test_files" / "test_gizi_3.png"
output_dir = BASE_DIR / "output"
output_dir.mkdir(parents=True, exist_ok=True)
preprocessed_image = preprocess_for_ocr(input_image, output_dir)


def run_ocr_extract_nutrition(source_image: Path, output_dir: Path):
    """Run OCR for one source image and return extracted nutrition + diagnostics."""
    output = list(ocr.predict(str(source_image)))

    all_nutrition = []
    all_scores = []

    for res in output:
        res.save_to_json(str(output_dir))

        rec_texts = []
        rec_scores = []
        rec_polys = []

        result_data = getattr(res, "res", None)
        if isinstance(result_data, dict):
            rec_texts = result_data.get("rec_texts", [])
            rec_scores = result_data.get("rec_scores", [])
            rec_polys = result_data.get("rec_polys", []) or result_data.get("dt_polys", [])

        # Fallback: read generated JSON for this specific source image.
        if not rec_texts:
            json_path = output_dir / f"{source_image.stem}_res.json"
            if json_path.exists():
                with json_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                rec_texts = payload.get("rec_texts", [])
                rec_scores = payload.get("rec_scores", [])
                rec_polys = payload.get("rec_polys", []) or payload.get("dt_polys", [])

        if not rec_scores:
            rec_scores = [1.0] * len(rec_texts)
        if not rec_polys:
            rec_polys = [
                [[0, i * 100], [100, i * 100], [100, i * 100 + 50], [0, i * 100 + 50]]
                for i in range(len(rec_texts))
            ]

        rows = group_into_rows(rec_polys, rec_texts, rec_scores)
        nutrition = extract_nutrition_wide(rows)
        all_nutrition.append(nutrition)
        all_scores.extend(rec_scores)

    # Keep best page for single-image usage by maximizing non-empty extracted values.
    best_nutrition = {}
    best_non_empty = -1
    for nutrition in all_nutrition:
        non_empty = sum(1 for v in nutrition.values() if str(v).strip())
        if non_empty > best_non_empty:
            best_non_empty = non_empty
            best_nutrition = nutrition

    avg_score = (sum(all_scores) / len(all_scores)) if all_scores else 0.0
    return {
        "nutrition": best_nutrition,
        "avg_score": avg_score,
        "non_empty": best_non_empty,
    }


def candidate_rank(candidate):
    """Rank candidate by extraction completeness first, confidence second."""
    nutrition = candidate["nutrition"]
    return (
        candidate["non_empty"],
        len(nutrition),
        candidate["avg_score"],
    )


# ---------------------------------------------------------------------------
# Helpers - spatial grouping
# ---------------------------------------------------------------------------

def _bbox_center_y(poly):
    """Return the vertical centre of a 4-point polygon."""
    ys = [p[1] for p in poly]
    return sum(ys) / len(ys)


def _bbox_left_x(poly):
    """Return the leftmost x coordinate of a 4-point polygon."""
    return min(p[0] for p in poly)


def _bbox_height(poly):
    """Return the approximate height of a bounding box."""
    ys = [p[1] for p in poly]
    return max(ys) - min(ys)


def group_into_rows(polys, texts, scores, y_threshold_ratio=0.45):
    """Group detected text fragments into rows based on Y-coordinate proximity.

    Returns a list of rows, where each row is a list of
    dicts with keys: center_y, left_x, height, text, score.
    Rows are sorted top-to-bottom, entries within rows left-to-right.
    """
    if not texts:
        return []

    entries = []
    for poly, text, score in zip(polys, texts, scores):
        if not text.strip():
            continue
        entries.append({
            "center_y": _bbox_center_y(poly),
            "left_x": _bbox_left_x(poly),
            "height": _bbox_height(poly),
            "text": text.strip(),
            "score": score,
        })

    if not entries:
        return []

    entries.sort(key=lambda e: (e["center_y"], e["left_x"]))

    avg_height = sum(e["height"] for e in entries) / len(entries)
    y_threshold = avg_height * y_threshold_ratio

    rows = []
    current_row = [entries[0]]
    for entry in entries[1:]:
        if abs(entry["center_y"] - current_row[0]["center_y"]) <= y_threshold:
            current_row.append(entry)
        else:
            current_row.sort(key=lambda e: e["left_x"])
            rows.append(current_row)
            current_row = [entry]
    current_row.sort(key=lambda e: e["left_x"])
    rows.append(current_row)

    return rows


# ---------------------------------------------------------------------------
# Helpers - nutrition extraction (wide CSV format)
# ---------------------------------------------------------------------------

# Standardized nutrient column names and their keyword patterns for matching.
# Order matters: first match wins. Patterns are checked against the merged
# row text (lowercased).
NUTRIENT_PATTERNS = [
    ("Takaran Saji",          [r"takaran\s*saji"]),
    ("Sajian per Kemasan",    [r"sajian\s*per\s*kemasan", r"per\s*kemasan"]),
    ("Energi Total",          [r"energi\s*total"]),
    ("Energi dari Lemak Jenuh", [r"energi\s*dari\s*lemak\s*jenuh"]),
    ("Energi dari Lemak",     [r"energi\s*dari\s*lemak"]),
    ("Lemak Trans",           [r"lemak\s*trans"]),
    ("Lemak Tidak Jenuh Tunggal", [r"lemak\s*tidak\s*jenuh\s*tunggal"]),
    ("Lemak Tidak Jenuh Ganda",   [r"lemak\s*tidak\s*jenuh\s*ganda"]),
    ("Lemak Jenuh",           [r"lemak\s*jenuh"]),
    ("Lemak Total",           [r"lemak\s*total"]),
    ("Kolesterol",            [r"kolesterol"]),
    ("Protein",               [r"protein"]),
    ("Karbohidrat Total",     [r"karbohidrat\s*total"]),
    ("Serat Pangan",          [r"serat\s*pangan", r"serat"]),
    ("Gula Total",            [r"gula\s*total"]),
    ("Gula",                  [r"gula"]),
    ("Sukrosa",               [r"sukrosa"]),
    ("Natrium",               [r"natrium", r"garam.*natrium", r"garam"]),
    ("Vitamin D",             [r"vitamin\s*d"]),
    ("Vitamin E",             [r"vitamin\s*e"]),
    ("Vitamin B1",            [r"vitamin\s*b1"]),
    ("Vitamin B2",            [r"vitamin\s*b2"]),
    ("Vitamin B3",            [r"vitamin\s*b3"]),
    ("Vitamin B6",            [r"vitamin\s*b6"]),
    ("Vitamin B12",           [r"vitamin\s*b12"]),
    ("Vitamin C",             [r"vitamin\s*c"]),
    ("Vitamin A",             [r"vitamin\s*a"]),
    ("Kalium",                [r"kalium"]),
    ("Kalsium",               [r"kalsium"]),
    ("Fosfor",                [r"fosfor"]),
    ("Magnesium",             [r"magnesium"]),
    ("Zat Besi",              [r"zat\s*besi"]),
    ("Zink",                  [r"zink"]),
    ("Selenium",              [r"selenium"]),
]

# Rows whose merged text matches any of these patterns are skipped entirely
# (headers, footers, disclaimers).
SKIP_PATTERNS = [
    r"informasi\s*nilai\s*gizi",
    r"jumlah\s*per\s*sajian",
    r"persen\s*akg",
    r"\*persen",
    r"kebutuhan\s*energi",
    r"berdasarkan",
    r"^%\s*akg",
]


def _extract_numeric(text):
    """Extract the first numeric value from text (int or float).

    Examples:
        '130kkal' -> '130'
        '2.5g'   -> '2.5'
        '0mg'    -> '0'
        '15%'    -> '15'
        '200ml'  -> '200'
    """
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    return m.group(1) if m else ""


def _extract_serving_value(text):
    """For Takaran Saji, keep the value+unit together (e.g. '200ml', '25g')."""
    m = re.search(r"(\d+(?:\.\d+)?\s*(?:ml|g|oz|L))", text, re.I)
    return m.group(1).replace(" ", "") if m else _extract_numeric(text)


def _extract_serving_count(text):
    """For Sajian per Kemasan, extract just the count."""
    m = re.search(r"(\d+)", text)
    return m.group(1) if m else ""


def _extract_numeric_from_token(text):
    """Extract numeric value from one OCR token, normalizing decimal commas."""
    cleaned = text.replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    return m.group(1) if m else ""


def _looks_like_percent_token(text):
    return bool(re.match(r"^\s*\d+(?:[\.,]\d+)?\s*%\s*$", text))


def _is_numeric_like_token(text):
    """Heuristic for value tokens such as 7g, 180kkal, 95mg, 0%."""
    return bool(
        re.match(
            r"^\s*[:\-]?\s*\d+(?:[\.,]\d+)?\s*"
            r"(?:kkal|kcal|kal|g|mg|mcg|ug|ml|l|%)?\s*$",
            text,
            re.I,
        )
    )


def _find_key_boundary_index(row, patterns):
    """Find the last token index that belongs to the nutrient key phrase."""
    last_match_idx = -1
    prefix_parts = []
    for idx, entry in enumerate(row):
        prefix_parts.append(entry["text"])
        prefix_text = " ".join(prefix_parts).lower()
        if any(re.search(pat, prefix_text) for pat in patterns):
            last_match_idx = idx
    return last_match_idx


def _extract_aligned_value(row, col_name, patterns):
    """Extract value from tokens to the right of the matched nutrient key."""
    key_idx = _find_key_boundary_index(row, patterns)

    if key_idx >= 0:
        candidate_entries = row[key_idx + 1:]
    else:
        candidate_entries = row

    candidate_texts = [e["text"].strip() for e in candidate_entries if e["text"].strip()]
    merged_row = " ".join(e["text"] for e in row)

    if col_name == "Takaran Saji":
        candidate_merged = " ".join(candidate_texts)
        return _extract_serving_value(candidate_merged) or _extract_serving_value(merged_row)

    if col_name == "Sajian per Kemasan":
        candidate_merged = " ".join(candidate_texts)
        return _extract_serving_count(candidate_merged) or _extract_serving_count(merged_row)

    # Prefer numeric-like tokens on the right side of the key.
    for token in candidate_texts:
        if _looks_like_percent_token(token):
            continue
        if _is_numeric_like_token(token):
            value = _extract_numeric_from_token(token)
            if value:
                return value

    # Fallback: any non-percent token containing digits on the right side.
    for token in candidate_texts:
        if _looks_like_percent_token(token):
            continue
        value = _extract_numeric_from_token(token)
        if value:
            return value

    # Last fallback: extract from full row text.
    return _extract_numeric(merged_row)


def extract_nutrition_wide(rows):
    """Parse grouped OCR rows into a flat dict of nutrient -> value.

    Returns an ordered dict matching the NUTRIENT_PATTERNS order,
    only including nutrients that were actually found.
    """
    result = {}
    used_rows = set()

    for row_idx, row in enumerate(rows):
        merged = " ".join(e["text"] for e in row)
        merged_lower = merged.lower()

        # Skip header/footer rows
        if any(re.search(p, merged_lower) for p in SKIP_PATTERNS):
            # But check if this row *also* contains serving info
            if not re.search(r"takaran|sajian", merged_lower):
                used_rows.add(row_idx)
                continue

        # Try to match against known nutrient patterns
        for col_name, patterns in NUTRIENT_PATTERNS:
            if col_name in result:
                continue  # already found this nutrient
            for pat in patterns:
                if re.search(pat, merged_lower):
                    # Value is picked from the same row, to the right of the key.
                    val = _extract_aligned_value(row, col_name, patterns)

                    result[col_name] = val
                    used_rows.add(row_idx)
                    break

    # Second pass: try to pick up Natrium value that might be on a
    # separate row from "Garam" (e.g. "(natrium) 100mg" on its own row)
    if "Natrium" in result and not result["Natrium"]:
        for row_idx, row in enumerate(rows):
            if row_idx in used_rows:
                continue
            merged = " ".join(e["text"] for e in row)
            if re.search(r"natrium", merged, re.I):
                val = _extract_numeric(merged)
                if val:
                    result["Natrium"] = val
                    break

    return result


# ---------------------------------------------------------------------------
# Run OCR candidates and choose best extraction
# ---------------------------------------------------------------------------
original_candidate = run_ocr_extract_nutrition(input_image, output_dir)
preprocessed_candidate = run_ocr_extract_nutrition(preprocessed_image, output_dir)

best_candidate = max(
    [original_candidate, preprocessed_candidate],
    key=candidate_rank,
)

nutrition = best_candidate["nutrition"]

# Keep result artifacts from the chosen source for easy visual inspection.
chosen_source = input_image if best_candidate is original_candidate else preprocessed_image
chosen_output = list(ocr.predict(str(chosen_source)))
for res in chosen_output:
    res.save_to_img(str(output_dir))


# ---------------------------------------------------------------------------
# Write output from best candidate
# ---------------------------------------------------------------------------
csv_path = output_dir / f"{input_image.stem}.csv"

headers = list(nutrition.keys())
values = list(nutrition.values())

with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerow(values)

print(f"\n[OK] Output saved: {csv_path}")
print(f"\n{'=' * 70}")
print(f"  SOURCE USED: {chosen_source.name}")
print(f"  PREVIEW - {input_image.stem}")
print(f"{'=' * 70}")
for h, v in zip(headers, values):
    print(f"  {h:<30s}  {v}")
print(f"{'=' * 70}")