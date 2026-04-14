from paddleocr import PaddleOCR
from pathlib import Path
import json
import csv
import re

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

input_image = Path("test_files/test_gizi_5.jpeg")
output = ocr.predict(str(input_image))
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)


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
                    # Extract value based on column type
                    if col_name == "Takaran Saji":
                        val = _extract_serving_value(merged)
                    elif col_name == "Sajian per Kemasan":
                        val = _extract_serving_count(merged)
                    else:
                        # Collect all numeric-like tokens from the row
                        # Prefer value+unit tokens (e.g. '7g', '0mg') over bare %
                        all_texts = [e["text"] for e in row]
                        val = ""
                        for t in all_texts:
                            # Skip percentage tokens
                            if re.match(r"^\d+(\.\d+)?%$", t):
                                continue
                            # Skip tokens that are purely text (nutrient name parts)
                            num = _extract_numeric(t)
                            if num and re.search(r"\d", t):
                                # Check it's a value token not a name token
                                # (name tokens like "B12" contain digits too)
                                if re.match(r"^\d+(?:\.\d+)?\s*[a-zA-Z]*$", t):
                                    val = num
                                    break

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
# Process results
# ---------------------------------------------------------------------------
for index, res in enumerate(output):
    res.save_to_img(str(output_dir))
    res.save_to_json(str(output_dir))

    input_path = Path(str(getattr(res, "input_path", input_image)))
    base_name = input_path.stem if input_path.stem else "ocr_result"

    # --- Load OCR data from the result or fallback to saved JSON -----------
    rec_texts = []
    rec_scores = []
    rec_polys = []

    result_data = getattr(res, "res", None)
    if isinstance(result_data, dict):
        rec_texts = result_data.get("rec_texts", [])
        rec_scores = result_data.get("rec_scores", [])
        rec_polys = result_data.get("rec_polys", []) or result_data.get("dt_polys", [])

    # Fallback: read the generated json
    if not rec_texts:
        json_path = output_dir / f"{base_name}_res.json"
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

    # --- Group into rows and extract nutrition data -----------------------
    rows = group_into_rows(rec_polys, rec_texts, rec_scores)
    nutrition = extract_nutrition_wide(rows)

    # --- Write wide CSV ---------------------------------------------------
    suffix = "" if len(output) == 1 else f"_page_{index + 1}"
    csv_path = output_dir / f"{base_name}{suffix}.csv"

    headers = list(nutrition.keys())
    values = list(nutrition.values())

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(values)

    # --- Print preview ----------------------------------------------------
    print(f"\n[OK] Output saved: {csv_path}")
    print(f"\n{'=' * 70}")
    print(f"  PREVIEW - {base_name}")
    print(f"{'=' * 70}")
    # Print as a readable key-value list
    for h, v in zip(headers, values):
        print(f"  {h:<30s}  {v}")
    print(f"{'=' * 70}")