"""
Stage 1 — Layout extraction.

extract_layout(image_path) -> list[dict]
  Each dict: {type, bbox: (x1,y1,x2,y2), confidence}

Also produces skeleton PNG in output/skeletons/.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SKELETONS_DIR = OUTPUT_DIR / "skeletons"
SKELETONS_DIR.mkdir(parents=True, exist_ok=True)

# Config flag: "auto" tries layoutparser, falls back to OpenCV; "opencv" forces OpenCV
LAYOUT_BACKEND = os.environ.get("LAYOUT_BACKEND", "auto")

# Try importing layoutparser (optional)
_lp_available = False
if LAYOUT_BACKEND != "opencv":
    try:
        import layoutparser as lp
        _lp_available = True
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# OpenCV-based layout extraction (always available)
# ---------------------------------------------------------------------------

def _opencv_extract(img_bgr: np.ndarray) -> list[dict]:
    """Extract layout elements using OpenCV morphological operations."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    elements = []

    # --- Table detection: morphological lines ---
    # Horizontal lines
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 20), 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    # Vertical lines
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 30, 20)))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)

    table_mask = cv2.add(horiz, vert)
    # Dilate to connect nearby lines into table regions
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    table_regions = cv2.dilate(table_mask, dilate_k)
    table_contours, _ = cv2.findContours(table_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in table_contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < 0.005 * w * h:  # skip tiny fragments
            continue
        elements.append({"type": "table", "bbox": (x, y, x + cw, y + ch), "confidence": 0.8})

    # --- Text block detection: connected components on dilated text ---
    text_only = cv2.subtract(binary, table_mask)
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
    text_dilated = cv2.dilate(text_only, text_kernel)
    text_contours, _ = cv2.findContours(text_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in text_contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < 0.001 * w * h:
            continue
        # Heuristic: tall narrow regions are probably logos/figures
        aspect = cw / max(ch, 1)
        if aspect > 0.3 and ch < h * 0.25:
            elem_type = "text"
        elif aspect < 0.5 and area > 0.01 * w * h:
            elem_type = "figure"
        else:
            elem_type = "text"
        # Skip if mostly covered by a table region
        already_table = any(
            _iou((x, y, x + cw, y + ch), e["bbox"]) > 0.6
            for e in elements if e["type"] == "table"
        )
        if already_table:
            continue
        elements.append({"type": elem_type, "bbox": (x, y, x + cw, y + ch), "confidence": 0.6})

    # --- Logo/figure detection: large non-text, roughly square blobs ---
    large_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    large_regions = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, large_kernel)
    large_contours, _ = cv2.findContours(large_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in large_contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < 0.003 * w * h or area > 0.2 * w * h:
            continue
        aspect = cw / max(ch, 1)
        # Square-ish and moderate size → likely logo
        if 0.5 < aspect < 2.0 and area > 0.005 * w * h:
            already_covered = any(
                _iou((x, y, x + cw, y + ch), e["bbox"]) > 0.5
                for e in elements
            )
            if not already_covered:
                elements.append({"type": "figure", "bbox": (x, y, x + cw, y + ch),
                                  "confidence": 0.5})

    # --- Title detection: very wide short blocks near the top ---
    top_zone = int(h * 0.25)
    title_candidates = [
        e for e in elements
        if e["type"] == "text" and e["bbox"][1] < top_zone
        and (e["bbox"][2] - e["bbox"][0]) > w * 0.3
        and (e["bbox"][3] - e["bbox"][1]) < h * 0.06
    ]
    for e in title_candidates:
        e["type"] = "title"

    return elements


def _iou(boxA, boxB) -> float:
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


# ---------------------------------------------------------------------------
# layoutparser-based extraction (optional)
# ---------------------------------------------------------------------------

_lp_model = None

def _get_lp_model():
    global _lp_model
    if _lp_model is None:
        import layoutparser as lp
        # PubLayNet model — may fail if weights not cached
        _lp_model = lp.Detectron2LayoutModel(
            "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
            label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
        )
    return _lp_model


def _lp_extract(img_rgb: np.ndarray) -> list[dict]:
    import layoutparser as lp
    model = _get_lp_model()
    layout = model.detect(img_rgb)
    elements = []
    for block in layout:
        x1, y1, x2, y2 = (int(v) for v in [block.block.x_1, block.block.y_1,
                                              block.block.x_2, block.block.y_2])
        elements.append({
            "type": block.type.lower(),
            "bbox": (x1, y1, x2, y2),
            "confidence": float(block.score),
        })
    return elements


# ---------------------------------------------------------------------------
# Skeleton generation
# ---------------------------------------------------------------------------

def _make_skeleton(img_bgr: np.ndarray, elements: list[dict]) -> np.ndarray:
    """Mask all text regions to white, leaving structural elements visible."""
    skeleton = img_bgr.copy()
    for elem in elements:
        if elem["type"] in ("text", "title", "list"):
            x1, y1, x2, y2 = elem["bbox"]
            # Pad slightly to catch all text pixels
            pad = 2
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(img_bgr.shape[1], x2 + pad), min(img_bgr.shape[0], y2 + pad)
            skeleton[y1:y2, x1:x2] = 255  # white-out
    return skeleton


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_layout(image_path: str) -> list[dict]:
    """
    Extract layout elements from an invoice PNG.
    Returns list of dicts with keys: type, bbox (x1,y1,x2,y2), confidence.
    Side-effect: writes skeleton PNG to output/skeletons/.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Choose backend
    use_lp = _lp_available and LAYOUT_BACKEND != "opencv"
    if use_lp:
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            elements = _lp_extract(img_rgb)
        except Exception as e:
            print(f"[WARN] layoutparser failed ({e}), using OpenCV fallback", file=sys.stderr)
            elements = _opencv_extract(img_bgr)
    else:
        elements = _opencv_extract(img_bgr)

    # Produce and save skeleton
    skeleton_bgr = _make_skeleton(img_bgr, elements)
    stem = Path(image_path).stem
    skeleton_path = SKELETONS_DIR / f"{stem}_skeleton.png"
    cv2.imwrite(str(skeleton_path), skeleton_bgr)

    return elements


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python layout_extraction.py <image_path>")
        sys.exit(1)
    elems = extract_layout(path)
    for e in elems:
        print(e)
