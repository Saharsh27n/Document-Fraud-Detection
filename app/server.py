"""
Web API for the showcase demo (FastAPI).

Wraps the existing pipeline stages unchanged:
    layout_extraction → graph_builder → embedding → similarity

Run from the repo root (no --reload: on this Windows setup, spawning a child
process after torch + pandas are loaded deadlocks):

    .venv\\Scripts\\python -m uvicorn app.server:app --port 8000
"""

import json
import sys
import threading
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so the pipeline modules import

DATA_DIR = REPO_ROOT / "data"
INVOICES_DIR = DATA_DIR / "invoices"
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = REPO_ROOT / "output"
STATIC_DIR = Path(__file__).resolve().parent / "static"
CALIBRATION_PATH = OUTPUT_DIR / "calibration.json"

# The pipeline (torch, easyocr, matplotlib) is not thread-safe; FastAPI runs
# sync endpoints in a thread pool, so serialise pipeline calls.
_pipeline_lock = threading.Lock()

app = FastAPI(title="SIH1441 Structural Template Fraud Detection")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calibration() -> dict:
    if CALIBRATION_PATH.exists():
        return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    return {"structural_threshold": 0.6, "text_threshold": 0.6, "amber_margin": 0.03}


def _resolve(doc: str) -> tuple[Path, str]:
    """Map a doc id ('invoice_x.png' or 'uploads/abc.png') to a file + URL.
    Only bare filenames are accepted, so requests can't escape the data folders."""
    if doc.startswith("uploads/"):
        name = Path(doc).name
        return UPLOADS_DIR / name, f"/uploads/{name}"
    name = Path(doc).name
    return INVOICES_DIR / name, f"/invoices/{name}"


def _band(score: float, thr: float, margin: float) -> str:
    if score >= thr:
        return "red"
    if score >= thr - margin:
        return "amber"
    return "green"


def _ocr_cached(path: Path) -> bool:
    import hashlib
    h = hashlib.md5(path.read_bytes()).hexdigest()
    return (OUTPUT_DIR / "ocr_cache" / f"{h}.txt").exists()


def _run_pair(doc_a: str, doc_b: str, include_text: bool = True) -> dict:
    from layout_extraction import extract_layout
    from graph_builder import build_spatial_graph
    from embedding import get_skeleton_embedding
    from similarity import (graph_similarity, embedding_similarity,
                            text_similarity, combined_fraud_score)
    from PIL import Image

    docs = []
    for doc in (doc_a, doc_b):
        path, url = _resolve(doc)
        if not path.exists():
            raise HTTPException(404, f"Not found: {doc}")
        elems = extract_layout(str(path))  # writes the skeleton PNG
        with Image.open(str(path)) as img:
            shape = (img.height, img.width)
        graph = build_spatial_graph(elems, shape, image_stem=path.stem)  # writes graph PNG
        emb = get_skeleton_embedding(str(OUTPUT_DIR / "skeletons" / f"{path.stem}_skeleton.png"))
        counts = {}
        for e in elems:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
        docs.append({
            "doc": doc, "path": path, "graph": graph, "emb": emb,
            "info": {
                "doc": doc,
                "original_url": url,
                "skeleton_url": f"/output/skeletons/{path.stem}_skeleton.png",
                "graph_url": f"/output/graphs/{path.stem}_graph.png",
                "n_nodes": graph.number_of_nodes(),
                "n_edges": graph.number_of_edges(),
                "element_counts": counts,
            },
        })

    a, b = docs
    gs = graph_similarity(a["graph"], b["graph"])
    es = embedding_similarity(a["emb"], b["emb"])
    cs = combined_fraud_score(gs, es)
    ts = text_similarity(str(a["path"]), str(b["path"])) if include_text else None

    cal = _calibration()
    s_thr, t_thr, margin = cal["structural_threshold"], cal["text_threshold"], cal["amber_margin"]
    return {
        "a": a["info"], "b": b["info"],
        "scores": {
            "graph_sim": round(gs, 4),
            "embedding_sim": round(es, 4),
            "combined_score": round(cs, 4),
            "text_sim": None if ts is None else round(ts, 4),
        },
        "structural_verdict": "FRAUD" if cs >= s_thr else "CLEAN",
        "structural_band": _band(cs, s_thr, margin),
        "text_verdict": None if ts is None else ("FRAUD" if ts >= t_thr else "CLEAN"),
        "calibration": cal,
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/pairs")
def pairs():
    labels = pd.read_csv(DATA_DIR / "labels.csv")
    manifest = pd.read_csv(DATA_DIR / "manifest.csv").set_index("filename")
    out = []
    for i, r in labels.iterrows():
        ma, mb = manifest.loc[r.doc_a], manifest.loc[r.doc_b]
        out.append({
            "id": int(i), "doc_a": r.doc_a, "doc_b": r.doc_b,
            "pair_type": r.pair_type, "is_fraud": int(r.is_fraud),
            "template_a": ma.template_id, "template_b": mb.template_id,
            "hospital_a": ma.hospital_name, "hospital_b": mb.hospital_name,
            "ocr_cached": _ocr_cached(INVOICES_DIR / r.doc_a) and _ocr_cached(INVOICES_DIR / r.doc_b),
        })
    return out


class CompareRequest(BaseModel):
    doc_a: str
    doc_b: str
    include_text: bool = True


@app.post("/api/compare")
def compare(req: CompareRequest):
    with _pipeline_lock:
        return _run_pair(req.doc_a, req.doc_b, req.include_text)


@app.post("/api/upload")
async def upload(file_a: UploadFile = File(...), file_b: UploadFile = File(...),
                 include_text: bool = True):
    ids = []
    for f in (file_a, file_b):
        ext = Path(f.filename or "").suffix.lower()
        if ext not in {".png", ".jpg", ".jpeg"}:
            raise HTTPException(400, "Upload PNG or JPG images")
        name = f"upload_{uuid.uuid4().hex[:10]}{ext}"
        (UPLOADS_DIR / name).write_bytes(await f.read())
        ids.append(f"uploads/{name}")
    with _pipeline_lock:
        return _run_pair(ids[0], ids[1], include_text)


@app.get("/api/results")
def results():
    summary_path = OUTPUT_DIR / "evaluation_summary.csv"
    results_path = OUTPUT_DIR / "evaluation_results.csv"
    summary = pd.read_csv(summary_path).fillna("").to_dict("records") if summary_path.exists() else []
    per_type = []
    if results_path.exists():
        df = pd.read_csv(results_path)
        per_type = (df.groupby("pair_type")[["combined_score", "text_sim", "graph_sim", "embedding_sim"]]
                    .mean().round(4).reset_index().to_dict("records"))
    return {"summary": summary, "mean_scores_by_type": per_type, "calibration": _calibration(),
            "roc_url": "/output/roc_pr_curves.png"}


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

app.mount("/invoices", StaticFiles(directory=str(INVOICES_DIR)), name="invoices")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))
