"""
Stage 4 — Similarity scores.

graph_similarity(ga, gb)       -> float [0, 1]
embedding_similarity(ea, eb)   -> float [0, 1]
text_similarity(path_a, path_b)-> float [0, 1]
combined_fraud_score(gs, es)   -> float [0, 1]
"""

import difflib
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "output" / "ocr_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GED_TIMEOUT_SECONDS = 5  # per pair

# nx.optimize_graph_edit_distance is exponential and, empirically on this
# dataset's real invoice graphs (13-23 nodes), never converges within any
# reasonable timeout — every pair falls back to the structural feature
# vector anyway. Rather than spend GED_TIMEOUT_SECONDS finding that out via
# a background thread (or a subprocess, which hangs here because spawning a
# new process after both pandas and torch are loaded in this process
# deadlocks on this platform), skip the attempt outright above this size and
# go straight to the fallback. Below the threshold GED can plausibly
# converge quickly, so it's still attempted with a short thread timeout.
GED_MAX_NODES = 10


# ---------------------------------------------------------------------------
# Graph similarity
# ---------------------------------------------------------------------------

def _structural_feature_vector(G: nx.Graph) -> np.ndarray:
    """Fallback: encode a graph as a fixed-length structural feature vector."""
    if len(G.nodes) == 0:
        return np.zeros(10)

    type_counts = {"text": 0, "title": 0, "table": 0, "figure": 0, "list": 0, "unknown": 0}
    for _, d in G.nodes(data=True):
        t = d.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    n_nodes = len(G.nodes)
    n_edges = len(G.edges)
    avg_degree = (2 * n_edges / n_nodes) if n_nodes else 0

    align_edges = sum(1 for _, _, d in G.edges(data=True)
                      if d.get("align") in ("horizontal", "vertical"))
    align_ratio = align_edges / max(n_edges, 1)

    # Layout bounding box aspect ratio
    if n_nodes > 0:
        cx_vals = [d["centroid"][0] for _, d in G.nodes(data=True)]
        cy_vals = [d["centroid"][1] for _, d in G.nodes(data=True)]
        layout_w = max(cx_vals) - min(cx_vals) + 1e-6
        layout_h = max(cy_vals) - min(cy_vals) + 1e-6
        aspect = layout_w / layout_h
    else:
        aspect = 1.0

    type_dist = [type_counts.get(t, 0) / max(n_nodes, 1)
                 for t in ["text", "title", "table", "figure", "list", "unknown"]]

    feat = np.array([
        n_nodes / 30.0,       # normalised node count
        n_edges / 60.0,       # normalised edge count
        avg_degree / 6.0,
        align_ratio,
        aspect,
    ] + type_dist)
    return feat


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def graph_similarity(graph_a: nx.Graph, graph_b: nx.Graph) -> float:
    """
    Compute normalised graph similarity in [0, 1].
    Primary: GED with timeout. Fallback: structural feature cosine similarity.
    """
    na, nb = len(graph_a.nodes), len(graph_b.nodes)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0

    def _fallback():
        logger.debug("using structural_fallback")
        fa = _structural_feature_vector(graph_a)
        fb = _structural_feature_vector(graph_b)
        return (_cosine_sim(fa, fb) + 1) / 2  # map [-1,1] → [0,1]

    if max(na, nb) > GED_MAX_NODES:
        return _fallback()

    # Try GED with a wall-clock timeout via a background thread.
    # optimize_graph_edit_distance yields a monotonically improving upper
    # bound; we keep the best (last) one seen before the timeout.
    ged_result = [None]
    ged_exception = [None]

    def _run_ged():
        try:
            gen = nx.optimize_graph_edit_distance(
                graph_a, graph_b,
                node_subst_cost=lambda u, v: 0.0 if u.get("type") == v.get("type") else 1.0,
                node_del_cost=lambda u: 1.0,
                node_ins_cost=lambda u: 1.0,
            )
            for value in gen:
                ged_result[0] = value
        except Exception as e:
            ged_exception[0] = e

    t = threading.Thread(target=_run_ged, daemon=True)
    t.start()
    t.join(timeout=GED_TIMEOUT_SECONDS)

    if t.is_alive() or ged_result[0] is None:
        # Didn't converge (or failed) within the timeout → the best bound
        # seen so far may still be the trivial "delete all + insert all"
        # estimate, so fall back to the structural feature vector instead.
        # The thread (if still running) is small enough (<= GED_MAX_NODES
        # nodes) to finish on its own shortly after and is harmless as a
        # detached daemon thread.
        return _fallback()

    # GED converged before the timeout — use the exact/near-exact bound.
    max_ged = na + nb  # worst case: delete all a + insert all b
    sim = max(0.0, 1.0 - ged_result[0] / max_ged)
    logger.debug(f"GED={ged_result[0]:.2f}, sim={sim:.4f}")
    return float(sim)


# ---------------------------------------------------------------------------
# Embedding similarity
# ---------------------------------------------------------------------------

def embedding_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Cosine similarity normalised to [0, 1]."""
    raw = _cosine_sim(embedding_a, embedding_b)
    return (raw + 1) / 2  # map [-1,1] → [0,1]


# ---------------------------------------------------------------------------
# Text similarity (OCR-based baseline)
# ---------------------------------------------------------------------------

_easyocr_reader = None

# Full-resolution invoice pages (~1600x2250) crash easyocr's quantized
# recognition network on this platform (access violation inside the ONNX/
# oneDNN quantized LSTM). Downscaling before OCR avoids it entirely and is
# a fine tradeoff since text_similarity is only a comparison baseline, not
# part of combined_fraud_score.
_OCR_DOWNSCALE = 3


def _get_ocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


def _ocr_text(image_path: str) -> str:
    """Run EasyOCR on a downscaled copy of the image; cache result by file hash."""
    import hashlib
    with open(image_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    cache_path = CACHE_DIR / f"{file_hash}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    from PIL import Image
    reader = _get_ocr_reader()
    img = Image.open(image_path).convert("RGB")
    small = img.resize((max(1, img.width // _OCR_DOWNSCALE), max(1, img.height // _OCR_DOWNSCALE)))
    results = reader.readtext(np.array(small), detail=0)
    text = " ".join(results)
    cache_path.write_text(text, encoding="utf-8")
    return text


def text_similarity(image_path_a: str, image_path_b: str) -> float:
    """OCR both images and compute difflib SequenceMatcher ratio."""
    text_a = _ocr_text(image_path_a)
    text_b = _ocr_text(image_path_b)
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0
    return difflib.SequenceMatcher(None, text_a, text_b).ratio()


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------

def combined_fraud_score(
    graph_sim: float,
    embedding_sim: float,
    w1: float = 0.5,
    w2: float = 0.5,
) -> float:
    """Weighted combination of graph and embedding similarity (text excluded)."""
    return w1 * graph_sim + w2 * embedding_sim


if __name__ == "__main__":
    # Quick smoke test
    G1 = nx.path_graph(4)
    G2 = nx.path_graph(4)
    print("Same graph GED sim:", graph_similarity(G1, G2))
    G3 = nx.cycle_graph(6)
    print("Different graph sim:", graph_similarity(G1, G3))
