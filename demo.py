"""
Stage 6 — Demo: run the full pipeline on a single pair and display results.

Usage:
    python demo.py <img_a> <img_b>
    python demo.py <img_a> <img_b> --save output/figure.png
    python demo.py <img_a> <img_b> --ground-truth fraud --pair-type same_template_diff_content
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
INVOICES_DIR = DATA_DIR / "invoices"
OUTPUT_DIR = REPO_ROOT / "output"
SKELETONS_DIR = OUTPUT_DIR / "skeletons"
GRAPHS_DIR = OUTPUT_DIR / "graphs"


def _load_or_generate_skeleton(invoice_path: str) -> Path:
    stem = Path(invoice_path).stem
    skel = SKELETONS_DIR / f"{stem}_skeleton.png"
    if not skel.exists():
        from layout_extraction import extract_layout
        extract_layout(str(invoice_path))
    return skel


def _load_or_generate_graph_img(invoice_path: str) -> Path:
    stem = Path(invoice_path).stem
    g_img = GRAPHS_DIR / f"{stem}_graph.png"
    if not g_img.exists():
        from layout_extraction import extract_layout
        from graph_builder import build_spatial_graph
        from PIL import Image
        elems = extract_layout(str(invoice_path))
        img = Image.open(str(invoice_path))
        build_spatial_graph(elems, (img.height, img.width), image_stem=stem)
    return g_img


def run_demo(img_a: str, img_b: str, save_path: str = None,
             ground_truth: str = None, pair_type: str = None):
    from layout_extraction import extract_layout
    from graph_builder import build_spatial_graph
    from embedding import get_skeleton_embedding
    from similarity import (graph_similarity, embedding_similarity,
                             text_similarity, combined_fraud_score)
    from PIL import Image

    path_a = Path(img_a)
    path_b = Path(img_b)
    if not path_a.is_absolute():
        path_a = INVOICES_DIR / img_a
    if not path_b.is_absolute():
        path_b = INVOICES_DIR / img_b

    if not path_a.exists():
        print(f"[ERROR] File not found: {path_a}")
        sys.exit(1)
    if not path_b.exists():
        print(f"[ERROR] File not found: {path_b}")
        sys.exit(1)

    print("Running pipeline ...")

    # Layout
    elems_a = extract_layout(str(path_a))
    elems_b = extract_layout(str(path_b))

    # Graphs
    pil_a = Image.open(str(path_a))
    pil_b = Image.open(str(path_b))
    G_a = build_spatial_graph(elems_a, (pil_a.height, pil_a.width), image_stem=path_a.stem)
    G_b = build_spatial_graph(elems_b, (pil_b.height, pil_b.width), image_stem=path_b.stem)

    # Skeletons
    skel_a = _load_or_generate_skeleton(str(path_a))
    skel_b = _load_or_generate_skeleton(str(path_b))

    # Embeddings
    emb_a = get_skeleton_embedding(str(skel_a))
    emb_b = get_skeleton_embedding(str(skel_b))

    # Scores
    gs = graph_similarity(G_a, G_b)
    es = embedding_similarity(emb_a, emb_b)
    cs = combined_fraud_score(gs, es)
    ts = text_similarity(str(path_a), str(path_b))

    # Thresholds calibrated by evaluate.py (fallback 0.6 if not run yet)
    import json
    cal_path = OUTPUT_DIR / "calibration.json"
    cal = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else {}
    struct_thr = cal.get("structural_threshold", 0.6)
    text_thr = cal.get("text_threshold", 0.6)
    struct_verdict = "FRAUD" if cs >= struct_thr else "CLEAN"
    text_verdict = "FRAUD" if ts >= text_thr else "CLEAN"
    gt_str = ground_truth.upper() if ground_truth else "N/A"

    # Print score block
    print()
    print(f"Text Similarity (baseline):      {ts:.4f}  -> Naive verdict: [{text_verdict}]")
    print(f"Structural Similarity (graph):   {gs:.4f}")
    print(f"Structural Similarity (embed):   {es:.4f}")
    print(f"Combined Fraud Score:            {cs:.4f}  -> Our verdict: [{struct_verdict}]")
    print(f"Ground truth:                    {gt_str}" +
          (f"  (pair_type: {pair_type})" if pair_type else ""))

    # Plot 2×3 grid
    graph_a_img = _load_or_generate_graph_img(str(path_a))
    graph_b_img = _load_or_generate_graph_img(str(path_b))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    def _show(ax, path, title):
        if Path(path).exists():
            img = mpimg.imread(str(path))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    _show(axes[0][0], path_a, f"Invoice A\n{path_a.name}")
    _show(axes[0][1], skel_a, f"Skeleton A")
    _show(axes[0][2], graph_a_img, f"Graph A  (nodes={len(G_a.nodes)})")

    _show(axes[1][0], path_b, f"Invoice B\n{path_b.name}")
    _show(axes[1][1], skel_b, f"Skeleton B")
    _show(axes[1][2], graph_b_img, f"Graph B  (nodes={len(G_b.nodes)})")

    title = (
        f"Text sim: {ts:.3f} [{text_verdict}]  |  "
        f"Graph sim: {gs:.3f}  |  Embed sim: {es:.3f}  |  "
        f"Combined: {cs:.3f} [{struct_verdict}]"
    )
    if ground_truth:
        title += f"  |  GT: {gt_str}"
    fig.suptitle(title, fontsize=11, y=0.99)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")
    else:
        # Save to a default location since headless
        default_out = OUTPUT_DIR / "demo_output.png"
        fig.savefig(str(default_out), dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to {default_out}")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo: fraud detection on a pair of invoices")
    parser.add_argument("img_a", help="Path or filename of invoice A")
    parser.add_argument("img_b", help="Path or filename of invoice B")
    parser.add_argument("--save", metavar="PATH", help="Save figure to this path")
    parser.add_argument("--ground-truth", choices=["fraud", "clean"], help="Actual label")
    parser.add_argument("--pair-type", help="Pair type label")
    args = parser.parse_args()

    run_demo(args.img_a, args.img_b, args.save, args.ground_truth, args.pair_type)
