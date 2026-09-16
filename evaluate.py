"""
Stage 5 — Full pipeline evaluation.

Reads data/labels.csv, runs all four scores per pair, thresholds,
prints metric tables, writes output/evaluation_results.csv and
output/evaluation_summary.csv, and saves ROC curve plot.
"""

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
)

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
INVOICES_DIR = DATA_DIR / "invoices"
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SKELETONS_DIR = OUTPUT_DIR / "skeletons"

FRAUD_THRESHOLD = 0.6  # default decision threshold


def _get_skeleton_path(invoice_filename: str) -> Path:
    stem = Path(invoice_filename).stem
    return SKELETONS_DIR / f"{stem}_skeleton.png"


def _ensure_skeleton(invoice_path: str):
    """Run layout extraction if skeleton doesn't exist yet."""
    skel = _get_skeleton_path(invoice_path)
    if not skel.exists():
        from layout_extraction import extract_layout
        extract_layout(str(INVOICES_DIR / invoice_path))
    return skel


def evaluate(threshold: float = FRAUD_THRESHOLD):
    labels_path = DATA_DIR / "labels.csv"
    if not labels_path.exists():
        print(f"[ERROR] {labels_path} not found — run generate_dataset.py first")
        sys.exit(1)

    from layout_extraction import extract_layout
    from graph_builder import build_spatial_graph
    from embedding import get_skeleton_embedding
    from similarity import graph_similarity, embedding_similarity, text_similarity, combined_fraud_score

    from PIL import Image

    labels_df = pd.read_csv(labels_path)
    print(f"Loaded {len(labels_df)} pairs from {labels_path}")

    results = []

    for idx, row in labels_df.iterrows():
        doc_a = str(row["doc_a"])
        doc_b = str(row["doc_b"])
        pair_type = str(row["pair_type"])
        is_fraud = int(row["is_fraud"])

        path_a = INVOICES_DIR / doc_a
        path_b = INVOICES_DIR / doc_b

        if not path_a.exists() or not path_b.exists():
            print(f"[WARN] Missing file for pair {idx}: {doc_a} / {doc_b}")
            continue

        t0 = time.time()

        # Layout + skeleton
        skel_a = _ensure_skeleton(doc_a)
        skel_b = _ensure_skeleton(doc_b)

        # Graphs
        elems_a = extract_layout(str(path_a))
        elems_b = extract_layout(str(path_b))
        img_a = Image.open(str(path_a))
        img_b = Image.open(str(path_b))
        G_a = build_spatial_graph(elems_a, (img_a.height, img_a.width))
        G_b = build_spatial_graph(elems_b, (img_b.height, img_b.width))

        # Embeddings
        emb_a = get_skeleton_embedding(str(skel_a))
        emb_b = get_skeleton_embedding(str(skel_b))

        # Scores
        gs = graph_similarity(G_a, G_b)
        es = embedding_similarity(emb_a, emb_b)
        cs = combined_fraud_score(gs, es)
        ts = text_similarity(str(path_a), str(path_b))

        elapsed = time.time() - t0

        results.append({
            "doc_a": doc_a,
            "doc_b": doc_b,
            "pair_type": pair_type,
            "is_fraud": is_fraud,
            "graph_sim": round(gs, 4),
            "embedding_sim": round(es, 4),
            "combined_score": round(cs, 4),
            "text_sim": round(ts, 4),
            "elapsed_s": round(elapsed, 2),
        })

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(labels_df)} pairs done ...")

    # ---- Save per-pair results ----
    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nWrote {results_path}")

    # ---- Metrics ----
    y_true = results_df["is_fraud"].values
    y_struct = results_df["combined_score"].values
    y_text = results_df["text_sim"].values

    _print_metrics(y_true, y_struct, y_text, results_df, threshold)
    _save_roc_plot(y_true, y_struct, y_text)
    _save_summary(y_true, y_struct, y_text, results_df, threshold)


def _print_metrics(y_true, y_struct, y_text, df, threshold):
    print("\n" + "=" * 60)
    print(f"STRUCTURAL METHOD  (threshold={threshold})")
    print("=" * 60)
    pred_struct = (y_struct >= threshold).astype(int)
    print(classification_report(y_true, pred_struct, target_names=["clean", "fraud"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_true, pred_struct))

    print("\n" + "=" * 60)
    print(f"TEXT BASELINE  (threshold={threshold})")
    print("=" * 60)
    pred_text = (y_text >= threshold).astype(int)
    print(classification_report(y_true, pred_text, target_names=["clean", "fraud"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, pred_text))

    print("\n" + "=" * 60)
    print("PER PAIR-TYPE BREAKDOWN (Structural method)")
    print("=" * 60)
    for pt in df["pair_type"].unique():
        mask = df["pair_type"] == pt
        yt = y_true[mask]
        yp = (y_struct[mask] >= threshold).astype(int)
        f1 = f1_score(yt, yp, zero_division=0)
        print(f"  {pt:40s}  F1={f1:.3f}  n={mask.sum()}")

    try:
        roc_struct = roc_auc_score(y_true, y_struct)
        roc_text = roc_auc_score(y_true, y_text)
        print(f"\nROC-AUC  structural={roc_struct:.4f}  text={roc_text:.4f}")
    except ValueError:
        pass


def _save_roc_plot(y_true, y_struct, y_text):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # ROC
        fpr_s, tpr_s, _ = roc_curve(y_true, y_struct)
        fpr_t, tpr_t, _ = roc_curve(y_true, y_text)
        auc_s = roc_auc_score(y_true, y_struct)
        auc_t = roc_auc_score(y_true, y_text)
        axes[0].plot(fpr_s, tpr_s, label=f"Structural (AUC={auc_s:.3f})")
        axes[0].plot(fpr_t, tpr_t, label=f"Text baseline (AUC={auc_t:.3f})", linestyle="--")
        axes[0].plot([0, 1], [0, 1], "k:", label="Random")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve")
        axes[0].legend()

        # PR
        prec_s, rec_s, _ = precision_recall_curve(y_true, y_struct)
        prec_t, rec_t, _ = precision_recall_curve(y_true, y_text)
        ap_s = average_precision_score(y_true, y_struct)
        ap_t = average_precision_score(y_true, y_text)
        axes[1].plot(rec_s, prec_s, label=f"Structural (AP={ap_s:.3f})")
        axes[1].plot(rec_t, prec_t, label=f"Text baseline (AP={ap_t:.3f})", linestyle="--")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall Curve")
        axes[1].legend()

        fig.tight_layout()
        plot_path = OUTPUT_DIR / "roc_pr_curves.png"
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        print(f"Saved ROC/PR plot to {plot_path}")
    except Exception as e:
        print(f"[WARN] Could not save ROC plot: {e}")


def _save_summary(y_true, y_struct, y_text, df, threshold):
    pred_struct = (y_struct >= threshold).astype(int)
    pred_text = (y_text >= threshold).astype(int)

    rows = []
    for method, y_pred, y_score in [
        ("structural", pred_struct, y_struct),
        ("text_baseline", pred_text, y_text),
    ]:
        try:
            auc = roc_auc_score(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
        except ValueError:
            auc, ap = float("nan"), float("nan")

        rows.append({
            "method": method,
            "threshold": threshold,
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc": round(auc, 4),
            "pr_auc": round(ap, 4),
            "n_pairs": len(y_true),
        })

    # Per pair-type rows
    for pt in df["pair_type"].unique():
        mask = df["pair_type"] == pt
        yt = y_true[mask]
        for method, y_pred in [
            ("structural", pred_struct[mask]),
            ("text_baseline", pred_text[mask]),
        ]:
            rows.append({
                "method": f"{method}_{pt}",
                "threshold": threshold,
                "precision": round(precision_score(yt, y_pred, zero_division=0), 4),
                "recall": round(recall_score(yt, y_pred, zero_division=0), 4),
                "f1": round(f1_score(yt, y_pred, zero_division=0), 4),
                "roc_auc": "",
                "pr_auc": "",
                "n_pairs": int(mask.sum()),
            })

    summary_df = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=FRAUD_THRESHOLD)
    args = parser.parse_args()
    evaluate(threshold=args.threshold)
