"""
Stage 5 — Full pipeline evaluation.

Reads data/labels.csv, runs all four scores per pair, then:
  * splits the pairs 50/50 (seeded) into a calibration half and a test half,
  * picks each method's decision threshold on the calibration half
    (Youden's J = TPR - FPR on the ROC curve),
  * reports precision / recall / F1 / confusion matrix on the test half only,
  * reports ROC-AUC and PR-AUC over all pairs.

Writes output/evaluation_results.csv, output/evaluation_summary.csv,
output/calibration.json (thresholds used by demo.py and the web app), and the
ROC/PR plot.

Usage:
    python evaluate.py                    # all pairs
    python evaluate.py --limit 140        # first 140 pairs (fast: OCR cached)
    python evaluate.py --threshold 0.8    # force the structural threshold
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
)

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
INVOICES_DIR = DATA_DIR / "invoices"
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SKELETONS_DIR = OUTPUT_DIR / "skeletons"
CALIBRATION_PATH = OUTPUT_DIR / "calibration.json"

DEFAULT_THRESHOLD = 0.6  # used only if calibration is impossible
AMBER_MARGIN = 0.03      # scores within this distance below threshold → "amber"


def compute_document_features(invoice_path: Path):
    """Layout → skeleton → graph → embedding for one document."""
    from layout_extraction import extract_layout
    from graph_builder import build_spatial_graph
    from embedding import get_skeleton_embedding
    from PIL import Image

    elems = extract_layout(str(invoice_path))  # also writes the skeleton PNG
    with Image.open(str(invoice_path)) as img:
        shape = (img.height, img.width)
    graph = build_spatial_graph(elems, shape)
    skel = SKELETONS_DIR / f"{invoice_path.stem}_skeleton.png"
    emb = get_skeleton_embedding(str(skel))
    return graph, emb


def youden_threshold(y_true, y_score, fallback=DEFAULT_THRESHOLD) -> float:
    """Threshold maximising TPR - FPR. Falls back if only one class present."""
    if len(set(y_true)) < 2:
        return fallback
    fpr, tpr, thr = roc_curve(y_true, y_score)
    finite = np.isfinite(thr)
    j = (tpr - fpr)[finite]
    return float(thr[finite][int(np.argmax(j))])


def evaluate(threshold=None, limit=None, seed=42):
    labels_path = DATA_DIR / "labels.csv"
    if not labels_path.exists():
        print(f"[ERROR] {labels_path} not found — run generate_dataset.py first")
        sys.exit(1)

    from similarity import graph_similarity, embedding_similarity, text_similarity, combined_fraud_score

    labels_df = pd.read_csv(labels_path)
    if limit:
        labels_df = labels_df.head(limit)
    print(f"Evaluating {len(labels_df)} pairs from {labels_path}")

    # Per-document cache within this run (documents appear in several pairs)
    doc_cache = {}

    def features(name):
        if name not in doc_cache:
            doc_cache[name] = compute_document_features(INVOICES_DIR / name)
        return doc_cache[name]

    results = []
    for idx, row in labels_df.iterrows():
        doc_a, doc_b = str(row["doc_a"]), str(row["doc_b"])
        path_a, path_b = INVOICES_DIR / doc_a, INVOICES_DIR / doc_b
        if not path_a.exists() or not path_b.exists():
            print(f"[WARN] Missing file for pair {idx}: {doc_a} / {doc_b}")
            continue

        t0 = time.time()
        G_a, emb_a = features(doc_a)
        G_b, emb_b = features(doc_b)
        gs = graph_similarity(G_a, G_b)
        es = embedding_similarity(emb_a, emb_b)
        cs = combined_fraud_score(gs, es)
        ts = text_similarity(str(path_a), str(path_b))  # baseline only

        results.append({
            "doc_a": doc_a, "doc_b": doc_b,
            "pair_type": str(row["pair_type"]), "is_fraud": int(row["is_fraud"]),
            "graph_sim": round(gs, 4), "embedding_sim": round(es, 4),
            "combined_score": round(cs, 4), "text_sim": round(ts, 4),
            "elapsed_s": round(time.time() - t0, 2),
        })
        if (len(results)) % 20 == 0:
            print(f"  {len(results)}/{len(labels_df)} pairs done ...", flush=True)

    df = pd.DataFrame(results)

    # ---- Seeded 50/50 calibration / test split ----
    perm = np.random.RandomState(seed).permutation(len(df))
    split = np.full(len(df), "test", dtype=object)
    split[perm[: len(df) // 2]] = "calibration"
    df["split"] = split
    calib, test = df[df.split == "calibration"], df[df.split == "test"]

    struct_thr = threshold if threshold is not None else youden_threshold(
        calib.is_fraud.values, calib.combined_score.values)
    text_thr = youden_threshold(calib.is_fraud.values, calib.text_sim.values)

    df["struct_pred"] = (df.combined_score >= struct_thr).astype(int)
    df["text_pred"] = (df.text_sim >= text_thr).astype(int)
    results_path = OUTPUT_DIR / "evaluation_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\nWrote {results_path}")

    calibration = {
        "structural_threshold": round(struct_thr, 4),
        "text_threshold": round(text_thr, 4),
        "amber_margin": AMBER_MARGIN,
        "n_calibration_pairs": int(len(calib)),
        "n_test_pairs": int(len(test)),
        "seed": seed,
    }
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(f"Wrote {CALIBRATION_PATH}: {calibration}")

    test = df[df.split == "test"]
    _print_metrics(test, df, struct_thr, text_thr)
    _save_roc_plot(df.is_fraud.values, df.combined_score.values, df.text_sim.values)
    _save_summary(test, df, struct_thr, text_thr)


def _print_metrics(test, all_df, struct_thr, text_thr):
    y = test.is_fraud.values
    for name, col, thr in [("STRUCTURAL METHOD", "struct_pred", struct_thr),
                           ("TEXT BASELINE", "text_pred", text_thr)]:
        print("\n" + "=" * 64)
        print(f"{name}  (threshold={thr:.4f}, test half n={len(test)})")
        print("=" * 64)
        print(classification_report(y, test[col].values, target_names=["clean", "fraud"],
                                    zero_division=0))
        print("Confusion matrix (rows=actual, cols=predicted):")
        print(confusion_matrix(y, test[col].values, labels=[0, 1]))

    print("\n" + "=" * 64)
    print("PER PAIR-TYPE BREAKDOWN (test half) — accuracy = correctly labelled")
    print("=" * 64)
    print(f"  {'pair_type':32s} {'n':>3s}  {'struct_acc':>10s}  {'text_acc':>8s}")
    for pt, g in test.groupby("pair_type"):
        sa = accuracy_score(g.is_fraud, g.struct_pred)
        ta = accuracy_score(g.is_fraud, g.text_pred)
        print(f"  {pt:32s} {len(g):3d}  {sa:10.3f}  {ta:8.3f}")

    y_all = all_df.is_fraud.values
    try:
        print(f"\nROC-AUC (all pairs)  structural={roc_auc_score(y_all, all_df.combined_score):.4f}"
              f"  text={roc_auc_score(y_all, all_df.text_sim):.4f}")
    except ValueError:
        pass


def _save_roc_plot(y_true, y_struct, y_text):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fpr_s, tpr_s, _ = roc_curve(y_true, y_struct)
        fpr_t, tpr_t, _ = roc_curve(y_true, y_text)
        axes[0].plot(fpr_s, tpr_s, label=f"Structural (AUC={roc_auc_score(y_true, y_struct):.3f})")
        axes[0].plot(fpr_t, tpr_t, label=f"Text baseline (AUC={roc_auc_score(y_true, y_text):.3f})",
                     linestyle="--")
        axes[0].plot([0, 1], [0, 1], "k:", label="Random")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve")
        axes[0].legend()

        prec_s, rec_s, _ = precision_recall_curve(y_true, y_struct)
        prec_t, rec_t, _ = precision_recall_curve(y_true, y_text)
        axes[1].plot(rec_s, prec_s, label=f"Structural (AP={average_precision_score(y_true, y_struct):.3f})")
        axes[1].plot(rec_t, prec_t, label=f"Text baseline (AP={average_precision_score(y_true, y_text):.3f})",
                     linestyle="--")
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


def _save_summary(test, all_df, struct_thr, text_thr):
    rows = []
    y_all = all_df.is_fraud.values
    y = test.is_fraud.values
    for method, col, score_col, thr in [
        ("structural", "struct_pred", "combined_score", struct_thr),
        ("text_baseline", "text_pred", "text_sim", text_thr),
    ]:
        try:
            auc = roc_auc_score(y_all, all_df[score_col])
            ap = average_precision_score(y_all, all_df[score_col])
        except ValueError:
            auc, ap = float("nan"), float("nan")
        p = test[col].values
        rows.append({
            "method": method, "threshold": round(thr, 4),
            "accuracy": round(accuracy_score(y, p), 4),
            "precision": round(precision_score(y, p, zero_division=0), 4),
            "recall": round(recall_score(y, p, zero_division=0), 4),
            "f1": round(f1_score(y, p, zero_division=0), 4),
            "roc_auc": round(auc, 4), "pr_auc": round(ap, 4),
            "n_pairs": len(y),
        })

    # Per pair-type rows (test half). F1 is only defined for the fraud types;
    # accuracy is the meaningful number for the clean types.
    for pt, g in test.groupby("pair_type"):
        for method, col, thr in [("structural", "struct_pred", struct_thr),
                                 ("text_baseline", "text_pred", text_thr)]:
            yt, p = g.is_fraud.values, g[col].values
            rows.append({
                "method": f"{method}_{pt}", "threshold": round(thr, 4),
                "accuracy": round(accuracy_score(yt, p), 4),
                "precision": round(precision_score(yt, p, zero_division=0), 4),
                "recall": round(recall_score(yt, p, zero_division=0), 4),
                "f1": round(f1_score(yt, p, zero_division=0), 4),
                "roc_auc": "", "pr_auc": "", "n_pairs": int(len(g)),
            })

    summary_path = OUTPUT_DIR / "evaluation_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None,
                        help="Force the structural threshold (default: calibrated)")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N pairs")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the calibration/test split")
    args = parser.parse_args()
    evaluate(threshold=args.threshold, limit=args.limit, seed=args.seed)
