# Structural Fraud Detection for Medical Invoices — Phase 1

Detects fraudulent medical invoice template reuse via structural/layout analysis.
Core hypothesis: fraudsters reuse the same visual HTML template while swapping
patient names, amounts, and logos — structural similarity exposes this where
plain-text comparison fails.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium   # preferred renderer
```

If Chromium download fails, the pipeline falls back to ReportLab+PyMuPDF automatically.

## Reproduce the full experiment

```bash
# 1. Generate dataset (400 invoices, 200 labelled pairs)
python generator/generate_dataset.py --n-invoices 400 --n-pairs 200 --seed 42 --noise light

# 2. Run full evaluation (all six pipeline stages)
python evaluate.py

# 3. Demo on any pair
python demo.py invoice_0000_t1_classic.png invoice_0006_t1_classic.png --save output/demo.png
```

## Dataset composition (methodology note)

400 synthetic medical invoices generated from 6 structurally distinct HTML+CSS
templates (t1_classic through t6_corporate) filled with Faker-generated Indian
medical data. 10 placeholder logos (geometric shapes + monograms, Pillow-generated).

200 labelled pairs in four categories (50 each):

| pair_type | is_fraud | Description |
|---|---|---|
| `same_template_diff_content` | 1 | Same template, fully regenerated content |
| `same_template_diff_logo` | 1 | Same template, different logo + hospital |
| `different_template` | 0 | Two different templates |
| `same_issuer_updated_template` | 0 | Same hospital, different template — **hard negative** |

The `same_issuer_updated_template` class is the critical hard negative: it shares
the same issuer identity but uses a different layout, testing whether the method
has truly learned structural signatures rather than header text.

Every invoice is augmented with scan noise: rotation ±1.5°, perspective warp,
Gaussian noise, brightness/contrast jitter, JPEG recompression (70–90%), and
occasional scan lines or paper texture.

## Architecture

| Stage | File | Role |
|---|---|---|
| 0 | `generator/generate_dataset.py` | Synthetic dataset |
| 1 | `layout_extraction.py` | OpenCV layout detection → skeleton PNG |
| 2 | `graph_builder.py` | Spatial graph (kNN + alignment edges) |
| 3 | `embedding.py` | ResNet-18 skeleton embeddings (cached) |
| 4 | `similarity.py` | Graph GED + cosine embedding + OCR text |
| 5 | `evaluate.py` | Metrics, ROC/PR curves, CSV outputs |
| 6 | `demo.py` | 2×3 visualisation + score printout |

## Outputs

- `data/invoices/` — 400 PNG invoices
- `data/labels.csv` — labelled pairs
- `data/manifest.csv` — per-image provenance
- `output/skeletons/` — text-masked skeleton images
- `output/graphs/` — spatial graph visualisations
- `output/evaluation_results.csv` — per-pair scores
- `output/evaluation_summary.csv` — aggregate metrics table
- `output/roc_pr_curves.png` — ROC and PR curves

## Reproducibility

Everything is seeded from a single `--seed` integer. Re-running with `--seed 42`
produces byte-identical output.
