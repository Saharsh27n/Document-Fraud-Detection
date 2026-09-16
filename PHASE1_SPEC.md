# Phase 1 Spec — Structural Fraud Detection for Medical Invoices

## Goal

Build the Phase 1 pipeline of a system that detects **fraudulent medical invoice
template reuse** via structural/layout analysis.

**Core hypothesis:** structural similarity between two documents catches template
fraud that plain text similarity misses — because fraudsters reuse the same visual
template but swap patient names, amounts, dates, and logos.

This is the **real architecture at small scale**, not a simplified placeholder.
Every component here gets scaled up later (bigger dataset, fine-tuned GNN,
dashboard). Implement each stage properly. Do **not** substitute shortcuts like
perceptual hashing for the structural comparison.

---

## Repo structure

```
/generator/
    templates/          # HTML+CSS invoice templates (6 distinct layouts)
    logos/              # generated placeholder logos
    generate_dataset.py # synthetic invoice generator
/data/
    invoices/           # generated PNG invoices
    labels.csv          # ground truth pairs
    manifest.csv        # per-image provenance
/output/
    skeletons/
    graphs/
layout_extraction.py
graph_builder.py
embedding.py
similarity.py
evaluate.py
demo.py
requirements.txt
README.md
```

---

## Stage 0 — Synthetic dataset generator (build this FIRST)

The dataset is generated programmatically. No manual invoice creation.

### Templates

Create **6 visually distinct** medical invoice templates as HTML + CSS in
`/generator/templates/`. They must differ **structurally**, not just in colour:

| Template | Structural character |
|---|---|
| `t1_classic`   | header band, single-column line-item table with full borders |
| `t2_modern`    | left sidebar for hospital info, borderless table with row stripes |
| `t3_compact`   | two-column header (patient left / bill right), tight grid table |
| `t4_gov`       | boxed sections, heavy rules, stamp area bottom-right |
| `t5_clinic`    | logo centred top, no table borders, itemised list style |
| `t6_corporate` | right-aligned logo, split totals block, footer bar |

Each template is a Jinja2 template with placeholders for: hospital name, address,
logo path, patient name, patient ID, age/sex, bill number, bill date, admission
and discharge dates, doctor name, diagnosis, a variable-length list of line items
(description, qty, rate, amount), subtotal, tax, discount, total, payment mode.

### Content filling

Use `Faker` (with `Faker('en_IN')`) plus a hardcoded list of ~40 realistic Indian
medical line items (Consultation Fee, CBC, X-Ray Chest PA, ECG, Room Rent —
General Ward, IV Fluids, Nursing Charges, MRI Brain, Physiotherapy Session, etc.)
and ~25 diagnoses. Line item count varies 3–12 per invoice. Amounts computed
consistently so subtotal/tax/total actually add up.

### Logos

Generate ~10 simple placeholder logos programmatically with Pillow (coloured
geometric shapes + 2–3 letter monogram), saved to `/generator/logos/`. Never use
real hospital logos or brand assets.

### Rendering

Render HTML → PNG using **Playwright** (headless Chromium, A4 viewport, deviceScaleFactor 2).

```python
pip install playwright
playwright install chromium
```

If the Chromium download fails, fall back to a **ReportLab → PDF → PyMuPDF → PNG**
renderer and reimplement the six layouts with ReportLab primitives. Keep the
renderer behind a single interface `render(template_id, context) -> PIL.Image` so
the rest of the pipeline is unaffected.

### Scan-noise augmentation (important — do not skip)

Clean digital renders make the task trivially easy and will be rejected by
reviewers. Apply a randomised augmentation to every invoice:

- rotation ±1.5°
- perspective warp, slight
- Gaussian noise, low sigma
- brightness/contrast jitter
- JPEG recompression at quality 70–90
- occasional faint horizontal scan lines or a light paper-texture overlay

Expose the augmentation strength via a `--noise {none,light,heavy}` flag so we can
report an ablation over noise level in the paper.

### Pair construction

Generate **400 invoices**, then emit **200 labelled pairs** in `labels.csv` with
columns `doc_a, doc_b, pair_type, is_fraud`:

| `pair_type` | How it is generated | `is_fraud` |
|---|---|---|
| `same_template_diff_content` | same template, same logo, fully regenerated content | 1 |
| `same_template_diff_logo` | same template, different logo + different hospital name, different content | 1 |
| `different_template` | two different templates, different content | 0 |
| `same_issuer_updated_template` | same hospital identity, but template mutated: logo moved, a section reordered, a column added/removed, font size shifted | 0 |

Balance the four types roughly evenly (50 pairs each). The fourth type is the
**hard negative** — it is what stops the system being a trivial exact-layout matcher,
and it is the class the paper's error analysis will focus on.

Also write `manifest.csv` recording, for every generated image: filename,
template_id, logo_id, hospital name, noise seed, and the RNG seed. Everything must
be reproducible from a single `--seed` argument.

### CLI

```bash
python generator/generate_dataset.py --n-invoices 400 --n-pairs 200 --seed 42 --noise light
```

---

## Stage 1 — `layout_extraction.py`

- Load an invoice image with OpenCV/PIL.
- Detect layout elements: text blocks, tables, titles, figures/logos.
- Primary path: `layoutparser` with a pretrained PubLayNet model.
- **Fallback path (implement this regardless):** pure OpenCV detection —
  morphological horizontal/vertical line extraction for table borders, contour +
  dilation-based connected components for text blocks, large non-text contours for
  logo regions. PubLayNet is trained on academic papers and often underperforms on
  invoice tables, so the OpenCV path must be able to carry the pipeline alone.
- Select the backend with a config flag; default to OpenCV if layoutparser or its
  model backend fails to import.
- `extract_layout(image_path: str) -> list[dict]`, each dict `{type, bbox: (x1,y1,x2,y2), confidence}`.
- Also produce the **skeleton**: original image with all detected text-block
  regions masked to white, leaving lines, borders, table structure, and logo shape.
  Save to `/output/skeletons/`.

## Stage 2 — `graph_builder.py`

- `build_spatial_graph(layout_elements: list[dict]) -> networkx.Graph`
- Node per element, attributes: `type`, `bbox`, `centroid`, normalised width/height,
  area ratio relative to page.
- Edges: connect k-nearest neighbours by centroid distance (k=3), plus edges
  between elements aligned vertically or horizontally within a tolerance
  (~2% of page dimension). Edge attributes: normalised centroid distance,
  alignment type (`horizontal` / `vertical` / `knn`).
- All coordinates normalised to page size so the graph is resolution-independent.
- Render each graph to `/output/graphs/` with networkx + matplotlib, nodes coloured
  by element type and positioned at their true centroids (not a spring layout) —
  the visualisation should look like the invoice's structure.

## Stage 3 — `embedding.py`

- Pretrained ResNet-18 (torchvision ImageNet weights, no fine-tuning at this stage),
  final classification layer removed.
- `get_skeleton_embedding(skeleton_image_path: str) -> np.ndarray` — resize,
  ImageNet-normalise, forward pass, return flat penultimate feature vector.
- Cache embeddings to disk keyed by image hash so `evaluate.py` does not recompute
  across runs.

## Stage 4 — `similarity.py`

1. `graph_similarity(graph_a, graph_b) -> float`
   - Primary: `networkx.optimize_graph_edit_distance` with a timeout.
   - If it exceeds the timeout, fall back to a structural feature vector —
     node-type distribution, node count, edge count, average degree, alignment-edge
     ratio, layout bounding-box aspect ratio — compared by cosine similarity.
   - Log which method was used per pair; normalise output to 0–1.

2. `embedding_similarity(embedding_a, embedding_b) -> float` — cosine, normalised to 0–1.

3. `text_similarity(image_path_a, image_path_b) -> float`
   - `easyocr` on the **original** images (not skeletons).
   - `difflib.SequenceMatcher(None, text_a, text_b).ratio()`.
   - This is the **naive baseline** the paper compares against. Cache OCR output
     to disk — it is the slowest step in the pipeline.

4. `combined_fraud_score(graph_sim, embedding_sim, w1=0.5, w2=0.5) -> float`
   - Weighted combination of graph and embedding similarity only.
   - Text similarity is explicitly **excluded** — it is a separate baseline, not a
     component of the structural method.

## Stage 5 — `evaluate.py`

- Iterate all pairs in `labels.csv`, compute all four scores per pair.
- Threshold `combined_fraud_score` (default 0.6, configurable) → fraud/clean prediction.
- Do the same for the text baseline with its own threshold.
- Report precision, recall, F1, and confusion matrix for **both** methods.
- Additionally report a **per-`pair_type` breakdown** — this is what shows the
  method handles the hard-negative `same_issuer_updated_template` class.
- Sweep the threshold from 0.1 to 0.9, compute ROC-AUC and PR-AUC for both methods,
  and save an ROC curve plot to `/output/`.
- Write `/output/evaluation_results.csv` (per-pair scores) and
  `/output/evaluation_summary.csv` (aggregate metrics). Both are paper-ready tables.

## Stage 6 — `demo.py`

```bash
python demo.py <img_a> <img_b>
```

- Runs the full pipeline on one pair.
- Matplotlib 2×3 grid: original A, original B, skeleton A, skeleton B, graph A, graph B.
- Prints:

```
Text Similarity (baseline):      0.XX  -> Naive verdict: [FRAUD/CLEAN]
Structural Similarity (graph):   0.XX
Structural Similarity (embed):   0.XX
Combined Fraud Score:            0.XX  -> Our verdict: [FRAUD/CLEAN]
Ground truth:                    [FRAUD/CLEAN]  (pair_type: ...)
```

- Add a `--save <path>` flag to export the figure as PNG for slides.

---

## requirements.txt

```
opencv-python
playwright
jinja2
faker
reportlab
pymupdf
torch
torchvision
networkx
easyocr
scikit-learn
matplotlib
pandas
numpy
Pillow
layoutparser        # optional — pipeline must run without it
```

---

## Constraints

- Build **Stage 0 first** and verify generated invoices look plausible before
  touching the pipeline stages. Show me 3 sample renders for approval.
- Every stage must degrade gracefully: if `layoutparser` or Playwright fails,
  the documented fallback takes over and the pipeline still completes end to end.
- Prioritise `evaluate.py` running end-to-end over any single component being
  perfect. The comparison table is the primary deliverable.
- Fully reproducible from a seed. No hardcoded absolute paths.
- Clear inline comments — this will be extended in later phases and read by
  teammates and a faculty guide.
- Do not use real hospital names, real logos, or scraped real invoices. All content
  is synthetic.
- Write `README.md` with setup, the exact commands to reproduce the dataset and
  results, and a short note on dataset composition for the paper's methodology section.

## Acceptance checks

1. `python generator/generate_dataset.py --seed 42` produces 400 invoices + labels.csv + manifest.csv
2. Re-running with the same seed produces byte-identical output
3. `python evaluate.py` produces both CSVs and prints both metric tables
4. `python demo.py <a> <b>` produces the 2×3 figure and the score block
5. Structural F1 beats text-baseline F1 on the two fraud pair types
