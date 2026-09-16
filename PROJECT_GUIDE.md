# Project Guide — Structural Fraud Detection for Medical Invoices

This is a complete walkthrough of the project: what it does, why it's built the way
it is, what every file is responsible for, how data moves through the pipeline, and
how to run a live demonstration for someone who has never seen it before.

For the original design spec, see `PHASE1_SPEC.md`. For quick-start commands, see
`README.md`. This document goes deeper — it explains the *reasoning*, not just the
commands.

---

## 1. The Idea, In Plain English

Medical-billing fraud often works like this: someone takes a real invoice template
(or steals a hospital's visual layout) and reuses it over and over, just swapping
the patient name, the amounts, the dates, and maybe the logo. The *content* changes
every time, but the *skeleton* — the layout, the table structure, where the logo
sits, how sections are arranged — stays identical.

Traditional fraud detection compares **text**: OCR the invoice, compare the words.
That's exactly what fraudsters defeat, because the words are supposed to be
different (different patients, different bills). This project's bet is that
comparing **structure** instead of text catches the reuse that text comparison
misses.

So for every pair of invoices, the pipeline produces two totally different kinds of
similarity:

- **Structural similarity** — "do these two documents share the same visual
  skeleton?" (graph shape + CNN features of the layout, text masked out)
- **Text similarity** — "do these two documents contain the same words?" (OCR +
  string comparison — this is the naive baseline being compared against, not part
  of the actual fraud score)

The whole point of the project is to show that structural similarity flags
template reuse (fraud) while correctly *not* flagging two invoices that just happen
to share a hospital name but use a genuinely different layout (a legitimate
template update). That second case — same issuer, different template — is the
**hard negative** class, and it's the one that proves the system isn't just doing
something trivial like matching header text.

---

## 2. The Pipeline, Stage by Stage

Six stages, each stage is exactly one file, each stage's output is the next
stage's input:

```
Stage 0          Stage 1              Stage 2            Stage 3
generate_dataset  layout_extraction    graph_builder      embedding
   .py       ──►      .py        ──►      .py       ──►      .py
   │                  │                   │                  │
   ▼                  ▼                   ▼                  ▼
400 invoice      bounding boxes      networkx.Graph      512-dim ResNet
PNGs + labels     + skeleton PNG     (nodes=elements,     feature vector
                  (text masked out)   edges=layout)        per skeleton

                                                                │
                        ┌───────────────────────────────────────┘
                        ▼
                   Stage 4: similarity.py
                   graph_sim, embedding_sim, text_sim
                        │
                        ▼
                   Stage 5: evaluate.py          Stage 6: demo.py
                   runs all 200 pairs,            runs ONE pair,
                   metrics + ROC/PR curves        2x3 figure + score printout
```

Every stage is built with a **primary method and a documented fallback**, because
this pipeline is meant to keep working even when a heavyweight dependency
(Playwright's Chromium, `layoutparser`'s PubLayNet model) isn't available in the
environment. That "degrade gracefully" rule shows up everywhere below.

---

## 3. File-by-File Reference

### `generator/generate_dataset.py` — Stage 0: build the fake world

This is the only file that doesn't analyze anything — it **creates** the synthetic
dataset the rest of the pipeline runs on. There is no real patient data anywhere in
this project; everything is fabricated.

What it does, in order:

1. **Generates 10 placeholder logos** (`generate_logos`) — simple geometric shapes
   with a 2–3 letter monogram, drawn with Pillow. Never a real hospital's logo.
2. **Builds invoice content** (`make_context`, `make_hospital_identity`) — uses
   `Faker('en_IN')` plus a hardcoded list of realistic Indian medical line items
   (Consultation Fee, CBC, X-Ray Chest PA, MRI Brain, Room Rent, etc.) and
   diagnoses, with amounts that actually add up to the printed subtotal/tax/total.
3. **Renders each invoice to a PNG** (`render`) using one of two interchangeable
   backends behind the same `render(template_id, context) -> PIL.Image` interface:
   - **Primary: Playwright** — launches headless Chromium, renders the HTML+CSS
     template at A4 size with 2x device scale (so text stays sharp), screenshots it.
   - **Fallback: ReportLab → PDF → PyMuPDF → PNG** — if Chromium can't be
     downloaded/launched, the six layouts are reimplemented with ReportLab drawing
     primitives, rendered to a PDF, then rasterized to PNG with PyMuPDF. The rest
     of the pipeline never knows which path was used.
4. **Applies scan-noise augmentation** (`augment`) — every invoice gets a small
   random rotation, a slight perspective warp, Gaussian noise, brightness/contrast
   jitter, and JPEG recompression, plus occasionally a faint scan-line or paper
   texture overlay. This matters: a perfectly clean digital render would make the
   whole problem trivially easy (and unrealistic — real invoices come from
   photocopies and scans). Strength is controlled with `--noise {none,light,heavy}`.
5. **Constructs 200 labelled pairs** (`generate_dataset`) across four types (see
   §5 below) and writes `data/labels.csv` (`doc_a, doc_b, pair_type, is_fraud`) and
   `data/manifest.csv` (every image's template, logo, hospital identity, and RNG
   seed, for full provenance).

Everything is driven by one `--seed` integer (`random.Random(seed)`), so
`python generator/generate_dataset.py --seed 42` twice in a row produces
byte-identical PNGs and CSVs.

### `generator/templates/*.html` — the 6 layouts

Six Jinja2 HTML+CSS templates, each **structurally** different (not just
recolored):

| Template | What makes it structurally distinct |
|---|---|
| `t1_classic.html` | header band, single-column line-item table with full borders |
| `t2_modern.html` | left sidebar for hospital info, borderless table with row stripes |
| `t3_compact.html` | two-column header (patient left / billing right), tight grid table |
| `t4_gov.html` | boxed sections, heavy rules, stamp area bottom-right |
| `t5_clinic.html` | logo centered top, no table borders, itemized list style |
| `t6_corporate.html` | right-aligned logo, split totals block, footer bar |

This diversity is what lets the "same template, different content" fraud pairs
look genuinely different from "different template" clean pairs — the templates
have to actually look different for the experiment to mean anything.

### `layout_extraction.py` — Stage 1: find the layout elements

Input: an invoice PNG. Output: `list[dict]` of `{type, bbox: (x1,y1,x2,y2),
confidence}` for every detected element (text block, title, table, figure/logo),
plus a **skeleton PNG** written to `output/skeletons/` — the original image with
every text-block region painted white, leaving only lines, borders, table
structure, and logo shapes visible. The skeleton is the thing that gets embedded in
Stage 3 — the idea is to compare *layout*, with the actual words removed so the CNN
can't cheat by reading text.

Two backends behind `extract_layout()`:
- **Primary:** `layoutparser` + a pretrained PubLayNet model (`Detectron2LayoutModel`).
- **Fallback (always available): pure OpenCV.** Horizontal/vertical morphological
  line extraction finds table borders; dilation + contour detection finds text
  blocks; large non-text, roughly-square contours are classified as logos; wide
  short blocks near the top of the page are reclassified as titles.

In the current environment `layoutparser` fails to import cleanly (its Detectron2
model wrapper isn't available), so **the OpenCV path is what actually runs the
whole pipeline right now** — which is fine, because the spec requires the OpenCV
path to be able to carry the pipeline alone, and it does.

### `graph_builder.py` — Stage 2: turn layout into a graph

Input: the `list[dict]` from Stage 1, plus the image's `(height, width)`. Output: a
`networkx.Graph`.

- **One node per layout element**, with attributes: `type`, normalized `bbox`,
  `centroid`, normalized width/height, `area_ratio` (relative to the page). All
  coordinates are normalized to `[0, 1]` so the graph is resolution-independent —
  two invoices rendered at different DPIs still produce comparable graphs.
- **Edges** connect: (a) each node to its `k=3` nearest neighbours by centroid
  distance, and (b) any two nodes whose centroids line up horizontally or
  vertically within a 2%-of-page tolerance (this captures "these fields are in the
  same row/column" the way a human would read the layout).
- Also renders a picture of the graph to `output/graphs/<name>_graph.png` — nodes
  colored by type, placed at their *true* centroid position (not a force-directed
  layout), so the picture visually resembles the invoice's structure. This is
  purely for the demo/visualisation, not used in scoring.

### `embedding.py` — Stage 3: turn the skeleton into a feature vector

Input: a skeleton PNG path. Output: a 512-dimensional `np.ndarray`.

Loads a pretrained ResNet-18 (ImageNet weights, **no fine-tuning** at this stage),
strips off the final classification layer, and takes the penultimate layer's
output as a feature vector — the standard "use a pretrained CNN as a generic visual
feature extractor" trick. Image is resized/cropped/ImageNet-normalized first.

Results are **cached to disk** in `output/embedding_cache/`, keyed by MD5 hash of
the image file, so re-running `evaluate.py` doesn't recompute embeddings for images
it's already seen.

### `similarity.py` — Stage 4: the four scoring functions

This is the file with the most moving parts. Four functions:

1. **`graph_similarity(graph_a, graph_b) -> float`** — meant to use
   `networkx.optimize_graph_edit_distance` (graph edit distance, GED) as the
   primary method, with a structural-feature-vector fallback (node-type
   distribution, node/edge counts, average degree, alignment-edge ratio, layout
   aspect ratio — compared by cosine similarity) if GED doesn't converge in time.
   **In practice, exact GED is exponential and never converges within any sane
   timeout for graphs this size (13–23 nodes for real invoices)** — so
   `graph_similarity` skips attempting it for any graph pair above
   `GED_MAX_NODES = 10` nodes and goes straight to the fallback. Every real score
   you see today comes from the structural feature vector, not GED. (See §7 for why
   this matters and what was tried.)
2. **`embedding_similarity(embedding_a, embedding_b) -> float`** — plain cosine
   similarity between the two ResNet embeddings, remapped from `[-1, 1]` to `[0, 1]`.
3. **`text_similarity(image_path_a, image_path_b) -> float`** — runs `easyocr` on
   the **original** (not skeleton) images, then compares the extracted text with
   `difflib.SequenceMatcher`. This is the **naive baseline** — what a fraud
   detector that only reads text would do. OCR output is cached to disk (slowest
   step). Images are downscaled 3x before OCR purely for stability on this machine
   (see §7).
4. **`combined_fraud_score(graph_sim, embedding_sim, w1=0.5, w2=0.5) -> float`** —
   the actual structural fraud score: a 50/50 weighted average of graph similarity
   and embedding similarity. **Text similarity is deliberately excluded** — it's
   computed separately in `evaluate.py` purely for comparison, never folded into
   the real score. This is the central methodological point of the whole project.

### `evaluate.py` — Stage 5: run everything, produce the paper table

Reads `data/labels.csv`, and for every one of the 200 pairs: extracts layout,
builds graphs, gets embeddings, computes all four scores, times each pair. Then:

- Thresholds `combined_fraud_score` (default 0.6) into a fraud/clean prediction,
  does the same for `text_sim` with its own threshold.
- Prints precision/recall/F1 and a confusion matrix for **both** methods.
- Prints a **per-`pair_type` breakdown** — this is the important part, since it
  shows F1 separately for each of the four pair types, especially the hard
  negative.
- Computes ROC-AUC and PR-AUC for both methods and saves a two-panel ROC/PR curve
  plot to `output/roc_pr_curves.png`.
- Writes `output/evaluation_results.csv` (one row per pair — every score, every
  elapsed time) and `output/evaluation_summary.csv` (the aggregate table, including
  a `structural_<pair_type>` / `text_baseline_<pair_type>` row for each pair type).

These two CSVs plus the ROC plot are the actual deliverable of Phase 1 — the
"comparison table" the spec keeps calling the primary output.

### `demo.py` — Stage 6: the presentation tool

```bash
python demo.py <img_a> <img_b> [--save output/figure.png] [--ground-truth fraud|clean] [--pair-type ...]
```

Runs the exact same four-score pipeline as `evaluate.py`, but on just **one pair**,
and produces:

- A printed score block (text similarity + naive verdict, graph similarity,
  embedding similarity, combined score + our verdict, ground truth if supplied).
- A matplotlib 2×3 figure: original A / skeleton A / graph A on the top row,
  original B / skeleton B / graph B on the bottom row, with all four scores and
  both verdicts in the figure's title.

This is the file you'll use for the live demo in §6.

---

## 4. What Actually Happens to a Single Pair of Invoices

Concretely, tracing `invoice_A.png` and `invoice_B.png` through the pipeline:

1. `extract_layout(invoice_A.png)` → 15–25 bounding boxes (tables, text blocks,
   titles, logo), plus `output/skeletons/invoice_A_skeleton.png` (text painted
   white).
2. `build_spatial_graph(...)` → a graph with one node per box, ~3–4x as many edges
   (kNN + alignment), all coordinates normalized 0–1.
3. `get_skeleton_embedding(invoice_A_skeleton.png)` → a 512-number vector
   summarizing the *visual shape* of what's left after the text is erased.
4. Repeat 1–3 for `invoice_B.png`.
5. `graph_similarity(G_a, G_b)` → cosine similarity of structural feature vectors
   (node-type mix, density, alignment ratio, aspect ratio) → a number in `[0, 1]`.
6. `embedding_similarity(emb_a, emb_b)` → cosine similarity of the two 512-d
   vectors → a number in `[0, 1]`.
7. `combined_fraud_score = 0.5 * graph_sim + 0.5 * embedding_sim` → **this is the
   fraud score.** ≥ 0.6 → flagged as fraud (same template reused).
8. Separately, `text_similarity(invoice_A.png, invoice_B.png)` → OCR both, compare
   the words → the naive baseline's score, computed the same way but never mixed
   into step 7.

The bet the whole project is testing: for a `same_template_diff_content` pair,
step 7's score should be **high** (same skeleton) while step 8's score is **low**
(different patient name, amount, date) — proving structural comparison catches
what text comparison misses.

---

## 5. The Dataset — the Four Pair Types

`data/labels.csv` has 200 pairs, 50 of each type:

| `pair_type` | `is_fraud` | How it's built | What it's testing |
|---|---|---|---|
| `same_template_diff_content` | 1 | Same template, same logo, fully regenerated content | Can structure catch reuse when the text is completely different? |
| `same_template_diff_logo` | 1 | Same template, **different logo + different hospital name**, different content | Can structure catch reuse even when the branding is swapped too? |
| `different_template` | 0 | Two different templates, different content | Does the system correctly *not* flag genuinely unrelated invoices? |
| `same_issuer_updated_template` | 0 | **Same hospital identity**, but the template is mutated (logo moved, a section reordered, a column added/removed, font size shifted) | **The hard negative.** Does the system avoid the trap of just matching on hospital name/header text, and correctly say "clean" even though the issuer looks the same? |

That fourth row is the one to point at in a demo — it's specifically designed to
break a lazy solution (anything that just fuzzy-matches header text would call this
fraud; the structural method shouldn't).

---

## 6. Running a Live Demonstration

This is the recommended script if you're showing this to someone (a professor, a
teammate, a reviewer) who hasn't seen the project before.

### 6.1 One-time setup (do this before the audience arrives)

```bash
pip install -r requirements.txt
playwright install chromium        # preferred renderer; ReportLab fallback works without it
python generator/generate_dataset.py --n-invoices 400 --n-pairs 200 --seed 42 --noise light
```

This takes a few minutes and only needs to be done once (it's reproducible — same
seed always gives the same 400 invoices). Confirm it worked:

```bash
ls data/invoices | wc -l      # should print 400
head data/labels.csv          # should show doc_a,doc_b,pair_type,is_fraud rows
```

### 6.2 The live part — walk through `demo.py` on three carefully chosen pairs

Pick one pair of each flavour from `data/labels.csv` ahead of time (or just `grep`
for a `pair_type` and take the first match):

```bash
grep same_template_diff_content data/labels.csv | head -1
grep same_issuer_updated_template data/labels.csv | head -1
```

**Pair 1 — show the fraud case (`same_template_diff_content`):**

```bash
python demo.py invoice_XXXX.png invoice_YYYY.png --ground-truth fraud --pair-type same_template_diff_content --save output/demo_fraud.png
```

Open `output/demo_fraud.png` and narrate: *"Look at the two original invoices —
different patient, different amounts, different dates. A text-based checker would
see almost no overlap. But look at the skeleton and graph panels — nearly
identical. That's why the combined structural score is high and the text
similarity is low — this is exactly the fraud pattern we're built to catch."*

**Pair 2 — show the hard negative (`same_issuer_updated_template`):**

```bash
python demo.py invoice_XXXX.png invoice_YYYY.png --ground-truth clean --pair-type same_issuer_updated_template --save output/demo_hardneg.png
```

Narrate: *"Same hospital name in both invoices — a naive system might get
suspicious here. But the template was legitimately updated, so the layout is
different. Our structural score stays low and correctly says clean. This is the
case that proves we're not just matching on the hospital's name."*

**Pair 3 (optional) — show a genuinely unrelated pair (`different_template`)** for
contrast, and/or `same_template_diff_logo` if you want to show the system also
survives a swapped logo.

### 6.3 Then zoom out to the full evaluation

```bash
python evaluate.py
```

While it runs (it iterates all 200 pairs — expect it to take a while, see §7 for
why), explain that this is the same four-score pipeline from the demo, just run
200 times and scored against ground truth. When it finishes, show:

- The printed **per-pair-type F1 breakdown** in the terminal — point at the
  `same_issuer_updated_template` row specifically.
- `output/roc_pr_curves.png` — structural vs. text-baseline ROC/PR curves
  side-by-side.
- `output/evaluation_summary.csv` — the actual paper-ready numbers.

The headline claim to make (per the spec's acceptance criterion 5): **structural
F1 beats text-baseline F1 on the two fraud pair types.**

### 6.4 Fallback if you don't have time to run the full 200-pair evaluation live

`evaluate.py` can take a while end-to-end (OCR and layout extraction on 400
images). If you're demoing live and short on time, run `evaluate.py` once
*before* the audience arrives, keep the terminal output, and just show
`output/evaluation_summary.csv` / `output/roc_pr_curves.png` plus one or two
`demo.py` pairs run live for the interactive part. That's the best of both:
a live "watch it think" moment on individual pairs, plus pre-computed evidence at
scale.

---

## 7. Known Quirks (useful if someone asks "why does it do that?")

- **`layoutparser` fails to import** in this environment (its Detectron2 model
  wrapper isn't available), so **every run uses the OpenCV fallback** for layout
  extraction. This is by design — the spec requires the OpenCV path to carry the
  whole pipeline alone — but if someone asks "why don't I see a PubLayNet model
  load," that's why.
- **Graph edit distance (GED) never actually runs on real invoices.** Exact GED is
  exponential in graph size, and real invoice graphs (13–23 nodes) never converge
  within any timeout worth waiting for. `graph_similarity()` now skips attempting
  GED above 10 nodes and always uses the structural-feature-vector fallback for
  this dataset. The "primary/fallback" split in the code is real and tested, but
  for this dataset's scale, the fallback is what's actually driving every reported
  structural score. This is documented here rather than hidden, because a reviewer
  who reads `similarity.py` closely would otherwise wonder why GED never shows up
  in the debug logs.
- **OCR images are downscaled 3x before `easyocr.readtext()`.** Full-resolution
  invoice pages (~1600×2250px) triggered a native crash in `easyocr`'s quantized
  recognition network on this machine; downscaling avoids it entirely and is a
  reasonable trade since `text_similarity` is only the comparison baseline, not
  part of the actual fraud score.
- **`evaluate.py` is slow** (tens of minutes for all 200 pairs) mainly because of
  OCR (the slowest step, hence it's cached) and the per-pair structural-fallback
  computation. Both embeddings and OCR are cached to disk by file hash, so a second
  run over the same dataset is much faster than the first.

---

## 8. Command Cheat Sheet

```bash
# Setup
pip install -r requirements.txt
playwright install chromium

# Stage 0 — generate the dataset (do this first)
python generator/generate_dataset.py --n-invoices 400 --n-pairs 200 --seed 42 --noise light

# Stage 5 — run the full evaluation (the primary deliverable)
python evaluate.py

# Stage 6 — demo a single pair
python demo.py <img_a> <img_b>
python demo.py <img_a> <img_b> --save output/figure.png --ground-truth fraud --pair-type same_template_diff_content
```

## 9. Acceptance Criteria (from `PHASE1_SPEC.md`)

| # | Criterion | How to verify |
|---|---|---|
| 1 | `generate_dataset.py --seed 42` → 400 invoices + both CSVs | `ls data/invoices \| wc -l` → 400; `data/labels.csv` and `data/manifest.csv` exist |
| 2 | Re-running with the same seed → byte-identical output | Run twice, diff the output directories |
| 3 | `evaluate.py` → both output CSVs + metric printout | `output/evaluation_results.csv`, `output/evaluation_summary.csv` exist, terminal shows both metric tables |
| 4 | `demo.py <a> <b>` → 2×3 figure + score block | Run it, check `output/demo_output.png` (or `--save` path) and the printed score block |
| 5 | Structural F1 beats text-baseline F1 on the two fraud pair types | Compare `structural_same_template_diff_content` / `structural_same_template_diff_logo` rows against their `text_baseline_*` counterparts in `output/evaluation_summary.csv` |
