# SIH1441 — Complete Discussion Record & Master Plan

**Project:** Structure-Invariant Document Fingerprinting for Fraudulent Medical Insurance Template Reuse Detection
**Origin:** SIH1441 — Similar Document Template Matching (Bajaj Finserv Health Ltd.)
**Purpose of this file:** One place holding everything we discussed, every decision made and why, and the entire plan to implement the project and write the paper.

---

## Table of Contents
1. How we got here (discussion timeline)
2. Project pool analysis and shortlist
3. The two final picks and the final selection
4. Submission form details
5. Original implementation approach
6. Research paper angles considered
7. Presentation outline (for the faculty guide)
8. The 1-hour proof-of-concept (disposable)
9. The full-scale plan (the real one)
10. Phase 1 MVP: the opencode prompt
11. Evaluation design
12. Risks, open items, things to verify
13. Immediate next steps

---

## 1. How We Got Here (Discussion Timeline)

| Step | What was asked | What was decided |
|---|---|---|
| 1 | Review `Project_Pool.xlsx` of department-assigned project topics; find ones that show fundamental understanding, are unique or problem-solving, span different domains, plus self-proposed ideas | Shortlist across domains + 5 self-proposed ideas (Section 2) |
| 2 | One pick based on the person, one pick on pure discretion | Personal pick: SIH1285 (Legal Documentation Assistant). Discretion pick: SIH1441 (Template Matching for Insurance Fraud) |
| 3 | How to implement SIH1441 and what research paper can be written | 4-stage pipeline + three paper options; recommended Option 2 (novel method) with Option 1 (benchmark) as its evaluation section |
| 4 | Fill in the department submission fields | Filled as SP (self-proposed extension), Research Oriented = YES, Category = SW |
| 5 | Consolidate all details of the final selection | Consolidated profile (Section 4) |
| 6 | Presentation outline for the faculty guide (cold intro, 10 minutes) | 10-slide outline (Section 7) |
| 7 | Faculty wants to see work done so far; nothing exists yet | Scoped-down MVP concept |
| 8 | Build it in about 1 hour using opencode | Structured prompt; later revised to drop `pytesseract` (system binary risk) in favour of hardcoded text |
| 9 | Time constraint removed; wants a full-scale plan, with the MVP being a smaller version of the real system | Phased 14-week plan (Section 9) |
| 10 | Prompt for opencode to build the first functional version | Phase 1 prompt (Section 10) |
| 11 | This file | Everything consolidated |

---

## 2. Project Pool Analysis and Shortlist

*Note: IDs and titles below are as stated during the discussion. Re-check them against the spreadsheet before submitting anything (see Section 12).*

### Best for showing fundamental CS concepts
- **SIH1454: AI/ML Phishing Domain Detector** (Blockchain & Cybersecurity). Covers web page similarity, image comparison, NLP and classification in one problem. String distance metrics, DOM parsing, image feature extraction and probability scoring all show up.
- **SIH1451: AI/ML Tool for Non-IoC Based System Compromise Detection** (Cybersecurity). Behavioural anomaly detection from first principles: baselines, statistical deviation, defining "normal".

### Unique / genuinely solving a real problem
- **SIH1441: Similar Document Template Matching for Insurance Fraud** (Bajaj Finserv). The most industry-realistic problem in the pool. Fraudsters reuse identical medical invoice templates with swapped names. Needs layout-aware comparison, not plain text diff.
- **SBOM Generation Tool** (Smart Automation; discussed as SIH1449, once mislabelled SIH1129). Auto-identify libraries, versions and dependencies in custom software. Supply-chain security relevance.
- **SIH1450: Offline LLM Tool for Summarization** (Smart Automation, NTRO). Locally deployed LLM pipeline for air-gapped environments: model selection, quantization, prompt engineering.

### Picks across domains

| Domain | ID | Title | Why it is interesting |
|---|---|---|---|
| HealthTech | SIH1344 | AI tool for dermatological diagnosis from images | Image classification for real medical use |
| Smart Automation | SIH1516 | Multilingual AI grievance chatbot for urban citizens | NLP + multilingual ASR + routing logic |
| Education | SIH1460 | Adaptive MCQ system (IRT + AI) for AICTE's PARAKH | Adaptive testing + item response theory |
| Legal / Justice | SIH1285 | AI-powered Legal Documentation Assistant | RAG over legal text + plain-language generation |
| Space | SIH1518 | Change detection of man-made objects from satellite images | Sentinel-2 + computer vision |
| Disaster Mgmt | SIH1453 | AI de-smoking / de-hazing for indoor fire rescue video | Image restoration, safety-critical |

### Self-proposed ideas (outside the pool)
1. Court case outcome predictor using NLP on Indian legal judgments
2. RAG-based local study assistant for engineering syllabi (LangChain + small LLM + FAISS)
3. Resume-to-JD skill gap analyzer with personalised learning path generator
4. Privacy-preserving attendance + engagement monitor using facial landmarks / pose estimation
5. Fake review detection on Indian e-commerce platforms (linguistic + behavioural features)

---

## 3. The Two Final Picks and the Final Selection

**Pick based on you: SIH1285, AI-Powered Legal Documentation Assistant.** Reason: it follows the same pattern as your earlier projects (equity research platform, government scheme discovery tool): make complex domain knowledge accessible through AI, using FastAPI/Django + LangChain + RAG + ChromaDB.

**Pick on pure discretion: SIH1441, Similar Document Template Matching.** Reason: every other problem is a dashboard, chatbot, or "AI on a government process". This one comes from industry, has no clean off-the-shelf solution, needs real thinking about layout fingerprinting and template invariance, and sounds novel in an interview.

**Final selection: SIH1441.** Chosen because the stated main objective is a research paper, and this problem has a genuine research gap (see Section 6).

---

## 4. Submission Form Details

| Field | Details |
|---|---|
| SIH ID | SIH1441 |
| Problem ID / Govt Agency | Bajaj Finserv Health Ltd. |
| SP / SPP | SP (self-proposed extension: the original problem is Bajaj's, the research framing and graph-based pipeline are ours) |
| Project Title | Structure-Invariant Document Fingerprinting for Fraudulent Medical Insurance Template Reuse Detection |
| Technology / Domain | Python, OpenCV, PyMuPDF, LayoutParser, PyTorch, GNN, FastAPI, React. Domain: AI/ML + Document Intelligence + Fraud Detection |
| Description | Multi-stage pipeline that detects fraudulent reuse of medical invoice/prescription templates by analysing structural layout rather than text content. Converts documents to spatial graphs of element positions, encodes skeletons via CNN/ViT embeddings, scores similarity using graph-based comparison combined with embedding similarity, and flags fraud even when logos, colours and text are swapped. |
| Research Oriented | YES |
| Category | SW |

### Original problem statement (from the sheet)
Bajaj Finserv receives large volumes of medical invoices, prescriptions and lab reports. Fraudulent customers exploit identical templates, swapping provider names, logos, colours and text positioning, to file duplicate reimbursement claims that pass standard document checks.

---

## 5. Original Implementation Approach (Four Stages)

**Stage 1: Ingestion and preprocessing.** Accept PDF/image invoices, prescriptions, lab reports. Convert with `pdf2image` or `PyMuPDF`. Grayscale, denoise, deskew with OpenCV. Noisy input ruins everything downstream.

**Stage 2: Template feature extraction (the research core).** Three parallel views:
- *Structural/layout:* detect bounding boxes of text blocks, tables, logos (layoutparser / detectron2); represent the document as a spatial graph of elements.
- *Visual:* strip text, keep only the skeleton (lines, boxes, borders), encode with a CNN or ViT into an embedding.
- *Perceptual hashing:* pHash/dHash on the skeleton as a fast first-pass filter (kept as a baseline in the final plan).

**Stage 3: Similarity scoring.** Cosine similarity on embeddings + graph comparison on layout graphs, combined into a weighted fraud score. Starting thresholds: below 0.4 clean, 0.4 to 0.75 amber, above 0.75 red. (These are initial guesses; tune on validation data.)

**Stage 4: Dashboard.** Upload interface, side-by-side viewer, colour-coded flags with explanation of which structural elements matched, exportable report.

**Stack:** Python, PyMuPDF/pdf2image, OpenCV, layoutparser, PyTorch, scikit-learn, FastAPI, React.

---

## 6. Research Paper Angles Considered

### Option 1: Benchmark paper (most achievable)
*"Layout-Aware Template Similarity Detection in Medical Insurance Documents: A Multi-Modal Approach."* Build a synthetic dataset with controlled variations; benchmark OCR text similarity, pHash, layout graph matching and CNN embeddings. Contribution: the benchmark plus the finding that structural features beat text similarity. Level: undergraduate-friendly conferences (ICAECT, ICCCNT, IEEE ICIT).

### Option 2: Novel method paper (stronger) — **chosen**
*"Structure-Invariant Document Fingerprinting for Fraudulent Template Reuse Detection."* Pipeline: text removal, spatial graph construction, GNN-based similarity. Novelty: framing template fraud as a graph matching problem, rather than text similarity (which existing methods inherit from plagiarism detection). Candidate venues: Pattern Recognition Letters, Expert Systems with Applications, IEEE Access.

### Option 3: Dataset + method (most ambitious)
*"MedFraudDoc: A Synthetic Benchmark Dataset for Medical Invoice Template Fraud Detection."* 500 to 1000 programmatically generated labelled documents, released publicly, with a baseline method. Venue ideas listed in the discussion were Scientific Data / ACM MM / AAAI workshops; treat those as aspirational and verify fit before targeting.

### Decision
Target **Option 2**, with **Option 1** as the internal evaluation section and **Option 3's dataset** as a secondary contribution (MedFraudDoc is built anyway in Phase 2). Nothing built is wasted.

### Core claim the paper must prove
> Structural similarity catches template-reuse fraud that text similarity misses, and learned graph/embedding methods outperform simple structural hashing.

---

## 7. Presentation Outline (Faculty Guide, 10 Minutes, Cold Intro)

| # | Slide | Content |
|---|---|---|
| 1 | Title | Project title, names, SIH1441, Bajaj Finserv Health, department |
| 2 | The problem | Template-reuse fraud in medical claims; two near-identical invoices side by side; hook line: "Standard systems compare text. Fraudsters know this." |
| 3 | Why existing solutions fail | OCR + text similarity fooled by text swaps; manual review does not scale; watermark/signature checks bypassable; gap = nothing compares structure independent of content |
| 4 | Core idea | Fingerprint the layout skeleton, not the content. Visual: document, text stripped, skeleton, fingerprint |
| 5 | Architecture | Pipeline flow diagram |
| 6 | Technical deep dive | Layout parsing, graph construction, similarity scoring |
| 7 | Dashboard | Wireframe of the upload / compare / flag / report flow |
| 8 | Research angle | Novelty (graph matching framing), contributions (method + MedFraudDoc), paper title, target venues |
| 9 | Timeline and deliverables | 14-week phased plan |
| 10 | Why us + ask | Relevant past work; what we need from the guide (paper guidance, domain feedback, periodic reviews) |

**Tips:** lead visually with the two-invoice comparison; spend extra time on slide 8; name tools, do not justify each one; keep a backup flow diagram for questions.

**Caution:** the outline originally quoted a "₹45,000 Cr+" insurance fraud figure. That number was not sourced. Do not present it without a citable source, or drop it.

---

## 8. The 1-Hour Proof-of-Concept (Disposable)

Created because the faculty asked for work shown and nothing existed.

- **Scope:** OpenCV skeleton extraction (mask text regions) + pHash on the skeleton + text similarity using hardcoded known strings (no OCR) + matplotlib side-by-side + summary table.
- **OCR decision:** `pytesseract` needs a system binary, a likely time sink. Options weighed: `easyocr` (pip-only), hardcoded text for the demo (chosen for speed), hosted OCR APIs (overkill). Final 1-hour prompt used hardcoded text, with easyocr as a stretch.
- **The one demo that matters:** a same-template, different-content pair where text similarity is LOW (naive system says "not fraud") but structural similarity is HIGH (our system flags it).
- **Status:** explicitly disposable. It proves the hypothesis is plausible; it is not part of the final pipeline.
- **How to frame it to the guide:** "We validated the idea quickly; here is the real architecture we are building toward."

---

## 9. The Full-Scale Plan (The Real One)

### Principle
The MVP is a smaller, real instance of the final architecture. Components get scaled up, not rebuilt.

| | MVP (Phase 1) | Final system |
|---|---|---|
| Dataset | 20 to 30 invoice pairs | 500 to 1000 synthetic labelled pairs (MedFraudDoc) |
| Layout detection | layoutparser pretrained + OpenCV fallback | Trained / fine-tuned detector |
| Structural encoding | Pretrained ResNet-18 embedding of skeleton | Fine-tuned CNN/ViT with contrastive or triplet loss |
| Similarity | Approximate graph edit distance + cosine similarity | Trained GNN similarity + embedding fusion |
| Output | Scripts + visuals + evaluation table | FastAPI + React dashboard |
| Evaluation | Small labelled set, P/R/F1 | Held-out test set, P/R/F1, ROC-AUC, ablations |

### Phase 0: Problem formalisation (Week 1)
Define "same template" formally (identical structural layout regardless of text, logo colour, minor in-tolerance shifts). Define four pair types:
- **Type A (fraud):** same template, different patient / amount / date
- **Type B (fraud, harder):** same template, swapped logo/letterhead, different text
- **Type C (clean):** genuinely different templates and issuers
- **Type D (edge case):** same issuer, legitimately updated template (should NOT be flagged; hardest negative)

Deliverable: one-page problem statement + evaluation criteria (seeds the SRS and the paper's Introduction).

### Phase 1: MVP, real pipeline at small scale (Weeks 2 to 4)
1. **Dataset (manual):** 20 to 30 images across Types A to D; ground truth in `labels.csv`.
2. **Layout extraction:** layoutparser pretrained (PubLayNet-style). OpenCV fallback (Hough lines / morphology) for table borders, which pretrained academic-paper models often miss.
3. **Skeleton + spatial graph:** nodes = elements (type, bbox, centroid); edges = spatial relations (alignment or k-nearest-neighbour by centroid, k=3), weight = normalised distance. Built with `networkx`.
4. **Embedding:** pretrained ResNet-18, penultimate layer, no fine-tuning yet.
5. **Similarity:** approximate graph edit distance (or structural-statistics heuristic if too slow) + cosine similarity; `score = w1 * graph_sim + w2 * embedding_sim`, weights tuned by hand on the small set.
6. **Baseline:** text similarity via easyocr + SequenceMatcher/TF-IDF. Needed for the demo and the paper.
7. **Output:** skeleton and graph visualisations, comparison table (naive verdict vs structural verdict), precision/recall/F1.

### Phase 2: Dataset scale-up, MedFraudDoc (Weeks 5 to 7)
Generate synthetic invoices with `reportlab` + `Faker`; apply controlled variations for Types A to D; target 500 to 1000 labelled pairs; define train/val/test splits; write a dataset card; plan public release alongside the paper.

### Phase 3: Model upgrade (Weeks 8 to 10)
- Lightweight GNN (e.g. GCN via PyTorch Geometric) trained on spatial graphs for similarity/classification.
- Optionally fine-tune the CNN encoder with triplet/contrastive loss (same-template pulled together, different pushed apart).
- Full evaluation on the held-out test split: precision, recall, F1, ROC-AUC against baselines. This produces the paper's main results.

### Phase 4: Dashboard and integration (Weeks 11 to 12)
FastAPI backend wrapping the trained pipeline. React frontend: upload, side-by-side viewer, green/amber/red flags, which structural elements matched, exportable report.

### Phase 5: Research paper (Weeks 10 to 14, parallel with 3 and 4)
1. Abstract and Introduction
2. Related Work (document similarity/plagiarism detection, layout analysis such as PubLayNet and LayoutLM, insurance fraud detection)
3. Dataset: MedFraudDoc (generation method, statistics, why synthetic)
4. Method (spatial graph, GNN similarity, fusion with visual embeddings)
5. Experiments (baselines: text similarity, pHash, plain CNN; ablations: graph-only, embedding-only, combined)
6. Results and Discussion (where it wins, where it fails; Type D is an honest limitation)
7. Conclusion and Future Work

Targets: IEEE Access / Expert Systems with Applications (journal); IEEE ICIT / ICCCNT / ICAECT (conference, faster turnaround).

### Tech stack (full)

| Layer | Tools |
|---|---|
| Preprocessing | OpenCV, PyMuPDF / pdf2image |
| Layout detection | layoutparser (Detectron2 or PaddleDetection backend) |
| Graph construction | networkx |
| Embeddings | PyTorch, ResNet (pretrained, then fine-tuned) |
| GNN | PyTorch Geometric |
| Dataset generation | reportlab, Faker |
| OCR baseline | easyocr |
| Backend / Frontend | FastAPI / React |
| Evaluation | scikit-learn, matplotlib / seaborn, pandas |

### Timeline

| Weeks | Phase | Output |
|---|---|---|
| 1 | Problem formalisation | Problem doc |
| 2 to 4 | MVP | Small-scale working pipeline + baseline comparison |
| 5 to 7 | Dataset scale-up | MedFraudDoc |
| 8 to 10 | Model upgrade | Trained model + full evaluation |
| 10 to 14 | Paper (parallel) | Submission-ready draft |
| 11 to 12 | Dashboard | End-to-end system |

---

## 10. Phase 1 MVP: The opencode Prompt

```
Build a Python proof-of-concept pipeline for detecting fraudulent medical 
invoice template reuse using structural layout analysis. This is Phase 1 
of a larger system — build it as the real architecture at small scale, 
not a simplified placeholder. Every component here will later be scaled 
up (bigger dataset, fine-tuned models), so implement each stage properly 
even though the dataset is small.

## Project Structure
- `/data/invoices/` — sample invoice images (I will add 20-30 PNG/JPG invoices)
- `/data/labels.csv` — columns [doc_a, doc_b, pair_type, is_fraud]
  pair_type: same_template_diff_content, same_template_diff_logo, 
  different_template, same_issuer_updated_template
- `/output/skeletons/` — extracted skeleton images
- `/output/graphs/` — saved graph visualizations
- `layout_extraction.py`, `graph_builder.py`, `embedding.py`, 
  `similarity.py`, `evaluate.py`, `demo.py`, `requirements.txt`

## Step 1: layout_extraction.py
- Load invoice image (OpenCV/PIL)
- Use layoutparser with a pretrained PubLayNet-based model to detect text 
  blocks, tables, titles, figures/logos
- If it underperforms on invoice tables, add an OpenCV fallback (Hough 
  lines / morphological ops) for table borders and merge into the same 
  element list
- `extract_layout(image_path) -> list[dict]` with {type, bbox:(x1,y1,x2,y2)}
- Also produce the skeleton: original image with detected text-block 
  regions masked white, leaving lines/borders/logo shapes

## Step 2: graph_builder.py
- Build a networkx graph: node per element (type, bbox, centroid); edges 
  by spatial alignment within tolerance or k-nearest-neighbour by centroid 
  (k=3); edge weight = normalised centroid distance
- `build_spatial_graph(layout_elements) -> networkx.Graph`
- Save graph visualizations to /output/graphs/

## Step 3: embedding.py
- Pretrained ResNet-18 (torchvision, ImageNet weights, no fine-tuning), 
  classification layer removed
- `get_skeleton_embedding(skeleton_image_path) -> np.ndarray` (resize, 
  ImageNet normalisation, penultimate-layer features, flat array)

## Step 4: similarity.py
1. graph_similarity(graph_a, graph_b) -> float: networkx 
   optimize_graph_edit_distance (approximate) or a heuristic on node-type 
   distribution, node count, average degree, layout aspect ratio if GED 
   is too slow; normalise to 0-1
2. embedding_similarity(emb_a, emb_b) -> float: cosine, normalised 0-1
3. text_similarity(text_a, text_b) -> float: easyocr on the ORIGINAL 
   images, difflib.SequenceMatcher ratio. This is the naive baseline only
4. combined_fraud_score(graph_sim, embedding_sim, w1=0.5, w2=0.5) -> float
   (text similarity is NOT part of this score)

## Step 5: evaluate.py
- Loop over all pairs in labels.csv; compute text, graph, embedding and 
  combined scores
- Threshold on combined score (default 0.6, configurable) -> prediction
- Compare to ground truth; print precision, recall, F1, confusion matrix 
  for BOTH the structural method and the text baseline
- Save to /output/evaluation_results.csv

## Step 6: demo.py
- Input: two image paths. Run the full pipeline.
- matplotlib 2x3 grid: original A/B, skeleton A/B, graph A/B
- Print: text similarity (+ naive verdict), graph sim, embedding sim, 
  combined fraud score (+ our verdict)

## requirements.txt
opencv-python, layoutparser, torch, torchvision, networkx, easyocr, 
scikit-learn, matplotlib, pandas, numpy, Pillow

## Notes
- Do not substitute shortcuts like perceptual hashing; implement real 
  components even at small scale
- If layoutparser/Detectron2 install is problematic, use the PaddleDetection 
  backend; if still problematic, fall back to pure OpenCV layout detection 
  (contours + Hough lines) — the graph/embedding stages are unaffected
- Prioritise evaluate.py running end-to-end over any one component 
  being perfect
- Comment clearly; teammates will extend this in later phases

## Deliverable
`python evaluate.py` prints a precision/recall/F1 comparison (structural 
vs text baseline) across the labelled set; `python demo.py <img_a> <img_b>` 
gives a visual walkthrough for one pair.
```

---

## 11. Evaluation Design (For the Paper)

- **Metrics:** precision, recall, F1, ROC-AUC, confusion matrix. Recall matters most from a fraud-catching view, precision from a false-accusation view; report both.
- **Baselines:** (1) OCR text similarity, (2) pHash on skeleton, (3) plain pretrained CNN embedding, (4) graph edit distance alone.
- **Ablations:** graph-only vs embedding-only vs combined; with and without text masking; pretrained vs fine-tuned encoder.
- **Stress tests:** logo/colour swaps (Type B), small layout shifts, scan noise and rotation, and Type D (legitimate template update) to measure false positives.
- **Honest limitations to state:** synthetic data may not reflect real invoice diversity; no real Bajaj data access; layout models trained on academic papers may transfer imperfectly to invoices.

---

## 12. Risks, Open Items, Things to Verify

| Item | Why it matters | Action |
|---|---|---|
| Project IDs/titles from the sheet | A couple were mislabelled during the discussion (e.g. the SBOM entry) | Re-check against `Project_Pool.xlsx` before final submission |
| "₹45,000 Cr+" fraud statistic | Not sourced | Find a citable source (IRDAI, industry report) or remove |
| Venue claims | Acceptance and fit were stated confidently but not checked | Check each venue's scope, indexing and timeline before committing |
| layoutparser install | Detectron2 is fussy | PaddleDetection backend, else pure OpenCV |
| Graph edit distance speed | Exact GED is very slow | Use approximate GED or the heuristic fallback |
| Synthetic-only data | Reviewers may question realism | Include a small set of real or realistic templates as a held-out check |
| Type D false positives | Legit template updates may be flagged | Report honestly; consider tolerance tuning |
| Faculty expectations | Guide has not seen the idea | Use the Section 7 outline; frame the 1-hour demo as validation, not the product |
| Privacy/compliance | Real medical documents contain sensitive data | Use synthetic or fully anonymised documents only |

---

## 13. Immediate Next Steps

1. Run the Phase 1 opencode prompt (Section 10).
2. In parallel, build the labelled dataset: 20 to 30 images across Types A to D, with `labels.csv`.
3. Run `evaluate.py`, record the first precision/recall/F1 table (structural vs text baseline).
4. Show the guide: problem, architecture, first results, research angle (Section 7 outline).
5. Start Phase 2 (MedFraudDoc generation) once she signs off on the direction.
6. Begin paper skeleton (Introduction + Related Work) while models train.
