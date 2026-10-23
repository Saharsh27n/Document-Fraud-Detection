---
name: project-overview
description: "Phase 1 spec for structural fraud detection on medical invoices — pipeline stages, renderer fallbacks, and key design constraints"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1a138c91-77e8-42cb-b00d-036226b2df65
---

This project builds a 6-stage pipeline detecting fraudulent medical invoice template reuse via structural/layout analysis. The full spec is in `PHASE1_SPEC.md`.

**Core hypothesis:** structural similarity (graph + CNN embedding) beats text similarity for template fraud because fraudsters swap content but reuse the same visual layout.

**Build order:** Stage 0 (dataset generator) first — verify 3 sample renders before touching pipeline stages.

**Key constraints:**
- Every stage must degrade gracefully (Playwright → ReportLab fallback; layoutparser → OpenCV fallback)
- Everything reproducible from a single `--seed` integer
- Text similarity is a baseline only — NOT a component of `combined_fraud_score`
- Hard negative class `same_issuer_updated_template` is the critical test

**Why:** Academic paper deliverable. Primary metric is structural F1 vs. text-baseline F1 on fraud pair types. The comparison table is what gets submitted.

**How to apply:** When implementing any stage, check that the fallback path works independently before adding the primary path. When evaluating results, focus on the per-`pair_type` breakdown, especially `same_issuer_updated_template`.
