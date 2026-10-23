---
name: ged-fallback-bug
description: similarity.py graph_similarity() bug fixed 2026-09-16 — always returned 0.0; also hit a Windows multiprocessing deadlock and an easyocr segfault along the way
metadata: 
  node_type: memory
  type: project
  originSessionId: 1c06bc75-dd19-413a-a541-aa3e87723837
  modified: 2026-09-16T07:40:49.970Z
---

Three separate bugs surfaced while getting `evaluate.py` to run cleanly on 2026-09-16, all in `similarity.py`:

**1. graph_similarity() always returned 0.0.** It called `next(gen)` on `nx.optimize_graph_edit_distance(...)`, taking only the generator's *first* (loosest, trivial delete-all+insert-all) upper bound. Every pair collapsed to `sim=0.0`, which produced the all-zero `graph_sim` column and F1=0.0 in the very first smoke test.

**2. GED never converges on this dataset's graph sizes.** Real invoice layout graphs here have 13–23 nodes / 29–76 edges. `optimize_graph_edit_distance` is exponential and empirically never produces a better bound than the trivial one within any sane timeout for graphs this size — it always times out. So the fallback (structural feature vector cosine similarity) is what actually drives every reported "structural" score, not GED. Fixed by capping GED attempts to graphs with ≤`GED_MAX_NODES` (10) nodes — above that, skip straight to the fallback rather than waiting out a timeout that never pays off.

**3. Two nasty platform-specific traps hit while trying to bound the GED search safely — avoid repeating them:**
- A `daemon=True` thread running the GED search cannot be killed in Python. Leaving it running past the timeout (to "let it keep improving in the background") causes threads to pile up across the 200 pairs, each burning CPU/stack on an intractable search, until the interpreter segfaults from resource exhaustion. Lesson: an abandoned daemon thread is not free — it should only be used when the work is bounded (see fix #2, only attempted for small graphs that converge fast).
- Tried isolating GED (and separately, OCR) into a `multiprocessing.get_context("spawn").Process()` so it could be hard-terminated. This works fine in isolation, but **hangs forever (0% CPU, no child ever appears) the moment both `pandas` and `torch` have been imported in the parent process** before the `.start()` call — a DLL-loader-lock deadlock specific to this Windows environment. `evaluate.py` imports both, so every real run hung on the very first pair. Confirmed via isolated repro scripts in scratchpad. Lesson: don't reach for subprocess-based timeouts in this environment once pandas+torch are both loaded — use size/complexity gating instead to keep expensive work off the table entirely.

**Separately: easyocr crashes (access violation) when `readtext()` runs on the full-resolution invoice pages (~1600x2250).** Happens even with no other torch model in the process — it's specific to the image size / number of detected text regions hitting the ONNX/oneDNN quantized recognition network. `OMP_NUM_THREADS=1` avoids it but makes OCR ~100s/image (impractical for 200+ pairs). The actual fix: downscale the image by 3x (PIL resize) before calling `reader.readtext()` in `_ocr_text()` — avoids the crash entirely and runs in ~5–10s/image. Fine tradeoff since `text_similarity` is only a comparison baseline (see [[project-overview]]), not part of `combined_fraud_score`.

**How to apply:** if `evaluate.py` hangs with ~0% CPU and no visible child process, suspect the pandas+torch+spawn deadlock before anything else. If it segfaults, check whether the crash is in OCR (`readtext` on a full-res image) vs. something else — don't assume model co-location is the cause without reproducing in isolation first, it can be a red herring (it was, here).
