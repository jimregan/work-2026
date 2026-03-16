Love it — you’ve converged. Here’s a **concrete, completion-oriented TODO list**, ordered to minimise thrash and keep options open for the **Derived Text Formats workshop** *first*, while not foreclosing **SIGUL / SPEAKABLE** later.

I’ll keep this **operational**, not aspirational.

---

# ✅ MASTER TODO LIST (Derived Text Formats Workshop–first)

## Phase 0 — Lock scope (do this once, then stop re-deciding)

* [ ] **Primary target**: *Leveraging Derived Text Formats to Unlock Copyrighted Collections for Open Science*
* [ ] **Contribution focus**: *methods complete, dataset illustrative*
* [ ] **Languages**: Irish only (others as motivation/examples, not data)
* [ ] **Modalities**: HTML + image/OCR (PDF optional, only if cheap)

---

## Phase 1 — Representation & schema (highest leverage)

**Goal:** make the method unambiguous before touching data volume.

* [ ] Write a **1-page schema spec**:

  * source identifiers
  * landing URL vs direct URL
  * checksum strategy
  * modality enum: `html | ocr_image | pdf`
  * selector / bounding box representation
  * offset conventions (relative, not absolute)
* [ ] Decide **exact CoNLL-U fields**:

  * what lives in `MISC`
  * how audio start/end are encoded
  * how dialect/normalisation metadata is represented
* [ ] Version the schema (`schema_version = 0.1`)

➡️ *Stop here and sanity-check: could someone else implement this from the spec alone?*

---

## Phase 2 — Normalisation & dialect layer (keep it boring and pinned)

**Goal:** language-aware, minimal, defensible.

* [ ] Define **one language-specific normalisation profile**:

  * name it (e.g. `ga_IE_norm_v1`)
  * write the **trivial ICU transform**
  * document exactly what it does and does *not* do
* [ ] Decide how dialectal divergence is encoded:

  * rule IDs from the official lexicon tools
  * no surface forms
  * reproducible derivations only
* [ ] Write a **½-page motivation paragraph**:

  * framed as data quality + offset stability
  * no sociolinguistic debate

---

## Phase 3 — Pipeline scripts (minimal but end-to-end)

**Goal:** show that the method actually runs.

* [ ] Script: **fetch & verify**

  * download content (where lawful)
  * validate checksum
* [ ] Script: **extract text span**

  * HTML: CSS selector → text
  * OCR image: bounding box → text
  * (PDF if included)
* [ ] Script: **attach audio spans**

  * start/end times
  * reference only, no bundled audio
* [ ] Script: **emit CoNLL-U**

  * offsets, features, MISC fields
* [ ] Script: **validate derived file**

  * re-resolve anchors
  * check offsets
  * regenerate dialect annotations

---

## Phase 4 — Dockerisation (this is your credibility multiplier)

**Goal:** freeze the method in time.

* [ ] Write a `Dockerfile`

  * pin base image
  * pin ICU version
  * pin OCR engine version
  * pin Python deps
* [ ] Add:

  * `--version` output
  * embedded schema + sample config
* [ ] Define two entrypoints:

  * `validate`
  * (optional) `build`
* [ ] Build and tag image with date + commit hash

---

## Phase 5 — Sample dataset (stop early on purpose)

**Goal:** illustrate, not exhaust.

* [ ] Select:

  * 1 HTML source
  * 1 OCR/image source
  * (optional) 1 PDF
* [ ] For each:

  * ~10–30 aligned segments
  * at least a few dialectal cases
  * at least one case where normalisation matters
* [ ] Run pipeline → produce derived CoNLL-U files
* [ ] Run validator on your own output (important!)

---

## Phase 6 — Paper writing (don’t overthink this)

**Goal:** document what already exists.

* [ ] Introduction:

  * problem: access under copyright
  * why derived formats matter
* [ ] Method:

  * representation
  * modality handling
  * normalisation + dialect layer
* [ ] Implementation:

  * scripts
  * Docker image
* [ ] Dataset:

  * scope
  * limitations (explicitly incomplete)
* [ ] Discussion:

  * why this generalises
  * why audio matters here
* [ ] Reproducibility statement:

  * lawful access
  * containerised pipeline

---

## Phase 7 — Optional forward hooks (don’t implement now)

**Just mention, don’t build:**

* [ ] Scaling to other languages
* [ ] Reuse for ASR evaluation (SPEAKABLE)
* [ ] Language-specific dataset extensions (SIGUL)

---

# 🚦Completion criteria (important)

You are **done** when:

* the schema is frozen
* the Docker image runs
* the validator passes on the sample dataset
* the paper describes exactly that, no more

Not when:

* the dataset feels “complete”
* you’ve anticipated every reviewer
* you’ve solved Irish ASR

---

If you want, next I can:

* turn this into a **2–3 week timeboxed plan**, or
* help you write the **schema spec** (arguably the hardest and most valuable single document).

But as a TODO list?
This is solid, finite, and finishable.
