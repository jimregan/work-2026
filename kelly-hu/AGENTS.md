# AGENTS.md — Automate a Hungarian “Kelly-style” frequency list

## Objective
Reproduce the Swedish Kelly-list pipeline for **Hungarian**, producing a freely usable, learner-oriented frequency list of ~9,000 lemma+POS items with:
- rank (ID)
- raw frequency (RF)
- relative frequency (WPM)
- optional dispersion metric (ARF or equivalent)
- CEFR level banding (A1–C2) derived from frequency bands
- lemma (headword), POS
- optional grammar hints (language-specific)
- optional comments/examples field

Reference methodology: Swedish Kelly-list creation process (web corpus → tagging/lemmatization → ARF-based selection → proofreading → validation). See paper for details on corpus building, ARF/WPM, filtering, and validation-through-translation. :contentReference[oaicite:1]{index=1}

Deliverables must be **reproducible** (scripts + config + provenance).

---

## Background constraints (from the Swedish approach)
Implement these key principles:

1) **Modern web corpus, large**  
   Target **≥100 million tokens** from web sources to stabilize frequency statistics (Swedish used 114M). :contentReference[oaicite:2]{index=2}

2) **Seed-word driven crawling**  
   Use ~500 mid-frequency wordforms (length ≥5 chars; no digits; avoid foreign-looking forms) drawn from a “starter” corpus (Wikipedia dump is acceptable) to query a search engine / Common Crawl index. Swedish repeatedly chose 3 random seeds per query and crawled hit pages, deduplicating and cleaning. :contentReference[oaicite:3]{index=3}

3) **Annotate with POS + lemma**  
   Hungarian is morphologically rich; lemmatization quality is critical.

4) **Select by dispersion-aware frequency**  
   Swedish used ARF to down-rank domain-clustered words; then ordered final list by WPM. If ARF isn’t available, implement a close substitute (see below). :contentReference[oaicite:4]{index=4}

5) **Filter for learner-relevant POS + remove noise**  
   Exclude: punctuation, foreign words, proper names (mostly), participles, most numerals; remove items with digits/non-letter symbols. Include core POS (noun/verb/adj/adv/pron/det/conj/etc.) per feasibility. :contentReference[oaicite:5]{index=5}

6) **Proofread + validate**  
   Swedish did:
   - manual correction of top candidates (tag/lemma errors, spelling/style variants)
   - automatic cross-check vs lexicons
   - “validation through translation” by comparing with translated lists from partner languages (producing inclusion/deletion candidates). :contentReference[oaicite:6]{index=6}

For Hungarian, translation validation may be approximated using bilingual lexicons or parallel data if partner-language lists are unavailable.

---

All scripts must be runnable from project root.

---

## Step-by-step pipeline

### Step 0 — Choose tooling (agent must decide and document)
Pick one primary Hungarian NLP stack for tokenization+POS+lemmatization, prioritizing:
- robustness on noisy web text
- lemma accuracy
- licensing compatibility

Acceptable options:
- UDPipe Hungarian models
- Stanza Hungarian models
- spaCy Hungarian (if lemmatizer quality acceptable)
- Hungarian-specific analyzers (if installable and licensed)

Document exact versions, model names, and licenses in `docs/tooling.md`.

---

### Step 1 — Build a starter corpus for seed selection
Goal: collect ~2–20M tokens quickly from a reliable source (Hungarian Wikipedia dump is fine) to compute wordform frequencies.

Tasks:
1. Download/prepare Hungarian Wikipedia text dump (or similar).
2. Basic cleaning: remove markup, keep plain text.
3. Tokenize to wordforms; compute frequency.
4. Select **~500 mid-frequency wordforms**:
   - frequency range heuristic: emulate Swedish’s 1000–6000 range; adjust if starter corpus differs
   - length ≥ 5 characters
   - exclude digits, mixed scripts, URLs, obvious named entities if detectable
5. Save `data/seeds/seed_words.txt` and a CSV with counts.

Output:
- `data/seeds/seed_words.txt`
- `data/seeds/seed_words.csv`

---

### Step 2 — Web corpus acquisition (≥100M tokens)
Preferred: use an existing WAC-style corpus for Hungarian if accessible and licensable; otherwise build one.

If building:
1. Query generation:
   - repeatedly sample **3 random seed words** per query
2. Retrieval:
   - fetch result pages and extract main text
3. Cleaning:
   - boilerplate removal (nav/ads)
   - language-ID filtering (Hungarian vs others)
   - dedup near-identical documents and paragraphs
   - store provenance (URL, timestamp, hash)

Stop when cleaned corpus reaches target size.

Outputs:
- `data/corpus_raw/*.jsonl` (url + raw html or raw extracted text)
- `data/corpus_clean/*.jsonl` (clean text + metadata)
- `docs/corpus_stats.md` (token counts, dedup rates, lang-id pass rate)

---

### Step 3 — Linguistic annotation (lemma + POS)
Annotate `corpus_clean` with chosen NLP stack.

Requirements:
- Keep both token form and lemma
- Keep POS tags in a consistent tagset (Universal POS strongly preferred)
- Store sentence boundaries (for later dispersion calc)
- Log annotation failure rates and unknown tokens

Outputs:
- `data/annotated/*.conllu.zst` or `.jsonl.zst`
- `docs/annotation_quality.md` with summary stats

---

### Step 4 — Frequency + dispersion metrics
Compute for each **(lemma, POS)**:
- RF = total occurrences
- WPM = (RF / total_tokens) * 1,000,000
- Dispersion metric:
  - If ARF is implementable: do it
  - Otherwise implement a proxy:
    - split corpus into N shards (e.g., by document or equal-token chunks)
    - compute dispersion across shards (e.g., Juilland D, or normalized entropy)
    - compute “domain concentration penalty” that down-ranks items appearing in few shards
  - Document formula in `docs/metrics.md`

Selection procedure (Swedish pattern):
1. rank by dispersion-aware metric (ARF-like)
2. then order chosen headwords by WPM for final presentation :contentReference[oaicite:7]{index=7}

Outputs:
- `data/lists/freq_full.tsv` (lemma, pos, RF, WPM, dispersion_metric, rank fields)

---

### Step 5 — Filter noise + POS selection
Apply filters similar to Swedish M1 cleanup:
- remove tokens/lemmas containing digits or non-letter noise (`><=;\'*` etc.)
- remove punctuation, foreign words (lang-id or character heuristics), and most proper names
- remove numerals except a curated set (define for Hungarian; document)
- remove participles if your tagger distinguishes them; otherwise define exclusion rules

Then:
- create a filtered candidate list (target tens of thousands lemma+pos, like Swedish ~54k) :contentReference[oaicite:8]{index=8}

Outputs:
- `data/lists/m1_candidates.tsv`

---

### Step 6 — Select top ~9,000 and proofread (M2)
Select the top 9,000 candidates by your dispersion-aware ranking, then:

A) Automatic checks
- cross-check against at least one Hungarian lexical resource if available (wordlist / morphological lexicon)
- flag:
  - lemma not recognized
  - suspicious lemma derivations (common tagger errors)
  - POS inconsistencies for very high-frequency function words

B) Manual/semiautomatic cleanup (must be scripted as much as possible)
- merge spelling/style variants where reasonable (document rule)
- normalize headword form
- keep alternative variants in parentheses or a dedicated column

C) Hungarian-specific “grammar info” column
Swedish added article/infinitive markers to help learners; Hungarian lacks articles as lexical markers but you can add something useful, e.g.:
- verb prefix separation note (if you detect common particle verbs)
- definite/indefinite verb paradigm note (only if reliable; otherwise omit)
If unsure, leave blank and keep the column for future enrichment.

Outputs:
- `data/lists/m2_top9000_proofed.tsv`
- `data/lists/proofing_flags.tsv`
- `docs/proofing_rules.md`

---

### Step 7 — Validation / inclusion–deletion candidates
Swedish used “validation through translation” across partner languages to identify:
- deletion candidates: in monolingual list but never in translation lists
- inclusion candidates: appear in translation lists but absent in monolingual list :contentReference[oaicite:9]{index=9}

For Hungarian, implement one of these validation strategies (choose best feasible):

Option 1 (preferred if you can get them):
- Compare to existing learner vocabulary lists / graded wordlists for Hungarian.

Option 2:
- Use bilingual dictionaries or aligned parallel corpora with 2–3 high-resource languages.
  - Translate top-N from those languages into Hungarian (automatic MT is acceptable for candidate generation only)
  - Compare overlaps to generate inclusion/deletion candidate sets

Option 3:
- Compare against multiple corpora genres:
  - web corpus (core)
  - a balanced written corpus (control)
  - optionally subtitles/spoken-like data
Flag items with extreme domain skew.

Agent must:
- produce `inclusion_candidates.tsv` and `deletion_candidates.tsv`
- write a short rationale doc: `docs/validation.md`

After validation, finalize the list size (8,000–9,000 is acceptable; Swedish ended at 8,425). :contentReference[oaicite:10]{index=10}

---

### Step 8 — CEFR banding
Implement Swedish’s pragmatic approach:
- assign CEFR levels by **frequency bands**, approximately equal headword counts per level (Swedish ~1,404 per level across 6 levels). :contentReference[oaicite:11]{index=11}

For a 9,000 list:
- ~1,500 items per level (A1–C2)

Store:
- `cefr_level` column
- band thresholds in `docs/cefr_banding.md`

---

### Step 9 — Coverage evaluation
Compute coverage of:
- the web corpus itself (should be high; Swedish reported ~80% coverage by the Kelly items) :contentReference[oaicite:12]{index=12}
- at least one control corpus from a different time/genre distribution, to quantify drift and tagging mismatch effects

Report:
- coverage by CEFR level
- top reasons for mismatch (tagging, lemma differences, genre shift)

Outputs:
- `docs/coverage_report.md`
- `data/lists/coverage_by_level.tsv`

---

### Step 10 — Release packaging
Create final release artifacts:

1) `data/lists/hungarian_kelly.tsv` with columns:
- id
- rf
- wpm
- (optional) arf_or_dispersion
- cefr_level
- source (e.g., "webcorpus", "validation_add", "manual_add")
- gram_info (optional)
- lemma
- pos
- comment/example (optional)

2) `README.md` describing:
- how corpus was built
- annotation tools
- metrics
- known limitations (tagging noise, web genre skew)
- how to cite

3) `LICENSE`
Choose an open license compatible with your data + tools. Swedish used CC-BY-SA / LGPL for distribution; Hungarian list should be clearly licensed and reproducible. :contentReference[oaicite:13]{index=13}

Apache 2.0 is the best choice; MIT is also acceptable. CC-BY is best for data,
CC-BY-SA is acceptable if required.

---

## Acceptance criteria
- Reproducible pipeline: `scripts/run_all.sh` (or `make all`) builds final TSV from raw inputs
- Corpus size documented; token counts and dedup stats included
- Lemma+POS frequency list computed with WPM and a dispersion-aware metric
- Noise filtering + POS inclusion rules documented
- Final list: ~9,000 lemma+POS entries with CEFR bands
- Coverage report produced on ≥2 corpora (core + control)
- All key decisions recorded in `docs/`

---

## Implementation notes for the agent
- Prefer streaming processing and compressed artifacts (`.zst`) due to corpus size.
- Keep intermediate outputs; avoid “magic” notebooks.
- Log everything (seed selection parameters, crawl queries, tool versions, random seeds).
- Add unit tests for:
  - WPM calculation
  - dispersion metric
  - filtering rules
  - CEFR band assignment

---

## What not to do
- Do not handwave corpus collection; if using an existing corpus, document access and license clearly.
- Do not rely purely on raw frequency without dispersion control; domain skew is a known issue in web corpora. :contentReference[oaicite:14]{index=14}
- Do not output surface wordforms only; this must be lemma+POS (“lem-pos”) entries. :contentReference[oaicite:15]{index=15}

