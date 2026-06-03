# Notes: A Web-Derived Corpus of Spoken Irish

## Core concept

Distribute stand-off annotations (time-aligned text, phonetic transcriptions)
that reference copyrighted audio at its original URL rather than redistributing
audio. Users reconstruct the corpus by fetching audio themselves.

## Primary source: leighleat.ie

Irish-language audiobook platform. Each "page" is a URL
(`leighleat.com/pages/NNN` or `leighleat.com/poems/NNN`) with audio embedded.

### Text extraction cases

**Case 1 — HTML text available**
Some books have text accessible via CSS selectors in the page HTML.
Per-page selector maps are in the notebooks (see below).

Books with HTML selectors already mapped:
- Mallachtaí
- An Bhean Chaointe
- Clíona
- Jimín (Mháire Thaidhg)
- Fabhalsceálta (fables: Léon Baineann, Gaoth agus Grian, …)
- Dánta (poems)
- Mianta

**Case 2 — Image/PDF text (OCR path)**
Some books serve text as images (e.g. "Dron", page NNNs have `img_selector`).
A graphical tool (VGG Image Annotator / VIA) was used to mark bounding boxes
on page images; text is extracted from those regions.
VIA project JSON: `via_project_16May2025_16h0m (2).json` (in Colab at time of
notebook).

**Case 3 — Text accessible but audio not on leighleat**
*An tIriseoir* (Michelle Nic Pháidín, Cois Life 2016):
- Audio on SoundCloud + excerpt (Chapter 1) on another site
- Text extracted from PDF using the VIA bounding-box tool
- XML from pdf2xml is in `an_tiriseoir/An_tIriseoir.xml`

**Case 4 — Public domain text, leighleat audio**
*Mo Scéal Féin* (Peadar Ua Laoghaire):
- 32 chapters already processed with wav2vec
- JSON in `mo-sceal-fein/MsfChapter*.ogg.w2v.json`
- Text is freely available (public domain)

## Scraping infrastructure

Notebooks (in `/Users/joregan/Playing/notes/_notebooks/`):
- `2025-05-15-leighleat-pieces.ipynb` — MIANTA selector map, Dron image
  selectors, VIA bbox extraction
- `2025-05-15-mallachtai.ipynb` — MALLACHTAI, AN_BHEAN_CHAOINTE, CLIONA,
  JIMIN, FABHALSCEALTA, DANTA selector maps; scraping code using
  requests + BeautifulSoup; CoNLL stripping utilities

`get_selectors_from_dict(url_dict, url)` fetches a leighleat page, extracts
text from the listed CSS selectors, and records the MD5 hash + audio URLs.

## Alignment pipeline

`align_librivox/` contains WhisperX JSON → text alignment code.
The same pipeline applies here: fetch audio → WhisperX → align against
extracted text → output stand-off annotation.

## Source type taxonomy

1. **HTML text** — leighleat page; URL + CSS selector per segment
2. **Image scan** — leighleat page serves image; URL + bounding box
3. **Video read-along** — YouTube; URL + frame timestamp + bounding box;
   audio and text co-located in the same source; page-turn events give natural
   segmentation. (Channel TBD — not currently available.)
4. **Public domain text** — freely available reference text, no extraction
   needed (e.g. *Mo Scéal Féin*, *An Bhean Chaointe*)
5. **Publisher PDF** — bounding box on PDF page (e.g. *An tIriseoir*);
   PDFs provided by publisher, not publicly available

## What gets distributed

**Not distributed:** any `text` field — the text is reconstructed by the user
fetching the source URLs themselves.

**Distributed:**
- For HTML pages: URL + CSS selector(s) per segment
- For image pages: URL + bounding box coordinates per region
- Audio alignment: URL + timestamps (once alignment pipeline runs)

The OCR corrections in `stories.json` (`corr` field) can be derived
automatically elsewhere (e.g. spell-checking against a reference); they do not
need to be hard-coded into the distributed annotations.

## Planned: automated selector extraction

`leighleat-poems.json` (flat `text` list, no selectors) is the test case for a
script that, given the text content of a region, finds the most exactly
matching CSS selector in the fetched page HTML. This would replace manual
selector authoring for new books.

## Stand-off annotation format

### Original design: MD5-gated reconstruction

The corpus is distributed as annotations only. The reconstruction script checks
the local work directory for an audio file whose MD5 matches the one recorded
in the annotation. If no matching file is found, that item is silently skipped.
This means:
- Users who have obtained the audio (legitimately) can reconstruct the corpus.
- Users who have not cannot, so no copyrighted content is distributed.
- The annotations themselves are useful even without the audio (text + timing).

### Text from publishers

For some books the publisher provided PDFs (not generally made available
publicly). These PDFs are the source for text extraction via the VIA bounding-
box tool. The audio for these same books is already publicly available from
the publisher (e.g. on leighleat.ie or SoundCloud). So the text path is
privileged, but the audio path is open — the stand-off model still applies.

### Format (not yet decided)

Should record at minimum:
- Source URL (leighleat page)
- MD5 of audio file (for local reconstruction gating)
- Segment: `{start_s, end_s, text}`

Consider W3C Web Annotation or a simple JSON/TSV format.

## Background context (do not assert in paper without confirmation)

The audio on leighleat.ie appears to be freely accessible, possibly because the
market for Irish-language audiobooks is small enough that the publisher chose
not to restrict access. Do not state this as fact in the paper — confirm the
actual reason (licensing policy, accessibility initiative, etc.) before citing
it.

## Open questions

- What does Teanga's submission format require for audio corpora?
- Do leighleat ToS / robots.txt permit automated fetching?
- Is there dialect metadata per book / per speaker on leighleat?
- Where are the output JSON files from the notebooks (mallachtaí.json,
  an_bhean_chaointe.json, cliona.json, jimin.json, fabhalscealta.json)?
  Were they saved to Colab only, or committed somewhere?
