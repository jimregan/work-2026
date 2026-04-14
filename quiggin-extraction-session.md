# Quiggin Extraction Session Summary

## Project Overview

JSON extraction from Quiggin's *A Dialect of Donegal* (1906), a phonetic description of the Irish of Meenawannia, Co. Donegal. Working directory: `/Users/joregan/pron-ga/pron2/quiggin/`. Branch: `ux`.

## Key Files

- **scripts/quiggin/audit.py**: JSON validation tool. Scans `raw/` (or any directory) for parse errors, non-array fields, singular/plural key mismatches, typo keys (glass→gloss, transription→transcription, respondant→respondent), unknown keys, bare-string 'also' fields, and missing 'id' fields. Usage: `python audit.py raw/` or `audit.py --json`.

- **scripts/quiggin/workflow.py**: Core extraction coordinator using Anthropic API (claude-sonnet-4-6). Key components:
  - `claude_chat(messages, model, client)`: Calls Anthropic API, filters system messages into the system parameter
  - `validate_entries(entries, expected_section)`: Schema compliance checks including: missing 'raw', raw==phonetic identity, gloss without curly quotes, missing 'section', typo keys, singular keys that should be plural, non-array fields that should be arrays, non-array 'also', typo keys in examples
  - `extract(wiki_text, model, client, max_retries, ...)`: Runs extraction with retry loop, feeding validation failures back as context
  - `discover_pending(force_all)`: Globs `section*.wiki` only (excludes wordlist etc.), returns list of (stem, wiki_path) for files not in raw/ or no-entries/
  - `process_file()`: Orchestrates extraction for one file, writes JSON to unedited/
  - `WorkflowState`: Persists done/failed/skipped state to workflow_state.json

- **scripts/finck/fetch_wiki.py**: Fetches pages from de.wikisource.org/wiki/Seite:Die_araner_mundart.djvu/{n}?action=edit. Extracts wikitext from wpTextbox1, page number from wpHeaderTextbox using regex for `– N –` format. Saves as finck/wiki/pageN.wiki where N is the book page number. Uses 1.5s delay. Supports --start/--end/--force flags.

- **quiggin/CLAUDE.md**: Project docs listing known data quality issues.

## Data Flow

```
section*.wiki → workflow.py → unedited/*.json → (manual review) → raw/*.json
```

## System Prompt Constraints (workflow.py)

Critical rules added to prevent hallucination:
- Include only data explicitly present in the source text
- Do NOT infer, paraphrase, or add explanatory text
- If it is not a verbatim substring of the input, it does not belong in any field
- Gloss delimiters are Unicode U+2018/U+2019 (curly quotes), not U+0027 (ASCII apostrophes)
- Each quoted form with a gloss in prose sections = a separate entry

## Validation Checks

The validator flags:
- `raw` field missing
- `raw == phonetic` (lazy copy)
- Entry has gloss but raw contains no curly quotes (model mangled U+2018/U+2019 → U+0027)
- Missing `section` field
- Typo keys: `glass`, `transription`, `respondant`
- Singular keys that should be plural: `example`, `note`, `relation`, `see_section`
- Non-array fields that should be arrays: `etymology`, `source_refs`, `related`, `relations`, `derived_from`, `see_section`, `example`, `focus`, `contrasted_with`
- Non-array `also` field
- Typo keys in examples

## Known Data Quality Issues (from CLAUDE.md)

- 38 files have JSON parse errors (mostly missing/trailing commas)
- Type inconsistencies: many keys appear as both bare object and array — should all be normalised to arrays: `etymology`, `source_refs`, `related`, `relations`, `derived_from`, `see_section`, `example`, `focus`, `contrasted_with`
- `related` vs `relations`: `related` = cognates in other languages (Welsh, Norse, Manx, Scots Gaelic); `relations` = morphological relationships within the dialect. Consider merging with a `type` value like `"cognate"` or `"compare"`.
- `related` vs `etymology`: `etymology` = direct ancestor; `related` = parallel/cognate form
- Singular/plural key mismatches: `example`(7x) vs `examples`(52x), `note`(34x) vs `notes`(214x), `relation`(1x) vs `relations`(297x), `see_section`(78x) vs `see_sections`(1x)
- Typos: `examples[].glass` → `gloss`, `transription` → `transcription`, `respondant` → `respondent`
- Mixed int/string for numeric fields: `page`, `column`, `year`
- Missing `id` fields in many files (format: `"s106-1"`)

## Run Command

```bash
ANTHROPIC_API_KEY=<key> python scripts/quiggin/workflow.py
```

Output goes to `unedited/`, then manually reviewed before moving to `raw/`.

## Pending Work

- Run full extraction on all 61 pending section*.wiki files
- Extract and process finck/wiki/ pages into finck/unedited/ (fetch_wiki.py is in place)
- Review extracted JSON for quality before moving to raw/

## Lessons Learned

- Local LLMs (qwen3.5:9b, gemma4:9b) were inadequate for this structured extraction task
- Claude sonnet-4-6 estimated ~$3.50 for full batch of 61 files
- Empty output guard: if wiki contains `''''` but output is `[]`, treat as failure and retry
- The djvu page number ≠ book page number — must parse `– N –` from wpHeaderTextbox
