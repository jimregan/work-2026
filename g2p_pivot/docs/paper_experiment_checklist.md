# G2P Pivot Paper And Experiment Checklist

## Current Paper State

- The fallback/baseline text exists in the new journal direction, but `g2p-pivot-merged.tex` in this workspace is still LREC-formatted.
- The pasted NEJLT template version should be treated as the current journal-target draft until the workspace `.tex` file is updated or replaced.
- The baseline table from the LREC draft is stale because the earlier preprocessing removed stress markers. For a TTS-facing paper, stress must be preserved unless an experiment explicitly ablates it.

## Immediate Blocking Items

1. Decide which `.tex` file is canonical.
   - Candidate: pasted NEJLT version.
   - Current workspace file: `g2p-pivot-merged.tex`, still LREC-style.

2. Re-run the Phonetisaurus baseline with stress markers preserved.
   - Do not use the stress-stripping cleanup used for exploratory graphone bootstrapping.
   - Keep syllable/stress/boundary treatment explicit and documented.
   - Recompute WL, MWL, WLM, and RAW.
   - Replace the current PER table only after the rerun.

3. Record the exact phone normalization used for each experiment.
   - Baseline G2P for TTS: preserve stress markers.
   - IBM Model 1 graphone bootstrap: may strip/drop stress and boundary symbols only as a separate alignment diagnostic.
   - Prompted T5 experiments: decide whether stress is target output, auxiliary output, or separate task.

## Background Expansion For Journal Version

1. G2P as a TTS component.
   - Why lexicons are incomplete.
   - Why phone/stress representations remain useful despite end-to-end TTS.
   - Difference between orthographic transparency and TTS adequacy.

2. Classical G2P methods.
   - Rule-based systems.
   - Joint-sequence graphone models.
   - WFST/Phonetisaurus-style baselines.
   - What stress markers mean for TTS-oriented evaluation.

3. Multilingual and mixed-language G2P.
   - Monolingual assumptions.
   - Loanwords and names.
   - Language tags and SSML.
   - Phoneset mapping versus direct target-phoneset prediction.

4. Pivoting and prompting.
   - Pivot languages in MT as background analogy.
   - Target-language prompt tokens in multilingual neural models.
   - Why this paper is not broad zero-shot G2P.
   - The target use case: pronounce a foreign-origin word inside a Swedish TTS system using the Swedish speaker/system phoneset.

5. Swedish-specific structure.
   - Compounds.
   - Compound splitting as a multitask pivot task.
   - Foreign-origin components inside Swedish compounds.
   - Stress and compound prosody as TTS-relevant output.

6. Braxen/NST resources.
   - Braxen as a practical TTS lexicon.
   - Language-origin tags as useful but noisy.
   - NST compound and named-entity information, if included.
   - Decide whether NST is core method, appendix material, or future work.

## Baseline Rerun Checklist

1. Reconstruct the baseline data extraction.
   - Input: Braxen TSV.
   - Relevant fields: word column `0`, phones column `1`, language/origin column `3`.
   - Skip comment lines beginning with `#`.

2. Preserve stress-bearing phone tokens.
   - Do not strip prefixes such as `"`, `'`, or `,` when producing training/evaluation targets.
   - Do not drop syllable or boundary symbols unless the baseline being reproduced did so and the paper states that clearly.

3. Rebuild filtered language sets.
   - Start with the old retained set: `swe`, `eng`, `dan`, `fre`, `spa`, `ita`, `lat`.
   - Re-check counts after preserving stress.
   - Re-check any spelling-dictionary filtering.

4. Recreate the four baseline configurations.
   - WL: one model per language.
   - MWL: merged model with language-specific prefix.
   - WLM: merged model tested without prefix.
   - RAW: all entries mixed without prefix.

5. Recompute metrics.
   - PER with stress preserved.
   - Consider reporting both segmental PER and stress-sensitive PER if the difference is informative.
   - Keep the old table out of the paper or mark it as superseded until rerun.

6. Store reproducibility artifacts.
   - Split files.
   - Training lexicons.
   - Model commands/configuration.
   - Predictions.
   - Evaluation script output.

## Prompted/Pivot Experiment Checklist

1. Define prompted tasks.
   - `g2p|lang=sv`
   - `g2p|lang=sv|from=en`
   - `ipa2ipa|from=en|to=sv`
   - `compound|lang=sv`
   - possibly `compound-lang|lang=sv`

2. Decide stress handling in neural targets.
   - Single target sequence including phones and stress.
   - Separate auxiliary stress task.
   - Two metrics: segmental accuracy and stress-aware accuracy.

3. Build compound supervision.
   - Use NST-style decomposition if available.
   - Recover part-level origin labels where possible.
   - Treat compound splitting as a multitask pivot task, not merely preprocessing.

4. Build tag-diagnosis data.
   - Orthographic feature flags.
   - IBM Model 1 1:1 graphone seeds as diagnostic evidence.
   - Later graphone stages allowing multigraphs, deletions, insertions, and stress.

5. Ablation matrix.
   - Baseline only.
   - Prompted multilingual G2P.
   - Prompted donor-language G2P.
   - Explicit IPA-to-IPA pivot.
   - Compound task added.
   - Compound plus pivot.
   - Vary amount of direct foreign-word-to-Swedish target data.

## Paper Integration Checklist

1. Replace the introduction contribution paragraph.
   - Current baseline-only framing is too narrow for the intended journal paper.

2. Expand background before adding more results.
   - The journal version needs the historical and methodological bridge from classical G2P to prompted multitask/pivot models.

3. Move commented `junk.tex` ideas into either prose or checklist items.
   - Avoid leaving key claims only in comments.

4. Mark the current baseline table as provisional.
   - It should not stand as final because stress was removed.

5. Keep the central claim narrow.
   - The paper is about controllable TTS-bounded pronunciation of foreign-origin lexical material, not unconstrained zero-shot G2P.
