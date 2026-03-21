# Hub Model Evaluation Results

Source data: `/tmp/hub-results/` and `/tmp/sweep-results/`.
Model: `Pendrokar/spoken-sentence-transformer` (multi-axis WavLM, hub checkpoint).
Results JSON stores top-10 hits per query.

---

## Task 1: P315 → VCTK Cross-Speaker Retrieval

**Setup:** 160 labelled p315 utterances (mic1) queried against the full VCTK index (~44 k
utterances, ~100 speakers). Sentence IDs from `/tmp/p315-labels.json`. Target = any hit
whose label matches the query's sentence_id. 12 queries have no label entry and are excluded.
Some sentence IDs are non-vctk (e.g. `p306_019`, `s5_257`) — these appear in the index
under that label; `—` means not found in the top-10 stored hits.

### Per-axis recall (top-10)

| Axis | N | R@1 | R@5 | R@10 | MRR | Mean rank when found |
|---|---|---|---|---|---|---|
| semantic   | 160 | 0.3750 | 0.4625 | 0.5062 | 0.4149 | 1.93 |
| speaker_id | 160 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N/A |
| dialect    | 160 | 0.2062 | 0.3063 | 0.3812 | 0.2593 | 2.74 |
| gender     | 160 | 0.1062 | 0.2125 | 0.2250 | 0.1526 | 2.08 |

`speaker_id`: p315 is absent from training; zero hits in top-10 for all queries.

### Per-query detail — semantic axis (160 labelled queries)

| Query | Sentence ID | Rank | Sim | Top-1 label | Top-1 sim |
|---|---|---|---|---|---|
| p315_001_mic1.flac | vctk_7282 | 2 | 0.9706 | s5_001 | 0.9744 |
| p315_003_mic1.flac | vctk_7275 | 1 | 0.9829 | vctk_7275 | 0.9829 |
| p315_004_mic1.flac | vctk_7289 | 1 | 0.9786 | vctk_7289 | 0.9786 |
| p315_005_mic1.flac | vctk_7292 | 1 | 0.9773 | vctk_7292 | 0.9773 |
| p315_006_mic1.flac | vctk_7305 | 1 | 0.9855 | vctk_7305 | 0.9855 |
| p315_009_mic1.flac | vctk_7123 | 1 | 0.9797 | vctk_7123 | 0.9797 |
| p315_013_mic1.flac | vctk_7299 | 1 | 0.9819 | vctk_7299 | 0.9819 |
| p315_014_mic1.flac | vctk_7284 | 1 | 0.9828 | vctk_7284 | 0.9828 |
| p315_016_mic1.flac | vctk_7271 | 2 | 0.9856 | vctk_5774 | 0.9873 |
| p315_018_mic1.flac | vctk_7136 | 2 | 0.9842 | s5_018 | 0.9855 |
| p315_019_mic1.flac | p306_019 | — | — | vctk_7135 | 0.9713 |
| p315_020_mic1.flac | vctk_7117 | 1 | 0.9832 | vctk_7117 | 0.9832 |
| p315_022_mic1.flac | vctk_7110 | 1 | 0.9856 | vctk_7110 | 0.9856 |
| p315_023_mic1.flac | vctk_7103 | 1 | 0.9887 | vctk_7103 | 0.9887 |
| p315_025_mic1.flac | vctk_4093 | 1 | 0.9567 | vctk_4093 | 0.9567 |
| p315_026_mic1.flac | vctk_1978 | 1 | 0.9626 | vctk_1978 | 0.9626 |
| p315_027_mic1.flac | vctk_1832 | — | — | vctk_4634 | 0.9638 |
| p315_030_mic1.flac | vctk_8124 | — | — | vctk_3820 | 0.9612 |
| p315_033_mic1.flac | vctk_6768 | — | — | vctk_3393 | 0.9523 |
| p315_041_mic1.flac | vctk_6794 | — | — | vctk_670 | 0.9655 |
| p315_042_mic1.flac | vctk_398 | 1 | 0.9712 | vctk_398 | 0.9712 |
| p315_046_mic1.flac | vctk_3779 | 1 | 0.9769 | vctk_3779 | 0.9769 |
| p315_049_mic1.flac | vctk_6775 | 2 | 0.9613 | vctk_7878 | 0.9640 |
| p315_051_mic1.flac | vctk_2005 | 1 | 0.9673 | vctk_2005 | 0.9673 |
| p315_055_mic1.flac | vctk_5684 | 1 | 0.9528 | vctk_5684 | 0.9528 |
| p315_056_mic1.flac | vctk_3481 | — | — | vctk_8607 | 0.9409 |
| p315_057_mic1.flac | vctk_4614 | — | — | vctk_3143 | 0.9538 |
| p315_066_mic1.flac | vctk_4554 | — | — | vctk_1324 | 0.9686 |
| p315_068_mic1.flac | vctk_4954 | — | — | vctk_172 | 0.9580 |
| p315_070_mic1.flac | vctk_6701 | — | — | vctk_7274 | 0.9554 |
| p315_072_mic1.flac | vctk_6693 | — | — | vctk_10410 | 0.9615 |
| p315_073_mic1.flac | vctk_4044 | — | — | vctk_7015 | 0.9576 |
| p315_075_mic1.flac | vctk_5631 | 1 | 0.9411 | vctk_5631 | 0.9411 |
| p315_076_mic1.flac | vctk_4590 | 1 | 0.9561 | vctk_4590 | 0.9561 |
| p315_077_mic1.flac | vctk_3808 | — | — | vctk_6538 | 0.9571 |
| p315_078_mic1.flac | vctk_4584 | — | — | vctk_12936 | 0.9591 |
| p315_085_mic1.flac | vctk_3190 | 10 | 0.9547 | vctk_161 | 0.9595 |
| p315_093_mic1.flac | vctk_6710 | — | — | vctk_3978 | 0.9600 |
| p315_094_mic1.flac | vctk_2225 | 1 | 0.9729 | vctk_2225 | 0.9729 |
| p315_095_mic1.flac | p330_099 | — | — | vctk_4691 | 0.9682 |
| p315_096_mic1.flac | vctk_6665 | 1 | 0.9546 | vctk_6665 | 0.9546 |
| p315_099_mic1.flac | p330_105 | — | — | vctk_11237 | 0.9581 |
| p315_100_mic1.flac | vctk_2483 | 1 | 0.9696 | vctk_2483 | 0.9696 |
| p315_102_mic1.flac | vctk_7356 | 1 | 0.9552 | vctk_7356 | 0.9552 |
| p315_105_mic1.flac | p330_112 | — | — | vctk_9624 | 0.9527 |
| p315_107_mic1.flac | vctk_6658 | 1 | 0.9712 | vctk_6658 | 0.9712 |
| p315_109_mic1.flac | p330_117 | — | — | vctk_11118 | 0.9788 |
| p315_111_mic1.flac | vctk_6803 | 1 | 0.9711 | vctk_6803 | 0.9711 |
| p315_114_mic1.flac | vctk_6674 | 7 | 0.9518 | vctk_2561 | 0.9584 |
| p315_115_mic1.flac | vctk_5169 | — | — | vctk_2027 | 0.9605 |
| p315_121_mic1.flac | vctk_6650 | — | — | vctk_7274 | 0.9614 |
| p315_123_mic1.flac | vctk_6635 | — | — | vctk_8607 | 0.9538 |
| p315_124_mic1.flac | vctk_6644 | — | — | vctk_10107 | 0.9569 |
| p315_126_mic1.flac | vctk_6600 | 1 | 0.9574 | vctk_6600 | 0.9574 |
| p315_128_mic1.flac | vctk_6608 | — | — | vctk_7579 | 0.9509 |
| p315_129_mic1.flac | p330_135 | — | — | vctk_5922 | 0.9661 |
| p315_131_mic1.flac | vctk_6624 | — | — | vctk_7391 | 0.9507 |
| p315_136_mic1.flac | p330_144 | — | — | vctk_4087 | 0.9694 |
| p315_141_mic1.flac | vctk_6684 | 1 | 0.9689 | vctk_6684 | 0.9689 |
| p315_142_mic1.flac | p330_151 | — | — | vctk_11123 | 0.9619 |
| p315_149_mic1.flac | vctk_8066 | — | — | vctk_4132 | 0.9526 |
| p315_154_mic1.flac | vctk_7890 | 3 | 0.9580 | vctk_7977 | 0.9598 |
| p315_158_mic1.flac | vctk_6613 | 2 | 0.9554 | s5_026 | 0.9593 |
| p315_161_mic1.flac | vctk_3669 | — | — | vctk_11538 | 0.9360 |
| p315_164_mic1.flac | vctk_6626 | — | — | vctk_13728 | 0.9609 |
| p315_166_mic1.flac | vctk_3158 | 1 | 0.9708 | vctk_3158 | 0.9708 |
| p315_167_mic1.flac | vctk_5790 | 3 | 0.9460 | vctk_6214 | 0.9506 |
| p315_171_mic1.flac | vctk_6721 | 10 | 0.9616 | vctk_2034 | 0.9677 |
| p315_174_mic1.flac | vctk_5663 | 3 | 0.9609 | vctk_7188 | 0.9632 |
| p315_176_mic1.flac | vctk_6676 | 1 | 0.9525 | vctk_6676 | 0.9525 |
| p315_178_mic1.flac | vctk_4969 | 1 | 0.9547 | vctk_4969 | 0.9547 |
| p315_179_mic1.flac | p330_183 | — | — | vctk_5464 | 0.9595 |
| p315_183_mic1.flac | vctk_6616 | 1 | 0.9611 | vctk_6616 | 0.9611 |
| p315_184_mic1.flac | vctk_6657 | — | — | vctk_7289 | 0.9608 |
| p315_188_mic1.flac | p330_193 | — | — | vctk_11130 | 0.9624 |
| p315_193_mic1.flac | vctk_1893 | 4 | 0.9493 | vctk_6030 | 0.9534 |
| p315_196_mic1.flac | p305_227 | — | — | vctk_11224 | 0.9695 |
| p315_203_mic1.flac | vctk_2737 | 1 | 0.9706 | vctk_2737 | 0.9706 |
| p315_206_mic1.flac | vctk_2424 | 1 | 0.9481 | vctk_2424 | 0.9481 |
| p315_207_mic1.flac | vctk_5679 | 1 | 0.9559 | vctk_5679 | 0.9559 |
| p315_208_mic1.flac | vctk_3143 | 7 | 0.9586 | vctk_6742 | 0.9663 |
| p315_209_mic1.flac | vctk_6615 | — | — | vctk_3482 | 0.9639 |
| p315_210_mic1.flac | vctk_4641 | — | — | vctk_13738 | 0.9491 |
| p315_212_mic1.flac | vctk_576 | 1 | 0.9438 | vctk_576 | 0.9438 |
| p315_213_mic1.flac | p330_218 | — | — | vctk_6868 | 0.9532 |
| p315_214_mic1.flac | vctk_5685 | 1 | 0.9784 | vctk_5685 | 0.9784 |
| p315_216_mic1.flac | vctk_1104 | — | — | vctk_640 | 0.9622 |
| p315_217_mic1.flac | vctk_6671 | 2 | 0.9617 | vctk_7256 | 0.9670 |
| p315_219_mic1.flac | vctk_433 | 1 | 0.9621 | vctk_433 | 0.9621 |
| p315_221_mic1.flac | vctk_2544 | 1 | 0.9641 | vctk_2544 | 0.9641 |
| p315_226_mic1.flac | vctk_6731 | — | — | vctk_12932 | 0.9562 |
| p315_228_mic1.flac | vctk_3519 | 9 | 0.9429 | s5_156 | 0.9504 |
| p315_229_mic1.flac | vctk_6709 | — | — | vctk_161 | 0.9633 |
| p315_230_mic1.flac | vctk_6703 | 1 | 0.9658 | vctk_6703 | 0.9658 |
| p315_231_mic1.flac | vctk_6606 | — | — | vctk_2122 | 0.9702 |
| p315_232_mic1.flac | vctk_5764 | — | — | vctk_8035 | 0.9672 |
| p315_239_mic1.flac | p330_246 | — | — | vctk_12800 | 0.9607 |
| p315_241_mic1.flac | vctk_694 | — | — | vctk_134 | 0.9520 |
| p315_242_mic1.flac | p330_251 | — | — | vctk_2281 | 0.9592 |
| p315_248_mic1.flac | p330_257 | — | — | vctk_11151 | 0.9559 |
| p315_250_mic1.flac | vctk_6653 | 1 | 0.9586 | vctk_6653 | 0.9586 |
| p315_255_mic1.flac | vctk_6744 | 1 | 0.9565 | vctk_6744 | 0.9565 |
| p315_256_mic1.flac | vctk_6751 | — | — | vctk_6706 | 0.9617 |
| p315_257_mic1.flac | vctk_6778 | 1 | 0.9683 | vctk_6778 | 0.9683 |
| p315_260_mic1.flac | p330_270 | — | — | vctk_2064 | 0.9471 |
| p315_261_mic1.flac | p330_271 | — | — | vctk_11155 | 0.9644 |
| p315_262_mic1.flac | vctk_5062 | — | — | vctk_6937 | 0.9522 |
| p315_264_mic1.flac | vctk_6558 | 1 | 0.9580 | vctk_6558 | 0.9580 |
| p315_265_mic1.flac | p341_195 | — | — | vctk_4399 | 0.9548 |
| p315_266_mic1.flac | vctk_2370 | 2 | 0.9623 | vctk_12072 | 0.9625 |
| p315_280_mic1.flac | vctk_6577 | 1 | 0.9722 | vctk_6577 | 0.9722 |
| p315_281_mic1.flac | vctk_6573 | 1 | 0.9668 | vctk_6573 | 0.9668 |
| p315_282_mic1.flac | vctk_6560 | — | — | vctk_2757 | 0.9520 |
| p315_285_mic1.flac | vctk_7071 | 1 | 0.9507 | vctk_7071 | 0.9507 |
| p315_293_mic1.flac | vctk_2717 | 1 | 0.9690 | vctk_2717 | 0.9690 |
| p315_294_mic1.flac | vctk_4573 | 4 | 0.9540 | vctk_6295 | 0.9565 |
| p315_298_mic1.flac | vctk_6585 | — | — | vctk_4161 | 0.9693 |
| p315_302_mic1.flac | vctk_6216 | — | — | vctk_3505 | 0.9799 |
| p315_305_mic1.flac | vctk_4623 | 1 | 0.9720 | vctk_4623 | 0.9720 |
| p315_306_mic1.flac | vctk_6769 | — | — | vctk_12137 | 0.9555 |
| p315_307_mic1.flac | p340_249 | — | — | vctk_3996 | 0.9674 |
| p315_308_mic1.flac | vctk_6759 | 1 | 0.9661 | vctk_6759 | 0.9661 |
| p315_309_mic1.flac | vctk_1733 | — | — | vctk_2314 | 0.9555 |
| p315_310_mic1.flac | vctk_6579 | — | — | vctk_10981 | 0.9597 |
| p315_311_mic1.flac | s5_257 | — | — | vctk_8094 | 0.9510 |
| p315_312_mic1.flac | vctk_7027 | 1 | 0.9678 | vctk_7027 | 0.9678 |
| p315_314_mic1.flac | vctk_4577 | — | — | vctk_3739 | 0.9595 |
| p315_315_mic1.flac | vctk_4109 | 1 | 0.9587 | vctk_4109 | 0.9587 |
| p315_318_mic1.flac | vctk_6559 | — | — | vctk_8607 | 0.9430 |
| p315_319_mic1.flac | vctk_4619 | 1 | 0.9613 | vctk_4619 | 0.9613 |
| p315_320_mic1.flac | vctk_4558 | 1 | 0.9632 | vctk_4558 | 0.9632 |
| p315_327_mic1.flac | vctk_4683 | — | — | vctk_9801 | 0.9515 |
| p315_328_mic1.flac | p311_296 | — | — | vctk_11231 | 0.9530 |
| p315_336_mic1.flac | vctk_5782 | 1 | 0.9308 | vctk_5782 | 0.9308 |
| p315_343_mic1.flac | vctk_5033 | 6 | 0.9559 | vctk_1855 | 0.9616 |
| p315_344_mic1.flac | vctk_4580 | 1 | 0.9479 | vctk_4580 | 0.9479 |
| p315_345_mic1.flac | vctk_977 | 4 | 0.9428 | vctk_2607 | 0.9488 |
| p315_346_mic1.flac | vctk_5839 | 1 | 0.9720 | vctk_5839 | 0.9720 |
| p315_347_mic1.flac | vctk_6795 | — | — | vctk_3218 | 0.9494 |
| p315_350_mic1.flac | vctk_6952 | — | — | vctk_7888 | 0.9811 |
| p315_352_mic1.flac | vctk_6544 | — | — | vctk_4469 | 0.9546 |
| p315_357_mic1.flac | vctk_6685 | — | — | vctk_11713 | 0.9335 |
| p315_359_mic1.flac | vctk_4581 | — | — | vctk_12957 | 0.9505 |
| p315_360_mic1.flac | vctk_12 | — | — | vctk_401 | 0.9721 |
| p315_366_mic1.flac | p340_306 | — | — | vctk_243 | 0.9637 |
| p315_368_mic1.flac | p340_310 | — | — | vctk_12493 | 0.9533 |
| p315_372_mic1.flac | vctk_6771 | 1 | 0.9770 | vctk_6771 | 0.9770 |
| p315_375_mic1.flac | vctk_901 | 1 | 0.9765 | vctk_901 | 0.9765 |
| p315_380_mic1.flac | vctk_6556 | 1 | 0.9379 | vctk_6556 | 0.9379 |
| p315_388_mic1.flac | vctk_6772 | 1 | 0.9431 | vctk_6772 | 0.9431 |
| p315_390_mic1.flac | vctk_4521 | 8 | 0.9559 | vctk_5070 | 0.9596 |
| p315_391_mic1.flac | p336_364 | — | — | vctk_8261 | 0.9391 |
| p315_392_mic1.flac | vctk_7371 | 1 | 0.9563 | vctk_7371 | 0.9563 |
| p315_393_mic1.flac | vctk_6595 | — | — | vctk_7837 | 0.9556 |
| p315_397_mic1.flac | vctk_8001 | — | — | vctk_7697 | 0.9374 |
| p315_403_mic1.flac | vctk_850 | — | — | vctk_9593 | 0.9561 |
| p315_405_mic1.flac | vctk_4661 | — | — | s5_333 | 0.9639 |
| p315_408_mic1.flac | p330_414 | — | — | vctk_11173 | 0.9644 |
| p315_414_mic1.flac | vctk_6792 | — | — | vctk_607 | 0.9489 |
| p315_421_mic1.flac | s5_140 | 4 | 0.9549 | vctk_6162 | 0.9592 |

---

## Task 2: REHASP → OSR Cross-Corpus Retrieval

**Setup:** 1170 REHASP queries (Lucy patient speaker) retrieved against a corpus.
P@k = fraction of queries where at least one correct hit (same `sentence_id`, different `corpus`)
appears in the combined top-k. Correct = same sentence from OSR, not REHASP.
Combined similarity = `semantic × 1.0 + speaker_id × w`.

### OSR-only corpus (case 1 — no REHASP in index)

| Combined weights | P@1 | P@5 | P@10 |
|---|---|---|---|
| sem=1.0, spk=0.0 | **0.6547** | 0.6632 | 0.6632 |

### Mixed corpus: REHASP + OSR (cases 2 and 3)

| spk weight | P@1 | P@5 | P@10 |
|---|---|---|---|
| +1.0 (default) | 0.0000 | 0.3863 | 0.5701 |
| 0.0 | 0.0000 | 0.4034 | 0.5675 |
| −0.5 | 0.0000 | 0.4068 | 0.5590 |
| −1.0 | 0.0000 | 0.4085 | 0.5444 |
| −1.5 | 0.0111 | 0.3470 | 0.4530 |
| −2.0 | 0.0427 | 0.1650 | 0.2487 |

---

## Task 3: Speaker Weight Sweep (20 queries, Lucy rep006 → rep023 + OSR)

P@k = fraction of 20 queries where an OSR (cross-corpus) same-sentence hit appears in
combined top-k. Combined similarity = `semantic × 1.0 + speaker_id × w`.

| spk weight | P@1 | P@5 | P@10 |
|---|---|---|---|
| +5.0 | 0.00 | 0.40 | 0.75 |
| +2.0 | 0.00 | 0.45 | 0.80 |
| +1.0 | 0.00 | 0.50 | 0.80 |
| +0.5 | 0.00 | 0.55 | 0.80 |
| +0.2 | 0.00 | 0.55 | 0.80 |
| +0.1 | 0.00 | 0.55 | 0.80 |
| +0.05 | 0.00 | 0.55 | 0.80 |
| +0.01 | 0.00 | 0.55 | 0.80 |
| +0.001 | 0.00 | 0.55 | 0.80 |
| 0.0 | 0.00 | 0.55 | 0.80 |
| −1.0 | 0.00 | 0.70 | 0.80 |
| −2.0 | 0.95 | 1.00 | 1.00 |
| −2.5 | 1.00 | 1.00 | 1.00 |
| −3.0 | 1.00 | 1.00 | 1.00 |
| −5.0 | 1.00 | 1.00 | 1.00 |

Note: sweep P@k computed by checking for OSR-prefixed IDs in combined top-k results;
`retrieval_eval.py` did not run `--precision_k` for the sweep (no `--query_labels`).
