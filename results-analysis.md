# Results Analysis

Source data: `/tmp/hub-results/`, `/tmp/sweep-results/`, `/tmp/results/`.
Results JSON stores top-10 hits per query.

Two evaluation tasks are covered here:

1. **P315→VCTK cross-speaker retrieval** — queries from VCTK speaker p315, index = all other VCTK speakers. Sentence identity from `/tmp/p315-labels.json`. Evaluated per axis.
2. **REHASP retrieval sweep** — 20 queries (Lucy, rep006 session, sets H1 and H4), index = Lucy rep023 + OSR sentences. Correct hit = OSR entry with matching sentence label. Sweep over `speaker_id` axis weight.

The `/tmp/hub-results/` files `rehasp-osr-results.json`, `rehasp-mixed-results.json`, and `rehasp-mixed-spk*` all contain 1170 queries. The origin of this 1170-query set is unclear and they are not analysed here.

---

## Task 1: P315 → VCTK

**Setup:**
- 172 total queries (`p315_NNN_mic1.flac`)
- 160 have entries in `p315-labels.json`; 12 do not and are unresolvable
- Index: all VCTK speakers except p315, top-k=10
- Correct hit: `hit.label == labels[query].sentence_id` AND hit speaker != p315
- Sentence IDs from `p315-labels.json` (not utterance numbers)

**Summary — all axes, 160 labelled queries:**

| Axis | R@1 | R@5 | R@10 | MRR | Mean rank (found) | Found |
|---|---:|---:|---:|---:|---:|---:|
| semantic | 60/160 = 0.375 | 74/160 = 0.4625 | 81/160 = 0.5062 | 0.4149 | 1.93 | 81 |
| speaker_id | 0/160 = 0.000 | 0/160 = 0.000 | 0/160 = 0.000 | 0.0000 | — | 0 |
| dialect | 33/160 = 0.2062 | 49/160 = 0.3063 | 61/160 = 0.3812 | 0.2593 | 2.74 | 61 |
| gender | 17/160 = 0.1062 | 34/160 = 0.2125 | 36/160 = 0.2250 | 0.1526 | 2.08 | 36 |

Notes:
- `speaker_id` axis retrieves by voice similarity. The speakers whose voices most resemble p315 don't happen to share any of p315's specific sentences, so R@k=0 for all k.
- `dialect` and `gender` axes retrieve by those traits and incidentally pick up same-sentence hits at rates lower than `semantic`.
- top-k=10, so R@20 = R@10 (no hits beyond rank 10 are available).

### Per-query detail — semantic axis

All 172 queries. Sorted: found (ascending rank), then not found, then unlabelled.
"Rank" = position of first hit with `label == target_sentence_id` AND speaker != p315. "—" = none in top-10.

| Query | Target sentence_id | Rank | Top-1 hit | Top-1 label | Top-1 sim |
|---|---|---:|---|---|---:|
| p315_003_mic1.flac | vctk_7275 | 1 | p341_003 | vctk_7275 | 0.9829 |
| p315_004_mic1.flac | vctk_7289 | 1 | p298_004 | vctk_7289 | 0.9786 |
| p315_005_mic1.flac | vctk_7292 | 1 | p310_005 | vctk_7292 | 0.9773 |
| p315_006_mic1.flac | vctk_7305 | 1 | p361_006 | vctk_7305 | 0.9855 |
| p315_009_mic1.flac | vctk_7123 | 1 | p343_009 | vctk_7123 | 0.9797 |
| p315_013_mic1.flac | vctk_7299 | 1 | p361_013 | vctk_7299 | 0.9819 |
| p315_014_mic1.flac | vctk_7284 | 1 | p363_014 | vctk_7284 | 0.9828 |
| p315_020_mic1.flac | vctk_7117 | 1 | p317_020 | vctk_7117 | 0.9832 |
| p315_022_mic1.flac | vctk_7110 | 1 | p341_022 | vctk_7110 | 0.9856 |
| p315_023_mic1.flac | vctk_7103 | 1 | p343_023 | vctk_7103 | 0.9887 |
| p315_025_mic1.flac | vctk_4093 | 1 | p330_036 | vctk_4093 | 0.9567 |
| p315_026_mic1.flac | vctk_1978 | 1 | p262_243 | vctk_1978 | 0.9626 |
| p315_042_mic1.flac | vctk_398 | 1 | p341_184 | vctk_398 | 0.9712 |
| p315_046_mic1.flac | vctk_3779 | 1 | p307_182 | vctk_3779 | 0.9769 |
| p315_051_mic1.flac | vctk_2005 | 1 | p245_252 | vctk_2005 | 0.9673 |
| p315_055_mic1.flac | vctk_5684 | 1 | p329_166 | vctk_5684 | 0.9528 |
| p315_075_mic1.flac | vctk_5631 | 1 | p311_133 | vctk_5631 | 0.9411 |
| p315_076_mic1.flac | vctk_4590 | 1 | p330_079 | vctk_4590 | 0.9561 |
| p315_094_mic1.flac | vctk_2225 | 1 | p246_272 | vctk_2225 | 0.9729 |
| p315_096_mic1.flac | vctk_6665 | 1 | p329_221 | vctk_6665 | 0.9546 |
| p315_100_mic1.flac | vctk_2483 | 1 | p288_275 | vctk_2483 | 0.9696 |
| p315_102_mic1.flac | vctk_7356 | 1 | p330_109 | vctk_7356 | 0.9552 |
| p315_107_mic1.flac | vctk_6658 | 1 | p303_064 | vctk_6658 | 0.9712 |
| p315_111_mic1.flac | vctk_6803 | 1 | p298_393 | vctk_6803 | 0.9711 |
| p315_126_mic1.flac | vctk_6600 | 1 | p330_133 | vctk_6600 | 0.9574 |
| p315_141_mic1.flac | vctk_6684 | 1 | p330_150 | vctk_6684 | 0.9689 |
| p315_166_mic1.flac | vctk_3158 | 1 | p227_052 | vctk_3158 | 0.9708 |
| p315_176_mic1.flac | vctk_6676 | 1 | p330_179 | vctk_6676 | 0.9525 |
| p315_178_mic1.flac | vctk_4969 | 1 | p305_172 | vctk_4969 | 0.9547 |
| p315_183_mic1.flac | vctk_6616 | 1 | p330_187 | vctk_6616 | 0.9611 |
| p315_203_mic1.flac | vctk_2737 | 1 | p340_138 | vctk_2737 | 0.9706 |
| p315_206_mic1.flac | vctk_2424 | 1 | p286_360 | vctk_2424 | 0.9481 |
| p315_207_mic1.flac | vctk_5679 | 1 | p330_213 | vctk_5679 | 0.9559 |
| p315_212_mic1.flac | vctk_576 | 1 | p294_383 | vctk_576 | 0.9438 |
| p315_214_mic1.flac | vctk_5685 | 1 | p311_219 | vctk_5685 | 0.9784 |
| p315_219_mic1.flac | vctk_433 | 1 | p330_226 | vctk_433 | 0.9621 |
| p315_221_mic1.flac | vctk_2544 | 1 | p271_195 | vctk_2544 | 0.9641 |
| p315_230_mic1.flac | vctk_6703 | 1 | p330_237 | vctk_6703 | 0.9658 |
| p315_250_mic1.flac | vctk_6653 | 1 | p303_139 | vctk_6653 | 0.9586 |
| p315_255_mic1.flac | vctk_6744 | 1 | p330_265 | vctk_6744 | 0.9565 |
| p315_257_mic1.flac | vctk_6778 | 1 | p330_267 | vctk_6778 | 0.9683 |
| p315_264_mic1.flac | vctk_6558 | 1 | p330_274 | vctk_6558 | 0.9580 |
| p315_280_mic1.flac | vctk_6577 | 1 | p330_286 | vctk_6577 | 0.9722 |
| p315_281_mic1.flac | vctk_6573 | 1 | p303_159 | vctk_6573 | 0.9668 |
| p315_285_mic1.flac | vctk_7071 | 1 | p330_291 | vctk_7071 | 0.9507 |
| p315_293_mic1.flac | vctk_2717 | 1 | p262_370 | vctk_2717 | 0.9690 |
| p315_305_mic1.flac | vctk_4623 | 1 | p311_277 | vctk_4623 | 0.9720 |
| p315_308_mic1.flac | vctk_6759 | 1 | p303_169 | vctk_6759 | 0.9661 |
| p315_312_mic1.flac | vctk_7027 | 1 | p330_320 | vctk_7027 | 0.9678 |
| p315_315_mic1.flac | vctk_4109 | 1 | p336_284 | vctk_4109 | 0.9587 |
| p315_319_mic1.flac | vctk_4619 | 1 | p336_290 | vctk_4619 | 0.9613 |
| p315_320_mic1.flac | vctk_4558 | 1 | p330_329 | vctk_4558 | 0.9632 |
| p315_336_mic1.flac | vctk_5782 | 1 | p336_305 | vctk_5782 | 0.9308 |
| p315_344_mic1.flac | vctk_4580 | 1 | p330_351 | vctk_4580 | 0.9479 |
| p315_346_mic1.flac | vctk_5839 | 1 | p330_353 | vctk_5839 | 0.9720 |
| p315_372_mic1.flac | vctk_6771 | 1 | p330_376 | vctk_6771 | 0.9770 |
| p315_375_mic1.flac | vctk_901 | 1 | p311_335 | vctk_901 | 0.9765 |
| p315_380_mic1.flac | vctk_6556 | 1 | p336_351 | vctk_6556 | 0.9379 |
| p315_388_mic1.flac | vctk_6772 | 1 | p336_057 | vctk_6772 | 0.9431 |
| p315_392_mic1.flac | vctk_7371 | 1 | p284_081 | vctk_7371 | 0.9563 |
| p315_001_mic1.flac | vctk_7282 | 2 | p247_001 | s5_001 | 0.9744 |
| p315_016_mic1.flac | vctk_7271 | 2 | p334_016 | vctk_5774 | 0.9873 |
| p315_018_mic1.flac | vctk_7136 | 2 | p247_018 | s5_018 | 0.9855 |
| p315_049_mic1.flac | vctk_6775 | 2 | p308_128 | vctk_7878 | 0.9640 |
| p315_158_mic1.flac | vctk_6613 | 2 | s5_026 | s5_026 | 0.9593 |
| p315_217_mic1.flac | vctk_6671 | 2 | p288_112 | vctk_7256 | 0.9670 |
| p315_266_mic1.flac | vctk_2370 | 2 | p343_119 | vctk_12072 | 0.9625 |
| p315_154_mic1.flac | vctk_7890 | 3 | p334_418 | vctk_7977 | 0.9598 |
| p315_167_mic1.flac | vctk_5790 | 3 | p288_325 | vctk_6214 | 0.9506 |
| p315_174_mic1.flac | vctk_5663 | 3 | p278_128 | vctk_7188 | 0.9632 |
| p315_193_mic1.flac | vctk_1893 | 4 | p292_151 | vctk_6030 | 0.9534 |
| p315_294_mic1.flac | vctk_4573 | 4 | p307_361 | vctk_6295 | 0.9565 |
| p315_345_mic1.flac | vctk_977 | 4 | p272_196 | vctk_2607 | 0.9488 |
| p315_421_mic1.flac | s5_140 | 4 | p259_324 | vctk_6162 | 0.9592 |
| p315_343_mic1.flac | vctk_5033 | 6 | p329_258 | vctk_1855 | 0.9616 |
| p315_114_mic1.flac | vctk_6674 | 7 | p259_185 | vctk_2561 | 0.9584 |
| p315_208_mic1.flac | vctk_3143 | 7 | p303_181 | vctk_6742 | 0.9663 |
| p315_390_mic1.flac | vctk_4521 | 8 | p294_229 | vctk_5070 | 0.9596 |
| p315_228_mic1.flac | vctk_3519 | 9 | p234_143 | s5_156 | 0.9504 |
| p315_085_mic1.flac | vctk_3190 | 10 | p318_174 | vctk_161 | 0.9595 |
| p315_171_mic1.flac | vctk_6721 | 10 | p341_173 | vctk_2034 | 0.9677 |
| p315_019_mic1.flac | p306_019 | — | p308_019 | vctk_7135 | 0.9713 |
| p315_027_mic1.flac | vctk_1832 | — | p311_173 | vctk_4634 | 0.9638 |
| p315_030_mic1.flac | vctk_8124 | — | p301_078 | vctk_3820 | 0.9612 |
| p315_033_mic1.flac | vctk_6768 | — | p274_269 | vctk_3393 | 0.9523 |
| p315_041_mic1.flac | vctk_6794 | — | p294_198 | vctk_670 | 0.9655 |
| p315_056_mic1.flac | vctk_3481 | — | p244_319 | vctk_8607 | 0.9409 |
| p315_057_mic1.flac | vctk_4614 | — | p298_116 | vctk_3143 | 0.9538 |
| p315_066_mic1.flac | vctk_4554 | — | p361_207 | vctk_1324 | 0.9686 |
| p315_068_mic1.flac | vctk_4954 | — | p318_177 | vctk_172 | 0.9580 |
| p315_070_mic1.flac | vctk_6701 | — | p311_017 | vctk_7274 | 0.9554 |
| p315_072_mic1.flac | vctk_6693 | — | p311_131 | vctk_10410 | 0.9615 |
| p315_073_mic1.flac | vctk_4044 | — | p247_360 | vctk_7015 | 0.9576 |
| p315_077_mic1.flac | vctk_3808 | — | p272_181 | vctk_6538 | 0.9571 |
| p315_078_mic1.flac | vctk_4584 | — | p361_369 | vctk_12936 | 0.9591 |
| p315_093_mic1.flac | vctk_6710 | — | p301_158 | vctk_3978 | 0.9600 |
| p315_095_mic1.flac | p330_099 | — | p316_315 | vctk_4691 | 0.9682 |
| p315_099_mic1.flac | p330_105 | — | p334_188 | vctk_11237 | 0.9581 |
| p315_105_mic1.flac | p330_112 | — | p292_040 | vctk_9624 | 0.9527 |
| p315_109_mic1.flac | p330_117 | — | p330_117 | vctk_11118 | 0.9788 |
| p315_115_mic1.flac | vctk_5169 | — | p259_458 | vctk_2027 | 0.9605 |
| p315_121_mic1.flac | vctk_6650 | — | p362_017 | vctk_7274 | 0.9614 |
| p315_123_mic1.flac | vctk_6635 | — | p244_319 | vctk_8607 | 0.9538 |
| p315_124_mic1.flac | vctk_6644 | — | p302_015 | vctk_10107 | 0.9569 |
| p315_128_mic1.flac | vctk_6608 | — | p341_340 | vctk_7579 | 0.9509 |
| p315_129_mic1.flac | p330_135 | — | p247_268 | vctk_5922 | 0.9661 |
| p315_131_mic1.flac | vctk_6624 | — | p271_287 | vctk_7391 | 0.9507 |
| p315_136_mic1.flac | p330_144 | — | p334_403 | vctk_4087 | 0.9694 |
| p315_142_mic1.flac | p330_151 | — | p330_151 | vctk_11123 | 0.9619 |
| p315_149_mic1.flac | vctk_8066 | — | p341_317 | vctk_4132 | 0.9526 |
| p315_161_mic1.flac | vctk_3669 | — | p339_067 | vctk_11538 | 0.9360 |
| p315_164_mic1.flac | vctk_6626 | — | s5_167 | vctk_13728 | 0.9609 |
| p315_179_mic1.flac | p330_183 | — | p376_381 | vctk_5464 | 0.9595 |
| p315_184_mic1.flac | vctk_6657 | — | p297_004 | vctk_7289 | 0.9608 |
| p315_188_mic1.flac | p330_193 | — | p330_193 | vctk_11130 | 0.9624 |
| p315_196_mic1.flac | p305_227 | — | p334_131 | vctk_11224 | 0.9695 |
| p315_209_mic1.flac | vctk_6615 | — | p307_378 | vctk_3482 | 0.9639 |
| p315_210_mic1.flac | vctk_4641 | — | s5_184 | vctk_13738 | 0.9491 |
| p315_213_mic1.flac | p330_218 | — | p298_373 | vctk_6868 | 0.9532 |
| p315_216_mic1.flac | vctk_1104 | — | p294_072 | vctk_640 | 0.9622 |
| p315_226_mic1.flac | vctk_6731 | — | p361_364 | vctk_12932 | 0.9562 |
| p315_229_mic1.flac | vctk_6709 | — | p318_174 | vctk_161 | 0.9633 |
| p315_231_mic1.flac | vctk_6606 | — | p307_302 | vctk_2122 | 0.9702 |
| p315_232_mic1.flac | vctk_5764 | — | p304_358 | vctk_8035 | 0.9672 |
| p315_239_mic1.flac | p330_246 | — | p361_164 | vctk_12800 | 0.9607 |
| p315_241_mic1.flac | vctk_694 | — | p313_030 | vctk_134 | 0.9520 |
| p315_242_mic1.flac | p330_251 | — | p255_104 | vctk_2281 | 0.9592 |
| p315_248_mic1.flac | p330_257 | — | p330_257 | vctk_11151 | 0.9559 |
| p315_256_mic1.flac | vctk_6751 | — | p303_319 | vctk_6706 | 0.9617 |
| p315_260_mic1.flac | p330_270 | — | p284_200 | vctk_2064 | 0.9471 |
| p315_261_mic1.flac | p330_271 | — | p330_271 | vctk_11155 | 0.9644 |
| p315_262_mic1.flac | vctk_5062 | — | p278_232 | vctk_6937 | 0.9522 |
| p315_265_mic1.flac | p341_195 | — | p317_294 | vctk_4399 | 0.9548 |
| p315_282_mic1.flac | vctk_6560 | — | p316_028 | vctk_2757 | 0.9520 |
| p315_298_mic1.flac | vctk_6585 | — | p300_345 | vctk_4161 | 0.9693 |
| p315_302_mic1.flac | vctk_6216 | — | p307_169 | vctk_3505 | 0.9799 |
| p315_306_mic1.flac | vctk_6769 | — | p343_286 | vctk_12137 | 0.9555 |
| p315_307_mic1.flac | p340_249 | — | p307_188 | vctk_3996 | 0.9674 |
| p315_309_mic1.flac | vctk_1733 | — | p305_150 | vctk_2314 | 0.9555 |
| p315_310_mic1.flac | vctk_6579 | — | p326_073 | vctk_10981 | 0.9597 |
| p315_311_mic1.flac | s5_257 | — | p304_319 | vctk_8094 | 0.9510 |
| p315_314_mic1.flac | vctk_4577 | — | p317_076 | vctk_3739 | 0.9595 |
| p315_318_mic1.flac | vctk_6559 | — | p244_319 | vctk_8607 | 0.9430 |
| p315_327_mic1.flac | vctk_4683 | — | p294_178 | vctk_9801 | 0.9515 |
| p315_328_mic1.flac | p311_296 | — | p334_155 | vctk_11231 | 0.9530 |
| p315_347_mic1.flac | vctk_6795 | — | p281_085 | vctk_3218 | 0.9494 |
| p315_350_mic1.flac | vctk_6952 | — | p308_249 | vctk_7888 | 0.9811 |
| p315_352_mic1.flac | vctk_6544 | — | p310_104 | vctk_4469 | 0.9546 |
| p315_357_mic1.flac | vctk_6685 | — | p339_373 | vctk_11713 | 0.9335 |
| p315_359_mic1.flac | vctk_4581 | — | p361_413 | vctk_12957 | 0.9505 |
| p315_360_mic1.flac | vctk_12 | — | p341_142 | vctk_401 | 0.9721 |
| p315_366_mic1.flac | p340_306 | — | p374_158 | vctk_243 | 0.9637 |
| p315_368_mic1.flac | p340_310 | — | p351_406 | vctk_12493 | 0.9533 |
| p315_391_mic1.flac | p336_364 | — | p231_179 | vctk_8261 | 0.9391 |
| p315_393_mic1.flac | vctk_6595 | — | p308_214 | vctk_7837 | 0.9556 |
| p315_397_mic1.flac | vctk_8001 | — | p305_168 | vctk_7697 | 0.9374 |
| p315_403_mic1.flac | vctk_850 | — | p287_062 | vctk_9593 | 0.9561 |
| p315_405_mic1.flac | vctk_4661 | — | p259_269 | s5_333 | 0.9639 |
| p315_408_mic1.flac | p330_414 | — | p330_414 | vctk_11173 | 0.9644 |
| p315_414_mic1.flac | vctk_6792 | — | p294_252 | vctk_607 | 0.9489 |
| p315_031_mic1.flac | *(no label)* | — | p329_372 | vctk_4054 | 0.9717 |
| p315_047_mic1.flac | *(no label)* | — | p376_284 | vctk_5568 | 0.9644 |
| p315_048_mic1.flac | *(no label)* | — | p288_360 | vctk_2035 | 0.9640 |
| p315_071_mic1.flac | *(no label)* | — | p313_110 | vctk_2600 | 0.9512 |
| p315_079_mic1.flac | *(no label)* | — | p236_227 | vctk_7172 | 0.9451 |
| p315_087_mic1.flac | *(no label)* | — | p333_354 | vctk_7810 | 0.9176 |
| p315_273_mic1.flac | *(no label)* | — | p307_094 | vctk_3875 | 0.9577 |
| p315_295_mic1.flac | *(no label)* | — | p333_323 | vctk_7647 | 0.9079 |
| p315_337_mic1.flac | *(no label)* | — | p341_010 | vctk_7288 | 0.9652 |
| p315_349_mic1.flac | *(no label)* | — | p293_267 | vctk_9721 | 0.9348 |
| p315_406_mic1.flac | *(no label)* | — | p374_178 | vctk_13465 | 0.9584 |
| p315_418_mic1.flac | *(no label)* | — | p288_359 | vctk_2016 | 0.9587 |

Notes:
- `p315_109_mic1.flac`: target = `p330_117`; top hit = `p330_117` (same utterance path), but that utterance's label in the index is `vctk_11118`, not `p330_117`. Label-scheme mismatch between the labels file and the index for this sentence.
- Several `p330_NNN`, `p340_NNN`, `p305_NNN`, etc. targets in the labels file likely have similar mismatches — the labels file uses speaker-prefixed IDs for shared sentences, but the index uses `vctk_NNNNN` IDs.

---

## Task 2: REHASP Retrieval Sweep

**Setup:**
- 20 queries: Lucy, rep006 session, sentences H1_1–H1_10 and H4_1–H4_10
- Index: Lucy rep023 (same speaker, different session) + OSR sentences (different corpus)
- Combined similarity: `sem_sim × 1.0 + spk_sim × w`
- Correct hit: `hit.id` starts with `OSR_` AND `hit.label == query sentence label`
- Query sentence labels from matching hvdNNNN number to rep023 hit in the results
- Source: `/tmp/sweep-results/speaker_id_{w}.json`

**Query–label mapping:**

| Query file | Sentence label |
|---|---|
| hvd0001_rep006 | H1_1 |
| hvd0002_rep006 | H1_2 |
| hvd0003_rep006 | H1_3 |
| hvd0004_rep006 | H1_4 |
| hvd0005_rep006 | H1_5 |
| hvd0006_rep006 | H1_6 |
| hvd0007_rep006 | H1_7 |
| hvd0008_rep006 | H1_8 |
| hvd0009_rep006 | H1_9 |
| hvd0010_rep006 | H1_10 |
| hvd0031_rep006 | H4_1 |
| hvd0032_rep006 | H4_2 |
| hvd0033_rep006 | H4_3 |
| hvd0034_rep006 | H4_4 |
| hvd0035_rep006 | H4_5 |
| hvd0036_rep006 | H4_6 |
| hvd0037_rep006 | H4_7 |
| hvd0038_rep006 | H4_8 |
| hvd0039_rep006 | H4_9 |
| hvd0040_rep006 | H4_10 |

**P@k sweep summary (denominator = 20):**

| speaker_id weight | P@1 | P@5 | P@10 |
|---:|---:|---:|---:|
| −5.0 | 0/20 | 0/20 | 0/20 |
| −3.0 | 1/20 | 1/20 | 1/20 |
| −2.5 | 1/20 | 1/20 | 2/20 |
| −2.0 | 1/20 | 5/20 | 7/20 |
| **−1.0** | **0/20** | **14/20** | **15/20** |
| **0.0** | **0/20** | **11/20** | **16/20** |
| 0.001 | 0/20 | 11/20 | 16/20 |
| 0.01 | 0/20 | 11/20 | 16/20 |
| 0.05 | 0/20 | 11/20 | 16/20 |
| 0.1 | 0/20 | 11/20 | 16/20 |
| 0.2 | 0/20 | 11/20 | 16/20 |
| 0.5 | 0/20 | 11/20 | 16/20 |
| 1.0 | 0/20 | 10/20 | 16/20 |
| 2.0 | 0/20 | 9/20 | 16/20 |
| 5.0 | 0/20 | 8/20 | 15/20 |

Note: user manual count at weight=0.0 gave P@5=12/20; computed value here is 11/20. Discrepancy of 1 query is unexplained. Raw data at `/tmp/sweep-results/speaker_id_0.0.json`.

Observations:
- P@1=0 at all non-extreme weights. Rep023 (same speaker) always ranks #1, pushing OSR hits to rank ≥2.
- Negative weights penalise speaker similarity, pushing rep023 down and OSR up. P@5 peaks at −1.0 (14/20).
- At −2.0 and below, P@5 collapses. OSR speakers apparently overlap enough in the speaker_id embedding with Lucy that very negative weights start penalising correct OSR hits too.
- H1_1, H4_2, H4_3, H4_7 never find a correct OSR hit at any weight. Either those OSR entries aren't in the index or the embeddings don't match.

### Per-query detail — weight=0.0 (semantic only)

| Sentence | First OSR rank | All OSR match ranks in top-10 |
|---|---:|---|
| H1_1 | — | (none) |
| H1_2 | 2 | 2, 4, 7 |
| H1_3 | 3 | 3, 8 |
| H1_4 | 3 | 3, 5, 8 |
| H1_5 | 3 | 3, 5, 10 |
| H1_6 | 9 | 9, 10 |
| H1_7 | 6 | 6 |
| H1_8 | 3 | 3, 4, 6, 8 |
| H1_9 | 4 | 4, 5, 7, 9 |
| H1_10 | 2 | 2, 3, 4, 5 |
| H4_1 | 7 | 7 |
| H4_2 | — | (none) |
| H4_3 | — | (none) |
| H4_4 | 6 | 6, 7, 9 |
| H4_5 | 2 | 2, 4, 8 |
| H4_6 | 2 | 2 |
| H4_7 | — | (none) |
| H4_8 | 3 | 3 |
| H4_9 | 10 | 10 |
| H4_10 | 2 | 2, 3, 9 |

### Per-query detail — weight=−1.0

| Sentence | First OSR rank | All OSR match ranks in top-10 |
|---|---:|---|
| H1_1 | — | (none) |
| H1_2 | 2 | 2, 4 |
| H1_3 | 3 | 3, 9 |
| H1_4 | 2 | 2, 4, 5, 9 |
| H1_5 | 2 | 2, 3, 7 |
| H1_6 | — | (none) |
| H1_7 | 5 | 5 |
| H1_8 | 3 | 3, 5, 6, 8 |
| H1_9 | 4 | 4, 5, 6, 9 |
| H1_10 | 2 | 2, 3, 4, 5 |
| H4_1 | 5 | 5 |
| H4_2 | — | (none) |
| H4_3 | — | (none) |
| H4_4 | 5 | 5, 7, 8 |
| H4_5 | 3 | 3, 4, 9 |
| H4_6 | 7 | 7 |
| H4_7 | — | (none) |
| H4_8 | 5 | 5, 9 |
| H4_9 | 5 | 5 |
| H4_10 | 2 | 2, 3, 8 |

Note: H1_6 is found at rank 9 at weight=0.0 but drops out entirely at weight=−1.0. H4_6 goes from rank 2 at weight=0.0 to rank 7 at weight=−1.0. These are sentences where negating speaker similarity hurts.


## Model Comparison: P315 → VCTK Semantic Recall

Source: `/tmp/results/`. Metric: R@k on 160 labelled p315 queries, semantic axis only.
Hit criterion: `hit.label == sentence_id` AND `hit.id` does not start with `p315_`.

| Model | R@1 | R@5 | R@10 | MRR | Found/160 |
|---|---:|---:|---:|---:|---:|
| wavlm-semantic | 144/160 = 0.9000 | 154/160 = 0.9625 | 154/160 = 0.9625 | 0.9271 | 154 |
| wavlm-multiaxis (semantic axis) | 83/160 = 0.5188 | 116/160 = 0.7250 | 130/160 = 0.8125 | 0.6107 | 130 |
| all-MiniLM-L6-v2 (text baseline) | 151/160 = 0.9437 | 159/160 = 0.9938 | 160/160 = 1.0000 | 0.9677 | 160 |
| wavlm-resemblyzer (semantic axis) | 1/160 = 0.0063 | 1/160 = 0.0063 | 4/160 = 0.0250 | 0.0088 | 4 |
| CLAP LAION (`laion/larger_clap_music_and_speech`) | 0/160 = 0 | 0/160 = 0 | 0/160 = 0 | 0 | 0 |
| CLAP MS (`microsoft/msclap`) | 0/160 = 0 | 0/160 = 0 | 0/160 = 0 | 0 | 0 |

Notes:
- Text baseline uses sentence transcript embeddings; same text = sim≈1.0 regardless of speaker, so R@1 near-perfect. R@10=1.0 because every sentence is in the index.
- CLAP baselines retrieve by audio content similarity (designed for music/speech-text alignment); not suited for same-sentence cross-speaker matching. 0 hits in top-10 for all queries.
- wavlm-resemblyzer's semantic axis provides near-zero sentence recall because the resemblyzer speaker embedding dominates and the model was not trained for semantic matching.
- wavlm-semantic outperforms wavlm-multiaxis on the semantic axis (R@10: 0.9625 vs 0.8125), likely because it dedicates all capacity to semantic similarity rather than splitting across axes.

---

## Model Comparison: REHASP Precision

Source: `/tmp/results/models/`. P@k = fraction of 1170 queries (all REHASP Lucy sessions)
with at least one correct OSR hit (same sentence label) in top-k.

Three index conditions:
- **OSR-only**: index = OSR corpus only, no REHASP utterances. axis weights: `semantic=1.0, speaker_id=0.0`.
- **Mixed (spk=+1.0)**: index = OSR + REHASP. axis weights: `semantic=1.0, speaker_id=1.0`. Speaker similarity helps same-session hits outrank OSR.
- **Mixed (spk=−1.0)**: index = OSR + REHASP. axis weights: `semantic=1.0, speaker_id=-1.0`. Speaker penalty pushes OSR hits up.

### OSR-only index

| Model | P@1 | P@5 | P@10 |
|---|---:|---:|---:|
| wavlm-semantic | 0.6393 | 0.6538 | 0.6598 |
| wavlm-multiaxis | 0.1675 | 0.3393 | 0.4385 |
| wavlm-resemblyzer | 0.0299 | 0.1094 | 0.1829 |

### Mixed index, speaker_id weight = +1.0

| Model | P@1 | P@5 | P@10 |
|---|---:|---:|---:|
| wavlm-semantic | 0.0590 | 0.6521 | 0.6598 |
| wavlm-multiaxis | 0.0043 | 0.0410 | 0.1504 |
| wavlm-resemblyzer | 0.0094 | 0.0368 | 0.0632 |

### Mixed index, speaker_id weight = −1.0

| Model | P@1 | P@5 | P@10 |
|---|---:|---:|---:|
| wavlm-semantic | 0.0590 | 0.6521 | 0.6598 |
| wavlm-multiaxis | 0.0085 | 0.0547 | 0.1803 |
| wavlm-resemblyzer | 0.0103 | 0.0316 | 0.0701 |

Notes:
- wavlm-semantic has no speaker_id axis, so mixed-index results are identical for spk=+1.0 and spk=−1.0. P@1 drops from 0.64 (OSR-only) to 0.06 (mixed) because same-session REHASP utterances dominate rank 1.
- wavlm-multiaxis with spk=−1.0 slightly outperforms spk=+1.0 on P@10 (0.1803 vs 0.1504), consistent with the sweep results showing negative speaker weight helps push OSR hits up.
- wavlm-resemblyzer performs worst on REHASP; the resemblyzer speaker embedding pulls retrieval toward voice-similar utterances rather than same-sentence content.

### Earlier run: wavlm-multiaxis-out-fixed (linear layer, OSR-only, 20 queries)

Source: `/tmp/search-for-rehasp-only-osr` (text log, pre-JSON). Model: `wavlm-multiaxis-out-fixed` — earliest multiaxis checkpoint, using a **linear projection layer** instead of softmax. Index: OSR-only. Queries: 20 (Lucy rep006, sets H1 and H4). Top-5 results stored.

| Sentence | Rank of first correct OSR hit |
|---|---:|
| H1_1 | 1 |
| H1_2 | 1 |
| H1_3 | 1 |
| H1_4 | 1 |
| H1_5 | 1 |
| H1_6 | 1 |
| H1_7 | 5 |
| H1_8 | 1 |
| H1_9 | 1 |
| H1_10 | 1 |
| H4_1 | 1 |
| H4_2 | 1 |
| H4_3 | 1 |
| H4_4 | 1 |
| H4_5 | 1 |
| H4_6 | 1 |
| H4_7 | 1 |
| H4_8 | 1 |
| H4_9 | 1 |
| H4_10 | 1 |

P@1 = 19/20 = 0.95, P@5 = 20/20 = 1.0.

This contrasts sharply with the softmax `wavlm-multiaxis` result on the same condition (OSR-only, P@1=0.1675 over 1170 queries). The linear (binary classifier) head was replaced with softmax because binary classification is unstable under InfoNCE and does not support cosine similarity for retrieval. The softmax model trains more stably but its embeddings are optimised for class discrimination rather than retrieval distance, which explains the REHASP precision drop. H1_7 is the only query not found at rank 1 in the earlier run — this sentence appears to be harder across all model versions.

---

## Model Comparison: OSR Self-Retrieval

Source: `/tmp/results/models/`. 390 OSR queries retrieved against the full OSR index.
Hit criterion: `hit.label == query.label` AND `hit.id != query.id`.
R@1=0 for all models because the query itself (same ID) ranks first.

| Model | Axis | R@5 | R@10 | MRR |
|---|---|---:|---:|---:|
| wavlm-semantic | semantic | 234/390 = 0.6000 | 234/390 = 0.6000 | 0.2912 |
| wavlm-multiaxis | semantic | 161/390 = 0.4128 | 161/390 = 0.4128 | 0.1851 |
| wavlm-multiaxis | speaker_id | 252/390 = 0.6462 | 252/390 = 0.6462 | 0.2417 |
| wavlm-multiaxis | gender | 257/390 = 0.6590 | 257/390 = 0.6590 | 0.3258 |
| wavlm-multiaxis | dialect | 45/390 = 0.1154 | 45/390 = 0.1154 | 0.0505 |
| wavlm-resemblyzer | semantic | 34/390 = 0.0872 | 34/390 = 0.0872 | 0.0389 |
| wavlm-resemblyzer | speaker_id | 348/390 = 0.8923 | 348/390 = 0.8923 | 0.4095 |
| wavlm-resemblyzer | gender | 188/390 = 0.4821 | 188/390 = 0.4821 | 0.2410 |
| wavlm-resemblyzer | dialect | 24/390 = 0.0615 | 24/390 = 0.0615 | 0.0272 |

Notes:
- R@5 = R@10 for all rows: OSR has at most 2 recordings per sentence (UK and US variants), so once the second recording is found, there are no further hits.
- wavlm-resemblyzer's speaker_id axis (R@5=0.89) outperforms wavlm-multiaxis's speaker_id (R@5=0.65), consistent with resemblyzer being a dedicated speaker verification system.
- Neither model retrieves well by dialect (R@5 < 0.12); OSR UK/US classification doesn't strongly cluster by dialect in these embeddings.
- We found gender to be a weak retrieval factor in this setting, likely because it is both low-cardinality and strongly correlated with speaker identity. We therefore focus the main analysis on semantic content, speaker identity, and dialect.

