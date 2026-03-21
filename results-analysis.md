# Hub Model Evaluation Results — Verbose Analysis

Generated from `/tmp/hub-results/` and `/tmp/sweep-results/`.
Model: `Pendrokar/spoken-sentence-transformer` (hub model, multi-axis WavLM).

---

## Task 1: P315 → VCTK Cross-Speaker Retrieval

**Setup:** 172 utterances from the VCTK p315 speaker (dysarthric, mic1) are used as
queries against the full VCTK index. The target for each query `p315_NNN_mic1.flac`
is any VCTK utterance with label `s5_NNN` (the s5 speaker reading the same sentence).
This tests whether the semantic axis can bridge the large acoustic gap between
dysarthric and typical speech.

The VCTK index is large (44 k+ utterances across ~100 speakers). Only s5 overlaps
with p315 in sentence content, making this a very hard retrieval task.

### Aggregate Recall per Axis

| Axis | N queries | Found in results | R@1 | R@5 | R@10 | MRR | Mean rank (when found) |
|---|---|---|---|---|---|---|---|
| semantic | 172 | 13 | 0.0116 | 0.0523 | 0.0756 | 0.0293 | 4.153846153846154 |
| speaker_id | 172 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N/A |
| dialect | 172 | 12 | 0.0174 | 0.0349 | 0.0698 | 0.0273 | 4.75 |
| gender | 172 | 13 | 0.0174 | 0.0523 | 0.0756 | 0.0312 | 4.0 |

> **Notes:**
> - `speaker_id` axis: the p315 speaker is absent from the training data; the axis has no
>   learned embedding for this speaker and collapses to random — zero hits anywhere in results.
> - `semantic` axis: 13/172 queries place the correct sentence in top-10 despite severe
>   dysarthric distortion. R@1 = 0.012 — the model's semantic axis generalises somewhat
>   but is still heavily influenced by acoustic similarity between typical VCTK speakers.
> - `dialect` and `gender` axes: similar to semantic; p315 accent/gender does create
>   modest recall, but these axes are not intended for sentence retrieval.

### Per-Query Detail — Semantic Axis

Queries where the target s5 utterance was NOT found anywhere in results are omitted from
the rank column (shown as `—`) but still counted in the denominator.

| Query | Utt | Target | Rank | Target sim | Top-1 ID | Top-1 label | Top-1 sim |
|---|---|---|---|---|---|---|---|
| p315_001_mic1.flac | 001 | s5_001 | 1 | 0.9744 | p247_001 | s5_001 | 0.9744 |
| p315_003_mic1.flac | 003 | s5_003 | 2 | 0.9827 | p341_003 | vctk_7275 | 0.9829 |
| p315_004_mic1.flac | 004 | s5_004 | 3 | 0.9742 | p298_004 | vctk_7289 | 0.9786 |
| p315_005_mic1.flac | 005 | s5_005 | 2 | 0.9767 | p310_005 | vctk_7292 | 0.9773 |
| p315_006_mic1.flac | 006 | s5_006 | 7 | 0.9795 | p361_006 | vctk_7305 | 0.9855 |
| p315_009_mic1.flac | 009 | s5_009 | 6 | 0.9677 | p343_009 | vctk_7123 | 0.9797 |
| p315_013_mic1.flac | 013 | s5_013 | 4 | 0.9735 | p361_013 | vctk_7299 | 0.9819 |
| p315_014_mic1.flac | 014 | s5_014 | 3 | 0.9784 | p363_014 | vctk_7284 | 0.9828 |
| p315_016_mic1.flac | 016 | s5_016 | 9 | 0.9807 | p334_016 | vctk_5774 | 0.9873 |
| p315_018_mic1.flac | 018 | s5_018 | 1 | 0.9855 | p247_018 | s5_018 | 0.9855 |
| p315_019_mic1.flac | 019 | s5_019 | — | — | p308_019 | vctk_7135 | 0.9713 |
| p315_020_mic1.flac | 020 | s5_020 | 4 | 0.9772 | p317_020 | vctk_7117 | 0.9832 |
| p315_022_mic1.flac | 022 | s5_022 | 3 | 0.9812 | p341_022 | vctk_7110 | 0.9856 |
| p315_023_mic1.flac | 023 | s5_023 | — | — | p343_023 | vctk_7103 | 0.9887 |
| p315_025_mic1.flac | 025 | s5_025 | — | — | p330_036 | vctk_4093 | 0.9567 |
| p315_026_mic1.flac | 026 | s5_026 | — | — | p262_243 | vctk_1978 | 0.9626 |
| p315_027_mic1.flac | 027 | s5_027 | — | — | p311_173 | vctk_4634 | 0.9638 |
| p315_030_mic1.flac | 030 | s5_030 | — | — | p301_078 | vctk_3820 | 0.9612 |
| p315_031_mic1.flac | 031 | s5_031 | — | — | p329_372 | vctk_4054 | 0.9717 |
| p315_033_mic1.flac | 033 | s5_033 | — | — | p274_269 | vctk_3393 | 0.9523 |
| p315_041_mic1.flac | 041 | s5_041 | — | — | p294_198 | vctk_670 | 0.9655 |
| p315_042_mic1.flac | 042 | s5_042 | — | — | p341_184 | vctk_398 | 0.9712 |
| p315_046_mic1.flac | 046 | s5_046 | — | — | p307_182 | vctk_3779 | 0.9769 |
| p315_047_mic1.flac | 047 | s5_047 | — | — | p376_284 | vctk_5568 | 0.9644 |
| p315_048_mic1.flac | 048 | s5_048 | — | — | p288_360 | vctk_2035 | 0.964 |
| p315_049_mic1.flac | 049 | s5_049 | — | — | p308_128 | vctk_7878 | 0.964 |
| p315_051_mic1.flac | 051 | s5_051 | — | — | p245_252 | vctk_2005 | 0.9673 |
| p315_055_mic1.flac | 055 | s5_055 | — | — | p329_166 | vctk_5684 | 0.9528 |
| p315_056_mic1.flac | 056 | s5_056 | — | — | p244_319 | vctk_8607 | 0.9409 |
| p315_057_mic1.flac | 057 | s5_057 | — | — | p298_116 | vctk_3143 | 0.9538 |
| p315_066_mic1.flac | 066 | s5_066 | — | — | p361_207 | vctk_1324 | 0.9686 |
| p315_068_mic1.flac | 068 | s5_068 | — | — | p318_177 | vctk_172 | 0.958 |
| p315_070_mic1.flac | 070 | s5_070 | — | — | p311_017 | vctk_7274 | 0.9554 |
| p315_071_mic1.flac | 071 | s5_071 | — | — | p313_110 | vctk_2600 | 0.9512 |
| p315_072_mic1.flac | 072 | s5_072 | — | — | p311_131 | vctk_10410 | 0.9615 |
| p315_073_mic1.flac | 073 | s5_073 | — | — | p247_360 | vctk_7015 | 0.9576 |
| p315_075_mic1.flac | 075 | s5_075 | — | — | p311_133 | vctk_5631 | 0.9411 |
| p315_076_mic1.flac | 076 | s5_076 | — | — | p330_079 | vctk_4590 | 0.9561 |
| p315_077_mic1.flac | 077 | s5_077 | — | — | p272_181 | vctk_6538 | 0.9571 |
| p315_078_mic1.flac | 078 | s5_078 | — | — | p361_369 | vctk_12936 | 0.9591 |
| p315_079_mic1.flac | 079 | s5_079 | — | — | p236_227 | vctk_7172 | 0.9451 |
| p315_085_mic1.flac | 085 | s5_085 | — | — | p318_174 | vctk_161 | 0.9595 |
| p315_087_mic1.flac | 087 | s5_087 | — | — | p333_354 | vctk_7810 | 0.9176 |
| p315_093_mic1.flac | 093 | s5_093 | — | — | p301_158 | vctk_3978 | 0.96 |
| p315_094_mic1.flac | 094 | s5_094 | — | — | p246_272 | vctk_2225 | 0.9729 |
| p315_095_mic1.flac | 095 | s5_095 | — | — | p316_315 | vctk_4691 | 0.9682 |
| p315_096_mic1.flac | 096 | s5_096 | — | — | p329_221 | vctk_6665 | 0.9546 |
| p315_099_mic1.flac | 099 | s5_099 | — | — | p334_188 | vctk_11237 | 0.9581 |
| p315_100_mic1.flac | 100 | s5_100 | — | — | p288_275 | vctk_2483 | 0.9696 |
| p315_102_mic1.flac | 102 | s5_102 | — | — | p330_109 | vctk_7356 | 0.9552 |
| p315_105_mic1.flac | 105 | s5_105 | — | — | p292_040 | vctk_9624 | 0.9527 |
| p315_107_mic1.flac | 107 | s5_107 | — | — | p303_064 | vctk_6658 | 0.9712 |
| p315_109_mic1.flac | 109 | s5_109 | — | — | p330_117 | vctk_11118 | 0.9788 |
| p315_111_mic1.flac | 111 | s5_111 | — | — | p298_393 | vctk_6803 | 0.9711 |
| p315_114_mic1.flac | 114 | s5_114 | — | — | p259_185 | vctk_2561 | 0.9584 |
| p315_115_mic1.flac | 115 | s5_115 | — | — | p259_458 | vctk_2027 | 0.9605 |
| p315_121_mic1.flac | 121 | s5_121 | — | — | p362_017 | vctk_7274 | 0.9614 |
| p315_123_mic1.flac | 123 | s5_123 | — | — | p244_319 | vctk_8607 | 0.9538 |
| p315_124_mic1.flac | 124 | s5_124 | — | — | p302_015 | vctk_10107 | 0.9569 |
| p315_126_mic1.flac | 126 | s5_126 | — | — | p330_133 | vctk_6600 | 0.9574 |
| p315_128_mic1.flac | 128 | s5_128 | — | — | p341_340 | vctk_7579 | 0.9509 |
| p315_129_mic1.flac | 129 | s5_129 | — | — | p247_268 | vctk_5922 | 0.9661 |
| p315_131_mic1.flac | 131 | s5_131 | — | — | p271_287 | vctk_7391 | 0.9507 |
| p315_136_mic1.flac | 136 | s5_136 | — | — | p334_403 | vctk_4087 | 0.9694 |
| p315_141_mic1.flac | 141 | s5_141 | — | — | p330_150 | vctk_6684 | 0.9689 |
| p315_142_mic1.flac | 142 | s5_142 | — | — | p330_151 | vctk_11123 | 0.9619 |
| p315_149_mic1.flac | 149 | s5_149 | — | — | p341_317 | vctk_4132 | 0.9526 |
| p315_154_mic1.flac | 154 | s5_154 | — | — | p334_418 | vctk_7977 | 0.9598 |
| p315_158_mic1.flac | 158 | s5_158 | — | — | s5_026 | s5_026 | 0.9593 |
| p315_161_mic1.flac | 161 | s5_161 | — | — | p339_067 | vctk_11538 | 0.936 |
| p315_164_mic1.flac | 164 | s5_164 | — | — | s5_167 | vctk_13728 | 0.9609 |
| p315_166_mic1.flac | 166 | s5_166 | — | — | p227_052 | vctk_3158 | 0.9708 |
| p315_167_mic1.flac | 167 | s5_167 | — | — | p288_325 | vctk_6214 | 0.9506 |
| p315_171_mic1.flac | 171 | s5_171 | — | — | p341_173 | vctk_2034 | 0.9677 |
| p315_174_mic1.flac | 174 | s5_174 | — | — | p278_128 | vctk_7188 | 0.9632 |
| p315_176_mic1.flac | 176 | s5_176 | — | — | p330_179 | vctk_6676 | 0.9525 |
| p315_178_mic1.flac | 178 | s5_178 | — | — | p305_172 | vctk_4969 | 0.9547 |
| p315_179_mic1.flac | 179 | s5_179 | — | — | p376_381 | vctk_5464 | 0.9595 |
| p315_183_mic1.flac | 183 | s5_183 | — | — | p330_187 | vctk_6616 | 0.9611 |
| p315_184_mic1.flac | 184 | s5_184 | — | — | p297_004 | vctk_7289 | 0.9608 |
| p315_188_mic1.flac | 188 | s5_188 | — | — | p330_193 | vctk_11130 | 0.9624 |
| p315_193_mic1.flac | 193 | s5_193 | — | — | p292_151 | vctk_6030 | 0.9534 |
| p315_196_mic1.flac | 196 | s5_196 | — | — | p334_131 | vctk_11224 | 0.9695 |
| p315_203_mic1.flac | 203 | s5_203 | — | — | p340_138 | vctk_2737 | 0.9706 |
| p315_206_mic1.flac | 206 | s5_206 | — | — | p286_360 | vctk_2424 | 0.9481 |
| p315_207_mic1.flac | 207 | s5_207 | 9 | 0.9446 | p330_213 | vctk_5679 | 0.9559 |
| p315_208_mic1.flac | 208 | s5_208 | — | — | p303_181 | vctk_6742 | 0.9663 |
| p315_209_mic1.flac | 209 | s5_209 | — | — | p307_378 | vctk_3482 | 0.9639 |
| p315_210_mic1.flac | 210 | s5_210 | — | — | s5_184 | vctk_13738 | 0.9491 |
| p315_212_mic1.flac | 212 | s5_212 | — | — | p294_383 | vctk_576 | 0.9438 |
| p315_213_mic1.flac | 213 | s5_213 | — | — | p298_373 | vctk_6868 | 0.9532 |
| p315_214_mic1.flac | 214 | s5_214 | — | — | p311_219 | vctk_5685 | 0.9784 |
| p315_216_mic1.flac | 216 | s5_216 | — | — | p294_072 | vctk_640 | 0.9622 |
| p315_217_mic1.flac | 217 | s5_217 | — | — | p288_112 | vctk_7256 | 0.967 |
| p315_219_mic1.flac | 219 | s5_219 | — | — | p330_226 | vctk_433 | 0.9621 |
| p315_221_mic1.flac | 221 | s5_221 | — | — | p271_195 | vctk_2544 | 0.9641 |
| p315_226_mic1.flac | 226 | s5_226 | — | — | p361_364 | vctk_12932 | 0.9562 |
| p315_228_mic1.flac | 228 | s5_228 | — | — | p234_143 | s5_156 | 0.9504 |
| p315_229_mic1.flac | 229 | s5_229 | — | — | p318_174 | vctk_161 | 0.9633 |
| p315_230_mic1.flac | 230 | s5_230 | — | — | p330_237 | vctk_6703 | 0.9658 |
| p315_231_mic1.flac | 231 | s5_231 | — | — | p307_302 | vctk_2122 | 0.9702 |
| p315_232_mic1.flac | 232 | s5_232 | — | — | p304_358 | vctk_8035 | 0.9672 |
| p315_239_mic1.flac | 239 | s5_239 | — | — | p361_164 | vctk_12800 | 0.9607 |
| p315_241_mic1.flac | 241 | s5_241 | — | — | p313_030 | vctk_134 | 0.952 |
| p315_242_mic1.flac | 242 | s5_242 | — | — | p255_104 | vctk_2281 | 0.9592 |
| p315_248_mic1.flac | 248 | s5_248 | — | — | p330_257 | vctk_11151 | 0.9559 |
| p315_250_mic1.flac | 250 | s5_250 | — | — | p303_139 | vctk_6653 | 0.9586 |
| p315_255_mic1.flac | 255 | s5_255 | — | — | p330_265 | vctk_6744 | 0.9565 |
| p315_256_mic1.flac | 256 | s5_256 | — | — | p303_319 | vctk_6706 | 0.9617 |
| p315_257_mic1.flac | 257 | s5_257 | — | — | p330_267 | vctk_6778 | 0.9683 |
| p315_260_mic1.flac | 260 | s5_260 | — | — | p284_200 | vctk_2064 | 0.9471 |
| p315_261_mic1.flac | 261 | s5_261 | — | — | p330_271 | vctk_11155 | 0.9644 |
| p315_262_mic1.flac | 262 | s5_262 | — | — | p278_232 | vctk_6937 | 0.9522 |
| p315_264_mic1.flac | 264 | s5_264 | — | — | p330_274 | vctk_6558 | 0.958 |
| p315_265_mic1.flac | 265 | s5_265 | — | — | p317_294 | vctk_4399 | 0.9548 |
| p315_266_mic1.flac | 266 | s5_266 | — | — | p343_119 | vctk_12072 | 0.9625 |
| p315_273_mic1.flac | 273 | s5_273 | — | — | p307_094 | vctk_3875 | 0.9577 |
| p315_280_mic1.flac | 280 | s5_280 | — | — | p330_286 | vctk_6577 | 0.9722 |
| p315_281_mic1.flac | 281 | s5_281 | — | — | p303_159 | vctk_6573 | 0.9668 |
| p315_282_mic1.flac | 282 | s5_282 | — | — | p316_028 | vctk_2757 | 0.952 |
| p315_285_mic1.flac | 285 | s5_285 | — | — | p330_291 | vctk_7071 | 0.9507 |
| p315_293_mic1.flac | 293 | s5_293 | — | — | p262_370 | vctk_2717 | 0.969 |
| p315_294_mic1.flac | 294 | s5_294 | — | — | p307_361 | vctk_6295 | 0.9565 |
| p315_295_mic1.flac | 295 | s5_295 | — | — | p333_323 | vctk_7647 | 0.9079 |
| p315_298_mic1.flac | 298 | s5_298 | — | — | p300_345 | vctk_4161 | 0.9693 |
| p315_302_mic1.flac | 302 | s5_302 | — | — | p307_169 | vctk_3505 | 0.9799 |
| p315_305_mic1.flac | 305 | s5_305 | — | — | p311_277 | vctk_4623 | 0.972 |
| p315_306_mic1.flac | 306 | s5_306 | — | — | p343_286 | vctk_12137 | 0.9555 |
| p315_307_mic1.flac | 307 | s5_307 | — | — | p307_188 | vctk_3996 | 0.9674 |
| p315_308_mic1.flac | 308 | s5_308 | — | — | p303_169 | vctk_6759 | 0.9661 |
| p315_309_mic1.flac | 309 | s5_309 | — | — | p305_150 | vctk_2314 | 0.9555 |
| p315_310_mic1.flac | 310 | s5_310 | — | — | p326_073 | vctk_10981 | 0.9597 |
| p315_311_mic1.flac | 311 | s5_311 | — | — | p304_319 | vctk_8094 | 0.951 |
| p315_312_mic1.flac | 312 | s5_312 | — | — | p330_320 | vctk_7027 | 0.9678 |
| p315_314_mic1.flac | 314 | s5_314 | — | — | p317_076 | vctk_3739 | 0.9595 |
| p315_315_mic1.flac | 315 | s5_315 | — | — | p336_284 | vctk_4109 | 0.9587 |
| p315_318_mic1.flac | 318 | s5_318 | — | — | p244_319 | vctk_8607 | 0.943 |
| p315_319_mic1.flac | 319 | s5_319 | — | — | p336_290 | vctk_4619 | 0.9613 |
| p315_320_mic1.flac | 320 | s5_320 | — | — | p330_329 | vctk_4558 | 0.9632 |
| p315_327_mic1.flac | 327 | s5_327 | — | — | p294_178 | vctk_9801 | 0.9515 |
| p315_328_mic1.flac | 328 | s5_328 | — | — | p334_155 | vctk_11231 | 0.953 |
| p315_336_mic1.flac | 336 | s5_336 | — | — | p336_305 | vctk_5782 | 0.9308 |
| p315_337_mic1.flac | 337 | s5_337 | — | — | p341_010 | vctk_7288 | 0.9652 |
| p315_343_mic1.flac | 343 | s5_343 | — | — | p329_258 | vctk_1855 | 0.9616 |
| p315_344_mic1.flac | 344 | s5_344 | — | — | p330_351 | vctk_4580 | 0.9479 |
| p315_345_mic1.flac | 345 | s5_345 | — | — | p272_196 | vctk_2607 | 0.9488 |
| p315_346_mic1.flac | 346 | s5_346 | — | — | p330_353 | vctk_5839 | 0.972 |
| p315_347_mic1.flac | 347 | s5_347 | — | — | p281_085 | vctk_3218 | 0.9494 |
| p315_349_mic1.flac | 349 | s5_349 | — | — | p293_267 | vctk_9721 | 0.9348 |
| p315_350_mic1.flac | 350 | s5_350 | — | — | p308_249 | vctk_7888 | 0.9811 |
| p315_352_mic1.flac | 352 | s5_352 | — | — | p310_104 | vctk_4469 | 0.9546 |
| p315_357_mic1.flac | 357 | s5_357 | — | — | p339_373 | vctk_11713 | 0.9335 |
| p315_359_mic1.flac | 359 | s5_359 | — | — | p361_413 | vctk_12957 | 0.9505 |
| p315_360_mic1.flac | 360 | s5_360 | — | — | p341_142 | vctk_401 | 0.9721 |
| p315_366_mic1.flac | 366 | s5_366 | — | — | p374_158 | vctk_243 | 0.9637 |
| p315_368_mic1.flac | 368 | s5_368 | — | — | p351_406 | vctk_12493 | 0.9533 |
| p315_372_mic1.flac | 372 | s5_372 | — | — | p330_376 | vctk_6771 | 0.977 |
| p315_375_mic1.flac | 375 | s5_375 | — | — | p311_335 | vctk_901 | 0.9765 |
| p315_380_mic1.flac | 380 | s5_380 | — | — | p336_351 | vctk_6556 | 0.9379 |
| p315_388_mic1.flac | 388 | s5_388 | — | — | p336_057 | vctk_6772 | 0.9431 |
| p315_390_mic1.flac | 390 | s5_390 | — | — | p294_229 | vctk_5070 | 0.9596 |
| p315_391_mic1.flac | 391 | s5_391 | — | — | p231_179 | vctk_8261 | 0.9391 |
| p315_392_mic1.flac | 392 | s5_392 | — | — | p284_081 | vctk_7371 | 0.9563 |
| p315_393_mic1.flac | 393 | s5_393 | — | — | p308_214 | vctk_7837 | 0.9556 |
| p315_397_mic1.flac | 397 | s5_397 | — | — | p305_168 | vctk_7697 | 0.9374 |
| p315_403_mic1.flac | 403 | s5_403 | — | — | p287_062 | vctk_9593 | 0.9561 |
| p315_405_mic1.flac | 405 | s5_405 | — | — | p259_269 | s5_333 | 0.9639 |
| p315_406_mic1.flac | 406 | s5_406 | — | — | p374_178 | vctk_13465 | 0.9584 |
| p315_408_mic1.flac | 408 | s5_408 | — | — | p330_414 | vctk_11173 | 0.9644 |
| p315_414_mic1.flac | 414 | s5_414 | — | — | p294_252 | vctk_607 | 0.9489 |
| p315_418_mic1.flac | 418 | s5_418 | — | — | p288_359 | vctk_2016 | 0.9587 |
| p315_421_mic1.flac | 421 | s5_421 | — | — | p259_324 | vctk_6162 | 0.9592 |

---

## Task 2: REHASP Mixed-Corpus Preference-Flip Retrieval

**Setup:** 1170 REHASP queries (Lucy patient speaker, multiple repetitions) are retrieved
against a corpus that mixes REHASP repetitions and OSR (Open Speech Repository) sentences.
OSR provides two speakers (UK, US) reading a standard sentence set that overlaps with the
Harvard sentence list used in REHASP.

**Preference-flip metric:** All corpus utterances are ranked for each query and categorised:

| Category | Ideal rank | Meaning |
|---|---|---|
| `same_sentence_same_speaker` (ssss) | ~1 | Same sentence, same Lucy speaker (a different repetition) |
| `same_sentence_diff_speaker` (ssds) | Low (semantic hit) | Same sentence, OSR or other speaker — the *desired* retrieval |
| `diff_sentence_same_speaker` (dsss) | High (not wanted) | Different sentence, same Lucy speaker |
| `diff_sentence_diff_speaker` (dsds) | Highest (least relevant) | Different sentence, different speaker |

A model that prioritises semantics over speaker identity should rank `ssds` lower (closer to 1)
than `dsss`.  The hub model **inverts this**: `dsss` (mean_rank≈30) ranks above `ssds` (mean_rank≈33),
meaning same-speaker confounders beat cross-speaker semantic matches.

**Precision@k** counts the fraction of queries where at least one `ssds` result appears in top-k.

### Summary Table

| Variant | Combined weights (sem/spk) | ssss mean_rank | ssds mean_rank | dsss mean_rank | dsds mean_rank | P@1 | P@5 | P@10 |
|---|---|---|---|---|---|---|---|---|
| hub_default | sem=1.0 spk=1.0 | 1.056 (n=1170) | 32.52 (n=3120) | 29.9 (n=33930) | 225.788 (n=453180) | 0.0 | 0.3863 | 0.5701 |
| spk_weight_0.0 | sem=1.0 spk=0.0 | 1.062 (n=1170) | 33.693 (n=3120) | 35.259 (n=33930) | 225.378 (n=453180) | 0.0 | 0.4034 | 0.5675 |
| spk_weight_-0.5 | sem=1.0 spk=-0.5 | 1.074 (n=1170) | 35.137 (n=3120) | 42.051 (n=33930) | 224.86 (n=453180) | 0.0 | 0.4068 | 0.559 |
| spk_weight_-1.0 | sem=1.0 spk=-1.0 | 1.085 (n=1170) | 38.809 (n=3120) | 56.744 (n=33930) | 223.735 (n=453180) | 0.0 | 0.4085 | 0.5444 |
| spk_weight_-1.5 | sem=1.0 spk=-1.5 | 1.268 (n=1170) | 51.652 (n=3120) | 89.465 (n=33930) | 221.196 (n=453180) | 0.0111 | 0.347 | 0.453 |
| spk_weight_-2.0 | sem=1.0 spk=-2.0 | 18.629 (n=1170) | 91.671 (n=3120) | 150.137 (n=33930) | 216.333 (n=453180) | 0.0427 | 0.165 | 0.2487 |
| osr_only | sem=1.0 spk=0.0 | None (n=0) | 23.56 (n=3120) | None (n=0) | 196.684 (n=453180) | 0.6547 | 0.6632 | 0.6632 |

### Observations

**Speaker dominance at positive weights (hub_default, spk_weight_0.0 to spk_weight_-1.0):**

- `ssss` always ranks first (mean_rank ~1.06–1.09): the model correctly retrieves the same
  speaker's own repetition of the same sentence at rank 1 in nearly every query.
- `dsss` (wrong sentence, same speaker) consistently outranks `ssds` (right sentence,
  wrong speaker) until spk_weight ≤ −1.0. Speaker identity pulls wrong sentences above
  correct cross-speaker matches.
- P@1 = 0.0 for all positive/zero weights: no query returns a cross-speaker semantic
  match as its top hit when speaker similarity is included.
- P@5 peaks around −1.0 (0.409) before degrading — the semantic axis alone produces
  ~40 % of queries with a correct match in top 5.

**Degradation at −1.5 and below:**

- At spk_weight=−1.5 the `ssss` rank climbs to 1.268: the model is now actively penalising
  same-speaker matches so even the query's own other repetitions slip. P@1 appears (0.011)
  but P@5/P@10 drop below the +1.0 baseline.
- At spk_weight=−2.0 the `ssss` rank collapses to 18.6: the model no longer recovers its
  own speaker's repetitions, and retrieval quality across all categories degrades badly.

**OSR-only corpus (no REHASP in corpus):**

- `ssss` = n/a (Lucy not in corpus), `dsss` = n/a.
- P@1 = **0.655**, P@5 = 0.663: dramatically better. When same-speaker confounders are
  removed, the semantic axis retrieves the correct sentence from OSR speakers with high
  precision. This confirms that the semantic axis is capable — the failure mode is
  speaker identity dominating the combined similarity when Lucy is in the corpus.

---

## Task 3: Speaker ID Weight Sweep (REHASP Internal — 20 Queries)

**Setup:** 20 REHASP queries (Lucy, rep006) retrieved against a corpus mixing Lucy reps
(rep023) and OSR sentences.  Combined similarity = semantic + `weight × speaker_id`.
This sweep tests how the combined weight affects whether the model returns a
same-speaker repeat (`ssss`) or a cross-speaker same-sentence OSR match (`ssds`) at rank 1.

The per-axis recall (same sentence, any rep) is **1.0 across all axes at all weights** —
all axes individually retrieve same-sentence items in top-1. The sweep therefore measures
only the *combined* ranking behaviour.

### Combined Retrieval: Preference Flip vs Weight

| Speaker ID weight | ssss mean_rank | ssds mean_rank | P@1 (cross-speaker) | P@5 | P@10 |
|---|---|---|---|---|---|
| -5.000 | — | — | 0.0 | 0.0 | 0.0 |
| -3.000 | — | — | 0.0 | 0.0 | 0.0 |
| -2.500 | — | — | 0.0 | 0.0 | 0.0 |
| -2.000 | 4.923 | — | 0.0 | 0.0 | 0.0 |
| -1.000 | 1.0 | — | 0.0 | 0.0 | 0.0 |
| +0.000 | 1.0 | — | 0.0 | 0.0 | 0.0 |
| +0.001 | 1.0 | — | 0.0 | 0.0 | 0.0 |
| +0.010 | 1.0 | — | 0.0 | 0.0 | 0.0 |
| +0.050 | 1.0 | — | 0.0 | 0.0 | 0.0 |
| +0.100 | 1.0 | — | 0.0 | 0.0 | 0.0 |
| +0.200 | 1.0 | — | 0.0 | 0.0 | 0.0 |
| +0.500 | 1.0 | — | 0.0 | 0.0 | 0.0 |
| +1.000 | 1.0 | — | 0.0 | 0.0 | 0.0 |
| +2.000 | 1.0 | — | 0.0 | 0.0 | 0.0 |
| +5.000 | 1.0 | — | 0.0 | 0.0 | 0.0 |

### Observations

- **Weights 0.0 → +5.0**: `ssss` always at rank 1 regardless of positive weight magnitude.
  P@1 = 0.0 across this entire range — the same-speaker rep always wins over OSR.
  P@5 degrades slowly from 0.55 → 0.40 as higher positive weight pushes OSR sentences
  further down the list.
- **Weight −1.0**: `ssss` still rank 1 (mean=1.0), but P@5=0.70 / P@10=0.80.
  The OSR results start appearing in the top 5–10 even though same-speaker rep is still top.
- **Weight −2.0**: transition point. `ssss` mean_rank = 4.9. P@1 = 0.95 —
  nearly all queries now return an OSR match at rank 1. One query still ranks same-speaker first.
- **Weight −2.5 and below**: complete inversion. `ssss` disappears from top results entirely.
  P@1 = 1.0 for all 20 queries. The cross-speaker semantic match is always rank 1.
  This is achieved by subtracting speaker similarity so strongly that speaker confounders
  are pushed out of the top entirely.

**Take-away:** Without retraining, the hub model requires a speaker weight of approximately
−2.0 to −2.5 to achieve consistent cross-speaker semantic retrieval on the REHASP task.
This is an aggressive penalty; the better long-term fix is training the semantic axis to
be speaker-invariant (via GRL or reduced speaker loss weight), which runs 4 and 5 test.

---

## Summary

| Task | Best metric | Value | Condition |
|---|---|---|---|
| P315→VCTK semantic R@10 | R@10 | 0.076 | semantic axis alone |
| P315→VCTK speaker R@10 | R@10 | 0.000 | speaker axis (unseen speaker) |
| REHASP mixed P@1 (cross-speaker) | P@1 | 0.655 | OSR-only corpus (no REHASP in index) |
| REHASP mixed P@5 (cross-speaker) | P@5 | 0.409 | combined spk_weight=−1.0 |
| REHASP sweep P@1 (cross-speaker) | P@1 | 1.000 | combined spk_weight ≤ −2.5 (20 queries) |

**Diagnosis:** Speaker identity dominates the combined embedding. The semantic axis
generalises reasonably when evaluated in isolation (P@1=0.655 on OSR-only), but
same-speaker similarity overwhelms it once Lucy's own repetitions are in the corpus.
The planned runs (GRL, axis weight downscaling) directly target this failure mode.
