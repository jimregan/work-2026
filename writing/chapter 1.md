Phones and phonemes often confused

Phones are a better modeling unit

Trend towards "end to end" systems does not need to mean giving up the advantages of using phones; their use can simply be moved elsewhere.
 - in asr, as a modeling proxy
 - in tts, as a means of evaluation

TTS evaluation weaknesses:
 - "naturalness" is poorly defined and heavily entangled
 - ASR/WER tells us nothing of intelligibility - the goal of ASR is to maximally accept, regardless of correctness
... This line of thought leads me to think about phonetic recognition as a means of automatic validation, but that's going to be difficult

Tacotron took out the phones, Keith Ito put them back

Relatively little progress in G2P relative to advances in TTS