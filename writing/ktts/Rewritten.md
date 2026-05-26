
Classical text-to-speech (TTS) systems typically incorporate a great deal of linguistic information, primarily through text processing pipelines: part-of-speech tagging as part of homograph disambiguation; text normalisation for converting non-standard words--numbers, abbreviations, etc.; pronunciation lexicons and grapheme-to-phoneme (G2P) conversion for generating pronunciations from words; phonological rules for adapting the pronunciations of individual words to more naturally flow together; duration models [etc]

These individual components often require a depth of domain-specific knowledge that can vary both by language and by domain. Even when taken as off-the-shelf components, an expertly designed system can be inexpertly applied: one common example is the use of the CMU pronunciation dictionary (CMUdict) as a source of ground truth pronunciations for all of English, where it in fact represents a single dialect of American English.

Recent trends in TTS have therefore tended towards "end to end" systems, which attempt to learn a direct mapping from input text to output audio.

[Trade-off: shitty pronunciations]

[Expert-designed systems inexpertly applied -- for example, CMUdict is often treated as a source of ground truth for all of English, where in reality it represents one dialect of American English]


[It is not unheard of for a system to be described as "end to end" while incorporating a traditional text processing pipeline with little or no acknowledgement.]

