Word boundary adaptation for pronunciation dictionaries

Typical pronunciation dictionaries are written to model the word in isolation, but languages typically, if not always, have Sandhi at word boundaries: assimilation, insertion, deletion, etc. For example, in Swedish ‘jag har inte det’ in fast speech becomes something like ‘ja-ante de’. The aim of this project is to create a more accurate pronunciation dictionary by taking Sandhi into account, by applying all possible phonological changes, and filtering them using acoustic evidence.

Dialectal adaptation of pronunciation dictionaries

Pronunciation dictionaries are hard to find, and typically only cater to a single dialect, which does not always match the data: for example, some of our researchers are using voice data from an Irish speaker of English, but using an American pronunciation dictionary which reflects the Mary/marry/merry merger, which the speaker does not have. TTS models trained on this mismatched data therefore learn an artificially ambiguous set of sounds.