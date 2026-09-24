
> This paper investigates grapheme-to-phoneme conversion as a controlled setting for target-oriented multitask learning. Rather than adding auxiliary tasks simply because they are available, we select tasks that correspond to pronunciation-relevant structure: phoneset mappings, language-origin evidence, compound decomposition, morphosyntactic information, and structured views of the target pronunciation. The empirical setting is Swedish text-to-speech, where pronunciation lexicons contain both native Swedish material and foreign-origin words, including compounds whose components may provide conflicting orthographic evidence. We first establish finite-state baselines for monolingual, multilingual, and language-tagged G2P. We then use these baselines to motivate experiments with prompted sequence-to-sequence models, asking whether auxiliary tasks improve the model's ability to learn multiple graphone systems and to select appropriate mappings when origin-language prompts are absent.

Grapheme-to-phoneme (G2P) conversion is often described as a practical subtask in text-to-speech (TTS): given a written word, produce the phonetic or phonological representation required by the synthesis system. In practice, however, G2P is not only a mapping from letters to phones. A useful pronunciation component may need information about morphology, lexical category, stress, language origin, compound structure, and the phoneset available to the target voice or system.

This makes G2P a useful testbed for a broader question in multitask learning. Early neural NLP work made task sharing explicit, using auxiliary linguistic tasks to shape representations for a target task. More recent large language models and instruction-tuned systems also contain multitask behaviour, but the contribution of individual tasks is often obscured by the scale and heterogeneity of the training mixture. In this work, we return to a smaller and more controlled setting: auxiliary tasks are deliberately selected because each exposes structure that a pronunciation engineer might otherwise compute explicitly in a text-processing pipeline.

The concrete domain is Swedish TTS. Swedish pronunciation resources contain not only native vocabulary, but also names, loanwords, and compounds with foreign-origin components. A word may be tagged as Swedish for practical lexical purposes while still containing orthographic evidence associated with another language. For example, a compound may be Swedish as a lexical item but contain a component whose spelling suggests a different set of grapheme-to-phone correspondences. Conversely, some words require explicit control: in an SSML-marked TTS pipeline, a system may be told that a word is English-origin but should be pronounced using a Swedish target phoneset.

The central claim of this paper is therefore not simply that multitask learning can improve G2P. The claim to be tested is more specific: auxiliary tasks should help when they are chosen for their hypothesized relationship to the target pronunciation mapping. We consider tasks such as phoneset-to-phoneset mapping, language-origin identification, compound decomposition, compound-origin identification, morphosyntactic description, stress, and phonetic syllabification. Some are external analyses, some are generated from the same lexical resources, and some are structured views of the target pronunciation. Their common role is to expose pronunciation-relevant structure.

We ask two main research questions. First, can deliberately curated auxiliary tasks contribute meaningfully to a multitask G2P model, and do they help in the ways predicted by their linguistic role? Second, can such a model learn to select appropriate grapheme-to-phone mappings when the target language is specified but the origin language is not? The second question is intended as a more constrained alternative to broad claims about zero-shot G2P: the goal is not to produce arbitrary IPA strings for an unseen language, but to test whether implicit orthographic evidence and explicit auxiliary supervision can support controlled pronunciation behaviour.

\section{Background}

\subsection{G2P in TTS pipelines}

Pronunciation dictionaries are never complete in a living language. New words, names, compounds, product names, and foreign-origin expressions appear continuously, so TTS systems require a way to generate pronunciations for words not already present in a lexicon. G2P conversion is the usual name for this component, but the target representation is determined by the needs of the synthesis system: it may include phoneme identity, stress, syllable boundaries, or other prosodically relevant information.

; Pronunciation dictionaries are often hard to find, and are never complete in a living language. For this reason, the task of grapheme to phoneme (G2P) conversion exists: to automate the production of a phonetic representation of words not in the dictionary.

This pipeline setting matters because useful pronunciation decisions are often conditioned on information that is not visible in the raw character string alone. English homographs provide a familiar example: \textit{record} is pronounced differently depending on whether it is a noun or a verb. A text-processing pipeline that supplies part-of-speech information can therefore disambiguate the G2P target. Similar interactions appear in other languages through morphology, compounds, named entities, and lexical origin.

Grapheme-to-phoneme (G2P) conversion is typically considered in terms of a single language, but languages do not exist in isolation: words, names in particular, from other languages make frequent appearances.

Recent work in TTS has explored increasingly end-to-end systems that reduce or remove explicit phonetic transcription. Such systems are attractive, but they do not remove the practical need for pronunciation control. When text is produced for accessibility or other high-reliability settings, the ability to specify or repair pronunciations remains important, especially for names and foreign words. Standards such as Speech Synthesis Markup Language (SSML) \cite{w3c:ssml11} reflect this need by allowing language to be specified at several levels of a document, including paragraphs, sentences, tokens, and words.

While monolingual G2P systems perform well within their target language, they struggle to adapt to words borrowed from other languages. Unassimilated loanwords follow the rules of their source language, and can even have a negative impact on the source language if too many are introduced into the training data of a G2P system, by introducing confusion. For instance, even state-of-the-art end-to-end TTS systems such as DiTTo-TTS \cite{lee2025dittotts} occasionally mispronounce words from well-resourced languages like Polish, despite large multilingual datasets.

\subsection{Classical and statistical G2P}

A further complication arises from how text is marked up for synthesis. Standards such as the Speech Synthesis Markup Language (SSML) \cite{w3c:ssml11} permit the explicit specification of language on multiple levels—paragraph, sentence, or word--but these annotations are only useful if the G2P component itself can interpret them correctly.

Traditional G2P systems range from hand-written rewrite rules to data-driven models learned from pronunciation lexicons. Rule-based systems make intermediate decisions explicit: a pronunciation engineer may encode grapheme classes, morphological decomposition, stress rules, or exception handling. Statistical systems replace many hand-written decisions with learned correspondences, but the underlying problem remains structured. Joint-sequence graphone models, including WFST implementations such as Phonetisaurus \cite{NOVAK_MINEMATSU_HIROSE_2016}, learn correspondences between grapheme sequences and phone sequences and provide a strong classical baseline.

% TODO: Revise this paragraph so it states the intended direction from junk.tex, not only the Phonetisaurus baseline.

In this work, we explore the extent to which multilingual or mixed-language G2P training improves—or degrades—performance across languages. Using the Braxen lexicon \cite{tannander-edlund-2025-braxen}, a Swedish TTS resource containing extensive foreign-language entries, we compare monolingual, multilingual, and untagged models to quantify how cross-language exposure affects phoneme error rate (PER). Our goal is to better understand how multilingual data and language tagging influence pronunciation accuracy in practical TTS systems.

For foreign-origin words, one common engineering strategy is to run a G2P system for the source language and then map the resulting phones into the target system's phoneset. This has the advantage of modularity, but it requires a separate G2P component for each source language as well as phoneset mappings between languages. An alternative is to train a single model that can use language information directly and produce output in the desired target representation.

\subsection{Multitask learning and task curation}

% In this work, we attempt to use a combination of data selection and task prompting to guide a (by)T5 model, which has already been shown to perform well at the G2P task, towards being able to produce phonetic transcriptions of words from one language using the phoneset of another with minimal intervention.

% "minimal intervention", yuck. It's G2P, not a UN peacekeeping mission

% TODO: This seems like the intended contribution statement, but it needs revision before becoming running text.

Multitask learning is often motivated by the idea that related tasks can provide useful inductive bias for one another. In NLP, early neural multitask work explicitly selected linguistic tasks expected to support particular target tasks, rather than treating all available tasks as interchangeable. Later instruction-tuned language models cast many tasks into a shared text-to-text format, but at a scale where it is difficult to isolate why a particular auxiliary task helps.

\section{Background \& related work}

The present work is closer to target-oriented auxiliary-task selection. The auxiliary tasks are not selected simply because they are available; they are selected because each corresponds to an intermediate analysis, decision point, or structured target view relevant to pronunciation generation. This distinction is important because several tasks may be derived from the same lexical source. They are not necessarily independent sources of information. Instead, they expose different factorizations of the pronunciation problem.

Recent work in text to speech (TTS), such as \citet{gao2023e3tts}, attempts to move towards ``end to end'' synthesis, without the use of a phonetic transcription. As impressive as they are, they do not always pronounce everything correctly: on the samples page\footnote{\url{https://ditto-tts.github.io/}} for DiTTo-TTS \cite{lee2025dittotts}, for example, in the second Polish example the word ``niektórzy'' is mispronounced: Polish has quite a shallow orthography and a relatively large amount of publicly available data.

In this sense, the paper is not primarily a claim about replacing classical G2P rules with neural prompting. It asks whether the kinds of intermediate structure used in classical pronunciation pipelines can be exposed as supervised tasks for a sequence-to-sequence model. G2P supplies the measurable target task, while task selection is the main object of study.

% SSML etc.?

Speech Synthesis Markup Language (SSML) \cite{w3c:ssml11}, the most widely used standard for representing to a TTS engine how a piece of text should be read, allows language to be specified on multiple levels, including paragraph, sentence, token, and word. This degree of control over generated speech is desirable in many contexts, particularly in contexts where the end-users rely on a spoken form, such as those with print difficulties.

\subsection{Prompted G2P and controlled mapping selection}

One typical option for handling words of foreign origin is to perform a phoneset mapping, where the output of a G2P system for the other language is mapped to that of the current language. This requires a separate G2P component for each language individually, in addition to the mapping, which increases the complexity of the text processing pipeline.

Text-to-text models make it straightforward to represent many tasks through prompts. A basic monolingual G2P instance can be written as \texttt{g2p|lang=sv}, while a foreign-origin pronunciation task can be written as \texttt{g2p|lang=sv|from=en}. Phoneset mapping can likewise be represented directly as \texttt{ipa2ipa|from=en|to=sv}. These prompts allow explicit control, but they also make it possible to test what happens when that control is removed or contradicted.

% In a rule-based paradigm, there are two common options for adapting a system designed for one language to the phoneset of another: either by editing the G2P rules themselves so that they output phones appropriate for the second language, or by explicitly mapping the phones it outputs to those of the other language.

This distinction is central to the proposed experiments. A model as expressive as T5 may learn that particular characters or character sequences provide evidence for particular graphone systems. The question is not whether language identification should be inserted into a G2P pipeline as a hard prerequisite. Rather, we ask whether observable orthographic signals, auxiliary language-origin supervision, and explicit prompts interact in ways that improve mapping selection. This can be tested by comparing origin-prompted, origin-free, and conflicting-prompt conditions.

% Pivots (or bridge languages) have long been used in machine translation to augment lexicons in rule-based machine translation \cite{forcada2011apertium}, or phrase tables in statistical machine translation \cite{kumar-etal-2007-improving,CHEN08.733}.

\subsection{Swedish compounds and auxiliary structure}

% Google's Neural Machine Translation \cite{johnson-etal-2017-googles} learned an implicit pivot through the use of a single set of shared modules with a prepended token to signify target language, the alignments tended to converge in a space akin to an interlingua. In addition to improving performance among less resourced language pairs, they found that ``zero-shot'' translation became possible for language pairs that weren't seen in training, but for which a pivot was available.

For Swedish, compound decomposition is a particularly important auxiliary task. An unknown word may simply be a compound of known words, but decomposition is also pronunciation-relevant in its own right. The sequence \textit{sch}, for example, can represent a Swedish trigraph, as in \textit{dusch}, but it may also cross a compound boundary. In \textit{potatischips}, the relevant decomposition is \textit{potatis} + \textit{chips}; treating the surface string as containing the same trigraph would give the wrong evidence to the G2P model.

% Need to say something about T5, because it's a key step

% That is: the artificial token is a precursor to the kinds of prompts used in T5, but it broadly equivalent to using the same kind of token when concatenating FSTs.

Compounds may also contain components whose lexical status and pronunciation evidence do not align cleanly. A form such as \textit{tennisracket} may be Swedish enough to be treated as a Swedish lexical item, while the spelling of \textit{racket} still raises a pronunciation question better represented by a Swedish-oriented normalization such as \textit{rackett} than by a simple origin label. Such cases motivate auxiliary tasks that distinguish compound decomposition, component-origin evidence, and pronunciation-oriented normalization.

% A non-goal for this work is the kind of ``zero shot'' G2P claimed in \cite{zhu22_interspeech}. Although it does seem, as they claim, to be a useful basis for future G2P models for under-resourced languages, we consider the concrete case of G2P as a component of a text-to-speech system which must, as a result of human limitations, be bounded to the phoneme inventory of the speaker. Our intention, rather, is to prime the G2P system through data- and task-selection to produce something that is within the speaker's range, and hopefully broadly representative of how a speaker of one language pronounces loanwords from another.

% TODO: This non-goal is central to the intended framing, but should be cleaned up and moved into running text.

% TODO: Make the distance from broad ``zero-shot G2P'' explicit: the paper is not trying to produce arbitrary IPA for an unseen language, but to solve a TTS-bounded pronunciation problem where a foreign word is rendered inside the target speaker's usable phoneset.

The NST lexicons make some of this information explicit by providing compound decompositions. These decompositions also preserve linkers where this is useful for pronunciation: for example, \textit{Östermalmstorg} is better treated as \textit{Östermalms} + \textit{torg} than as \textit{Östermalm} + \textit{storg}. This is not only a lexical-analysis issue, but a pronunciation issue, since incorrect boundaries can create misleading grapheme-phone evidence.

\section{Method}

% OK cool. Now add placeholders in the method. Minimally two sentences per process step you need to go through to get this done. Language/writing quality does not matter at all though.

\subsection{Data}

As such, some of the vocabulary in Braxen reflects the realities of text ``in the wild'': it includes a large selection of words from other languages. From our perspective, the detail that is most interesting is that it includes language tags for these foreign words.

%As the primary mission of MTM is not

As MTM is primarily concerned with the practical matter of producing spoken forms of written material, rather than lexicography for its own sake, a number of errors and inconsistencies appear in the data in relation to the tagging of languages: the languages are tagged using a three-letter language code: in the case of Latin (`lat'), this is a single character away from the character codes of Latvian (`lav') and Lithuanian (`Lit'): consequently, we find a small number of words from those languages tagged as Latin.

By far the most represented foreign language in Braxen is English.

English is taught as a mandatory subject in schools across Sweden \cite{norrby2014english}: according to the 2024 Eurobarometer report, 90\% of Swedes report feeling comfortably conversational in English \cite{eurobarometer2024languages}, and according to the EF English Proficiency Index 2024, Swedes rank 4th in the world in terms of their English skills \cite{efepi2024}.

In addition to being the primary languages of neighbouring countries, Danish, Swedish, and Norwegian form a dialect continuum, and there is some degree of mutual intelligibility between their speakers. As such, Danish and Norwegian words are among the best represented in Braxen. However, as reflects the nature of text in the wild, many of these words are spelled using Swedish letters: the Norwegian and Danish alphabets have the letters `æ' and `ø' where Swedish has `ä' and `ö', respectively, with `æ' additionally rewritten as `ae'.

% In addition to the part of speech information also contained in Braxen, the NST lexicons include information about named entities, which is quite detailed in terms of category and subcategory. It also contains compound information, which is useful to the grapheme-to-phoneme process: the letter sequence `sch' is a trigraph representing a single sound in `dusch' (\textit{shower}), but represents `s' and `ch' in `potatischips' (\textit{potato chips}), for example.

% Compounds in NST are split into their composite parts, including the linker: while it might be more correct in terms of compounds to think of the linking `s' as a separate entity, rather than treating it as part of the noun (usually, making it identical to the genitive form), for the purposes of syllabification, it is preferable to keep this `s' attached to the correct noun: e.g., ``Östermalmstorg'' ought to be syllabified in terms of ``Östermalms'' $+$ ''torg'', not ``Östermalm'' $+$ ``*storg''.

% This particular example demonstrates a limitation with our method: ``Östermalm'' itself is a compound: ``öster'' + ``malm'' (\textit{east suburb}), but this cannot be recovered from the Braxen data, as the transcription lacks a compound marker. This is, however, likely to be preferable for other purposes: none of the compounds that contain it refer to anything other than the place.

% Although the named entity information is less information for our purposes, we choose to maintain it; our conversion of the NST data is complete, though modernised. First, the data has been converted to Unicode; transcriptions, which occupied a fixed number of fields in the original data, have been converted to a list structure; finally, in addition to maintaining the original transcription, we also add an IPA conversion: although the transcriptions are claimed to use SAMPA, in reality the original transcriptions are rather loosely based on SAMPA.

% TODO: Decide whether NST belongs in the same paper, an appendix, or a later extension.

\subsection{Finite-state baseline}

% TODO: Baseline method from lrec2026-example.tex starts here.

As a first pass, we attempted to find misclassified words by checking for characters that are typical of the language each word was tagged as belonging to. For instance, Swedish words rarely contain the letters `ç' or `ñ', whereas these are common in French and Spanish, respectively. We compiled a list of such characteristic letters for each language represented in Braxen and used it to flag cases where the spelling and language tag appeared inconsistent. While this simple heuristic highlighted a number of misclassified words, it generated too much noise: very few of the characters in the languages represented in Braxen are truly unique to that language alone, and even when that is the case, there are too many instances of words with incorrect accents, such as `\`a' instead of `\'a' in Spanish.

We concentrated our efforts by selecting only languages with 1,000 or more entries; as a replacement form of filtering, we passed each candidate word to a spelling checker, selecting only those that were marked as correct. One clear limitation of this approach is for languages whose script is not Latin: the majority of words classified as Russian in Braxen are transliterated, which reduced the initial number of entries from 1,320 to just 7.

Of the initial set of languages, only English, Danish, French, Spanish, Italian, and Latin were retained.