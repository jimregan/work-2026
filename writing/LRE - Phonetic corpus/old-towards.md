**Index Terms**: speech recognition, parliamentary data, broadcast data,
archival recordings

# Introduction

The availability of parliamentary data has been of benefit to the
development of ASR systems: for many languages, it is often the only
source of transcribed speech in sufficient quantity of speech and
diversity of speakers to train an ASR
system [@solberg-ortiz-2022-norwegian; @nouza2022norwegian; @helgadóttir2017; @mansikkaniemi17_interspeech].

Although their use as a source of training data for ASR systems is well
established, parliamentary recordings are inherently interesting in
their own right, for several reasons: they typically cover a large time
span, making them good sources for investigation into diachronic changes
in the language.

Parliamentary speeches can often bridge the gap between read speech and
spontaneous speech: although the speaker typically has a prepared script
to read from, they may diverge from it in reaction to another speech or
other events in the same session. Parliamentary representatives are
typically drawn from all regions of a country, yielding recordings of
dialects that might not otherwise be recorded in large amounts, and the
variety of topics covered means that in addition to relatively
unemotional speech, there can be speeches full of emotion, or
affectations of emotion for rhetorical purposes.

The availability of transcripts of speeches is what makes parliamentary
recordings attractive for ASR purposes; in many cases, however, the
transcripts are limited to scheduled speeches that were filed in
advance: there is no guarantee that the transcript reflects where a
speaker went off script, nor is there typically a transcript of
interactions outside of that script. In addition to the recordings
within the parliament itself, there are quite often additional
recordings from public meetings of working groups on various topics that
are not transcribed, and represent a more fully spontaneous source of
speech. The lack of text from these sessions makes them unsearchable,
leading to a gap in the public record of proceedings.

In this article, we investigate ASR aimed specifically at parliamentary
speech, with a view towards other limited domain archives, such as news
broadcasts, which share similar properties in terms of a finite number
of speakers in similar or equivalent speaking conditions, where the lack
of transcripts reflects a gap in the public record and imposes
limitations on the scope of possible research in the digital humanities.

## SweTerror

## \[Anonymised Project\]

The current work is carried out in the context of the SweTerror
project [@edlund2022sweterror], a large-scale interdisciplinary
investigation into Swedish parliamentary discourse on terrorism, where
addressing the gaps in the record is crucial in building an accurate
representation of that discourse. The current work is carried out in the
context of the \[Anonymous project\], a large-scale interdisciplinary
investigation into Swedish parliamentary discourse on \[potentially
revealing topic\], where addressing the gaps in the record is crucial in
building an accurate representation of that discourse.

The data produced by the Swedish parliament in the last decade alone
amounts to almost 6,000 hours of video recordings: manually transcribing
this amount of data is infeasible. Transcripts of speeches are
available, but not of whole recordings, and in many cases, even a number
of speeches are untranscribed. Of the 9,372 video files in our data,
2,864 had no accompanying transcription of any amount.

## Research aims

The primary aim of our research is to develop bespoke ASR systems for
transcribing archival data of broad interest within the sphere of
digital humanities research, and, more specifically, to facilitate our
own research within speech science.

We focus here on parliamentary data because we have access to a large
amount of transcribed text to accompany the audio; in other domains this
will not be available, or available only indirectly and in part--in the
domain of broadcast news, for example, we can assume that the news as
broadcast on television will broadly match that as reported in print to
enough of an extent that we can extract at least direct quotations.

There may be a possibility of getting access to the autocue text read by
the newsreader going forward, but we have no guarantees, and when
considering the large amounts of data already in the archive, we need to
consider the following questions:

- Can the intersection of two ASR systems be enough as is to develop an
  adequate bespoke ASR system?

- Is the potential improvement to be gained from collecting and aligning
  print news sufficient to justify expending the amount of effort
  required?

We are fortunate in the scope of this work in two aspects: high quality,
open-licensed models for Swedish ASR are readily available, and
parliamentary speeches are often accompanied by a text transcript.
Another project aimed at broadcast news archives has little or no
transcribed data, although there is the possibility of locating a
sufficient quantity of matching text within the equivalent print
archives.

In other planned projects for the minority languages of Sweden, there
are typically no available models, and little or no transcribed speech,
but there is interest in the linguistic communities around these
languages in sharing the workload, if an estimate of the amount of work
required can be provided. The minority languages of Sweden have
representation in either or both public television and radio, where
summaries of news items typically contain a sentence or two of
transcribed speech.

In addition to the purely technological questions, we also consider the
question of how much time and effort will be involved in obtaining human
transcriptions. In the case of Swedish, professional transcription
services are readily available; for the other languages of Sweden, we
cannot rely on the availability of professionals, but we can use the
times as a minimum bound for our expectations of non-professionals.

It is intended that this work be used as a yardstick for these planned
projects, as it represents an approximation of ideal conditions. The
experiences here will determine our approach towards the projects where
conditions are less favourable, and serve towards directing our efforts
where they are best placed.

# Data {#sec:data}

The training data is drawn from the proceedings of the Swedish
parliament (Riksdag) over an almost 10-year period, from September 2012
to January 2022. The public API was queried for items containing video,
and the API results for each video was stored.

The API response contains metadata, such as the date of the session, the
URI of the video, and, in many cases, broadly timed information about
individual speeches within the session, containing the name of the
speaker, their political party, and a transcript of the speech. In
total, 5925 hours of video data were downloaded.

A further decade of material is available through the API, recorded
under similar conditions, which we have also downloaded but have not yet
begun to process. Recordings are available for an earlier 50 years of
parliamentary debates, albeit with decreasing amounts of accompanying
text.

## Automatic transcription

The videos were automatically transcribed using two ASR systems:
Whisper [@whisper2022] and a wav2vec2-based system [@baevski2020wav2vec]
using a model pre-trained on over 10,000 hours of Swedish
speech [@kbwav2vec], which we will refer to as KB-wav2vec, to avoid
confusion with the model architecture and the models we fine tuned on
the same base model.

Whisper has gathered a lot of attention, partly because of its accuracy,
its multilingual capabilities, its ability to translate to English, and
its ability to generate capitalised and punctuated output. The
KB-wav2vec model, on the other hand, is still the most accurate model
available for Swedish when capitalisation and punctation are removed.

Whisper, for all its merits, suffers from a number of drawbacks, most
likely due to the use of large quantities of Youtube subtitles in its
training. In one of our sample files, which began with 5 minutes of
silence, Whisper output the phrase "Tack till mina supporters via
www.patreon.com" ("Thanks to my supporters via www.patreon.com"); while
Whisper was, in at least one case, able to override the language setting
when presented with a video in English, which it then transcribed as
English, in at least one other case, it translated the English audio to
Swedish--an impressive feat, as it was only directly trained to
translate *to* English.

The texts were aligned using scripts based on those used in the creation
of Librispeech [@panayotov2015librispeech], with adaptations to operate
directly on the output from the Riksdag API. It is intended that this
will form part of a pipeline for continuous processing of new videos as
they are made available through the API. The source[^1] code is
available under the Apache Public License.

Two separate sets of alignments were computed: one based on the output
of KB-wav2vec and the official transcripts, one based on the
intersection of both ASR systems. We also realigned the output of
KB-wav2vec and the official transcripts, by first passing the
transcripts through a text normalisation system: we ran a new
recognition pass employing biased 3-gram language models derived from
the normalised text, and performed alignment with the normalised text.

The output of KB-wav2vec was used in both cases because it is possible
to get reliable word level timing information: Whisper uses a prediction
model to estimate timings that are only available at the utterance
(sentence) level, and are unpredictable in their granularity: some files
will have utterance timings with sub-second timings, others with whole
seconds only. Word-level timings are important both for our further
processing the data, and for the ability to search the data itself: the
downstream researchers need not only to validate that the word was
spoken as transcribed, but the sentiment attached to the word, which is
more accurately determined from the manner in which it was spoken.

## Test and validation sets

For testing and validation, a set of speakers were selected at random,
balanced for gender: 8 men and women for each set. Two continuous
segments of speech lasting no less than two minutes was selected for
each speaker. A list of the speakers is presented in
table [2](#tab:test_val){reference-type="ref" reference="tab:test_val"}.

The segments were professionally transcribed independently by three
transcribers. The first transcriber prepared two sets of transcriptions,
for both two-minute segments, while the other transcribers prepared only
the first set of segments.

In addition to the text transcription, these subsets are additionally
annotated with markings to represent other acoustic events--such as lip
smacks, breaths, and coughs--with false starts and other partially
articulated words transcribed to the extent possible, and marked. No
express guidelines were given in relation to the expression of numbers,
and the annotators followed their own judgement in writing them, with
all three electing to spell out some, while using digits for others.

As part of the transcription process, a record was kept of the amount of
time spent on each annotation. The times are summarised in
table [1](#tab:transcription_time){reference-type="ref"
reference="tab:transcription_time"}; the average time per minute of
recorded audio, taking transcription and quality assurance into account,
is 9.27 minutes.

The test and validation sets were not designed for ASR as it is commonly
understood, and have phonetically-motivated transcriptions for some high
frequency Swedish words that better reflect their quickly spoken
pronunciations. Word Error Rate (WER) calculated on the outputs of
conventional ASR systems--all of those mentioned in this work--can be
expected to incur a penalty of up to $3.62$: as this can be expected to
affect all systems equally, we have not attempted to adjust for it.

::: {#tab:transcription_time}
  ID     Transcription   Quality assurance
  ---- --------------- -------------------
  A1               532                  95
  A2               202                  42
  A3               264                  52

  : Times spent by annotator on transcription and quality control of
  approx. 32 minutes of audio (64 in the case of A1). All times given in
  minutes.
:::

## Text normalisation

Text normalisation was performed using NVIDIA's NeMo text processing
library [@zhang21ja_interspeech][^2], a system based
on [@ebden_sproat_2015], to which we contributed support for Swedish.
Text normalisation was performed using a system based
on [@ebden_sproat_2015]. Aside from the expansion of numbers and
abbreviations, a set of approximately 50 words was added that were
harvested from a test pass of the aligner over the first 10 files of the
output from the aligner, which represented alternate pronunciations or
mispronunciations, alternate spellings of surnames, "word" equivalents
of spoken acronyms (e.g., "esvete" for "SVT"), or Swedish phonetic
transcriptions of non-Swedish words ("bolonja" for "Bologna").

::: {#tab:test_val}
  ID   Name                      Split         Gender
  ---- ------------------------- ------------ --------
  01   Jörgen Hellman            Test            M
  02   Agneta Gille              Validation      F
  03   Amir Adan                 Test            M
  04   Teresa Carvalho           Test            F
  05   Kerstin Nilsson           Validation      F
  06   Niclas Malmberg           Validation      M
  07   Carina Ståhl Herrstedt    Test            F
  08   Vasiliki Tsouplaki        Validation      F
  09   Cecilie Tenfjord Toftby   Validation      F
  10   Ann-Britt Åsebol          Test            F
  11   Karin Nilsson             Test            F
  12   Ingemar Nilsson           Test            M
  13   Mats Nordberg             Test            M
  14   Ulrika Jörgensen          Test            F
  15   Aylin Fazelian            Validation      F
  16   Björn Wiechel             Validation      M
  17   Sedat Dogru               Validation      M
  18   Oskar Öholm               Test            M
  19   Eva Lohman                Validation      F
  20   Karin Granbom Ellison     Test            F
  21   Åsa Karlsson              Validation      F
  22   Yilmaz Kerimo             Validation      M
  23   Aphram Melki              Test            M
  24   Yasmine Bladelius         Test            F
  25   Désirée Liljevall         Validation      F
  26   Erik Slottner             Validation      M
  27   Gustav Nilsson            Validation      M
  28   Linda Wemmert             Test            F
  29   Mats Sander               Validation      M
  30   Arin Karapet              Validation      M
  31   Daniel Andersson          Test            M
  32   David Josefsson           Test            M

  : Speakers in the test and validations splits
:::

# Experiment

Our primary concern was in how the data selection affected the quality
of the model; consequently, we used the same hyperparameters across all
training instances.

Fine tuning was performed using 8 NVIDIA GeForce RTX 3090 GPUs. Fine
tuning took approximately 5.8 hours to reach 12,000 steps; the models
were allowed to continue to fine tune beyond that point, but none
reached a better WER beyond that point.

## Base model

The base model we used is the KB-wav2vec base model described
in [@kbwav2vec], which the authors kindly shared with us. This base
model was pretrained on 10,000 hours of Swedish speech, selected to
represent a broad range of Swedish accents.

## Fine tuning

The base model was fine tuned by adding a layer on top of the model,
trained to classify into the number of characters in the vocabulary,
representing the letters of the Swedish alphabet along with two
punctuation characters necessary for orthographic conventions, and a
space character. The classifier was trained using Connectionist Temporal
Classification (CTC) loss [@graves2006ctc]. Fine tuning was performed
using the Fairseq toolkit.

All resulting models have been exported for use within the Huggingface
Transformers library, and made available via the Huggingface hub, where
full training metrics can be viewed. The configuration file for each
model is included in the repository.

# Results

The results of our experiments are summarised in
table [3](#tab:wer){reference-type="ref" reference="tab:wer"}, along
with results from state-of-the-art systems, for comparison.

::: {#tab:wer}
  Model                                    WER
  ---------------------------------------- -------
  Whisper small                            25.47
  Whisper large v2                         14.38
  KB-wav2vec                               12.87
  100h, whisper/wav2vec intersection[^3]   16.38
  200h, whisper/wav2vec intersection[^4]   16.38
  100h, wav2vec/transcripts, no LM         14.44
  49h, wav2vec/transcripts, LM, norm       13.98

  : WER results
:::

KB-wav2vec outperforms our models by quite a margin--this is
unsurprising, as theirs was fine tuned on a larger dataset that was
constructed to be phonetically diverse, with a range of dialects.

The most surprising result is that doubling the size of the training
data for the system using the intersection of two ASR systems did not
have any effect on the test data (there had been a small improvement on
the validation set during training). This may be explained by the nature
of the intersection, in that it is the lowest common denominator of two
systems. We have 1,186 hours in total of similar data: the increase in
the scale of the data may be enough by itself to boost the performance
of the system.

Another option might be to add the output of a third system, and select
where two or more agree: this would at least be more robust to the
idiosyncrasies of each individual system.

The relative increase gained by matching ASR output with the official
transcript would seem to suggest that the intersection of the ASR
systems included common errors. The KB-wav2vec model already produces
very high quality output, and the mismatches are quite often not due to
recognition error, but due to the speaker having changed something while
speaking.

The relatively small improvement from adding a biased language model and
normalisation is not disheartening, as the text normalisation system we
used is intended for more generic use, while parliamentary texts contain
some very specific items, such as legal code references, that will
require some extension to support. It is worth considering that the size
of the training data for this model was slightly less than half that of
the others; the slight improvement did not seem worth it by itself to
justify processing more data, but moving to a more flexible system that
can produce more than a single possible normalisation does seem
worthwhile.

# Discussion

Because of the requirement of contiguous segments, the test and
validation sets overwhelmingly contain audio that commences at the start
of a speech: consequently, they both typically begin with a phrase like
"tack herr/fru talman" ("thanks mister/madam speaker"). The words "herr"
and "fru" are relatively uncommon in modern Swedish, and are typically
misrecognised by both KB-wav2vec and Whisper: KB-wav2vec, because of
their similarity in pronunciation to other, more common words; Whisper,
because it typically omits them: many of the speeches appear with
subtitles on the Riksdag Youtube channel, and follow transcription
conventions, which include starting all speeches with "Talman!", no
matter what was actually spoken.

A large part of the minor improvement seen in our 49 hour model over our
other models can be attributed to the use of a language model, and its
influence over the generation of compounds, rather than splitting them
into their components. In terms of post-editing, this represents the
deletion of a single character, whereas in terms of WER, each
incorrectly split compound results in both an insertion and a
substitution.

Another factor in favour of our 49 hour model is the use of text
normalisation: although only a single possible representation was
generated, it was risk-free, as the string of digits would not otherwise
have matched the output of KB-wav2vec. We have extended our work on
normalisation to the type of audio-based normalisation described
by [@bakhturina2022shallow], which would allow multiple possible outputs
to be generated before being filtered by the acoustic evidence, but it
was not completed in time to be used in this work.

The output of this audio-based normalisation can also be used as a means
of automatically creating a relatively reliable corpus, for use with
more current neural network-based methods of text
normalisation [@sproat2016rnn], as there currently does not exist any
suitable corpus for Swedish [@tannander2022towards].

Text normalisation only goes so far, however, and there are differences
between the spoken and written versions of the speeches that are outside
of its scope: general text processing tools that could have been
employed are typically not trained on spoken text, and would require
adaptation to be useful.

Compounding that problem, text processing tools for Swedish are not as
widely available as for other languages. The problem of phrases that are
spoken at one end of a sentence, but transcribed at the other, could
easily be tackled if there were a constituency parser or chunker
available; as is, only dependency parsers are available[^5].

There remains scope for more useful output from the intersection of
KB-wav2vec and Whisper. As KB-wav2vec outputs a quite faithful phonetic
rendering in its output, it usually transcribes false starts; through
the use of prompts, Whisper can be guided towards doing the same.
Although for most end-user purposes, it is more favourable to omit them,
false starts are useful for our purposes, as they can often be
indicative of the attitude of the speaker and provide a more complete
illustration of what was actually uttered, which is desirable for the
purposes of speech science.

Noise markers, in a CTC-based framework, would be best represented as a
'character' in the alphabet, whose representation can be selected as a
matter of configuration (in much the same way that the word separator
token is converted to a space character).

# Conclusions

In this work we investigated the development of custom ASR systems for
the transcription of archival data as an input to other research in the
realms of digital humanities.

We concentrated on parliamentary data as representative of almost ideal
conditions, where a large amount of transcribed text was available.

We investigated whether the intersection of two ASR systems was adequate
as a source of training data; although the results were quite far from
the state of the art, they were promising. In the Swedish context,
training a model in this manner makes less sense than simply using the
state of the art models, which are open-licensed. In the context of the
minority languages of Sweden, where models are either not available, or
perform less well, the matter is less settled.

We also attempted to determine whether or not the availability of
partial, approximate transcripts made enough of a difference to justify
collecting print news for use in transcribing a broadcast news archive.
We consider the improvement enough to justify the effort; this is also
transferable to other projects based around the minority languages of
Sweden, where broadcast news is a potential source of training data.

# Acknowledgements

Project needs to be mentioned here KBLab, for providing us with their
pre-trained model.

[^1]: <https://github.com/jimregan/sync-asr>https://github.com/jimregan/sync-asr

[^2]: <https://github.com/NVIDIA/NeMo-text-processing>

[^3]: \[Potentially identifying link to model\]

[^4]: \[Potentially identifying link to model\]

[^5]: Proponents of dependency parsing will conceivably argue that
    reconstructing the equivalent of a phrase chunk is at least
    theoretically possible using the output of a dependency parser; in
    practice, the effort exceeds the rewards.
