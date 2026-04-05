Segment-level transcription of the audio was performed using whisperX~\cite{bain2022whisperx}, which was prompted to produce filler and repeated words.
Word-level transcriptions were obtained by passing the resulting segments to the Montreal Forced Aligner.
To obtain more accurate results from the aligner, the lexicon was automatically expanded to