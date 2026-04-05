# **%Of the various combinations, only RIR returned an improvement for both FER (9.6\%) and PER (21.8\%), when compared with a model trained with a combination of PSST and unaugmented TIMIT (FER 10.2\%, PER 22.5\%).**

%The best performing model, 5-gram with silences removed from PSST and TIMIT, but with CMUdict data with silence tokens added at the end, achieved PER of 22.1\%, compared with the baseline of PSST and unaugmented TIMIT without a language model (PER 22.5\%). A plot of the results of this language model and a selection of the results from section~\ref{ssect:timitaug} can be viewed in figure~\ref{fig: results}.

The Wav2vec 2.0 base model trained on the PSST data + TIMIT with RIR achieved the best accuracy of the various combinations of augmentations, improving the scores by 4.7\% for PER (21.8 vs 22.2) and 7.3\% for FER (9.6 vs 10.2).

THE END.