To obtain segment level transcriptions of the audio, we used whisperX~\cite{bain2022whisperx}, which employs a voice activity detection module to obtain better timestamps. For better accuracy, we prompted whisperX with some in-domain text that had been edited to include filled pauses, repeated words, and number words instead of digits. Although this improved the accuracy of the transcription, we still needed to post-process the results to replace remaining digits. Although whisperX can produce word-level timestamps, they were insufficiently accurate for our purposes. To obtain these timestamps, we used the tools created for the librispeech corpus~\cite{panayotov2015librispeech} to intersect whisperX’s output with the output from a system based on Kaldi~\cite{Povey_ASRU2011}, taking the timestamps from that system, while accepting any insertions of a set of filled pauses regardless of context, and other words if the word to either side was the same. In the event of remaining disagreements between the systems, we used the Montreal Forced Aligner~\cite{mcauliffe17_interspeech}

@article{bain2022whisperx,
  title={Whisper{X}: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  journal={Proc. Interspeech 2023},
  year={2023}
}

@INPROCEEDINGS{panayotov2015librispeech,
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={Proc. ICASSP 2015},
  title={Librispeech: An {ASR} corpus based on public domain audio books}, 
  year={2015},
  pages={5206-5210},
  doi={10.1109/ICASSP.2015.7178964}
}

@INPROCEEDINGS{Povey_ASRU2011,
         author = {Povey, Daniel and Ghoshal, Arnab and Boulianne, Gilles and Burget, Lukas and Glembek, Ondrej and Goel, Nagendra and Hannemann, Mirko and Motlicek, Petr and Qian, Yanmin and Schwarz, Petr and Silovsky, Jan and Stemmer, Georg and Vesely, Karel},
          month = dec,
          title = {The {K}aldi Speech Recognition Toolkit},
      booktitle = {Proc. ASRU 2011},
           year = {2011},
      publisher = {IEEE Signal Processing Society},
}

@inproceedings{mcauliffe17_interspeech,
  author={McAuliffe, Michael and Socolof, Michaela and Mihuc, Sarah and Wagner, 
Michael and Sonderegger, Morgan},
  title={{Montreal Forced Aligner}: Trainable Text-Speech Alignment Using {K}aldi},
  year=2017,
  booktitle={Proc. Interspeech 2017},
  pages={498--502},
  doi={10.21437/Interspeech.2017-1386}
}