# Supervision meeting 22 Feb 2023


Skip background and framing

We have:
rel. well formed data

end goal is to transcribe — data set of interest, domain specific

Data:
Everything extracted via API

10 years

Get approximate amount of time from whisper/w2v
Number of tokens
How much has a transcription

Project motivation: transcribe everything terrorism, some missing not acceptable


Fine-tune both (whisper, wav2vec) on a small amount of data


Grab from start and end, up until point that there’s a discrepency — how much time does that give us?
— maybe think to extract all pieces greater 

#allow for shorter segments at the beginning and end

Output all chunks: start, end, file id

Filter anything below 
- 7 seconds init/final
- 15 seconds otherwise

5 words(?)

(Can either filter on number of words or times, go with times

Output anything more than 2 seconds (not for training)
Can reduce based on number of words afterwards

No pure discrepancies
…or any chunks with them
No text chunks

Add an extra token at the end that says initial, internal or final

Ok… generate CTM-ish for statistics that’s basically ID, start, end, words-if-matched, nothing-if-not, initial/internal/final

Plot a histogram over times
Both for matched/unmatched segments

Simply retrain one/both on this data, test old + new models on the test set (coming Friday), hope there’s an improvement