Get list of detected languages - use as feature vector
train language estimator on that vector

run the language ID with variants of speech windows

step 100 ms
window 3 seconds

stick with whisper for now


range that takes audio, chops it into window/steps
feeds them to whisper
saves for each start, end, language estimates

smooth window by step, moving average