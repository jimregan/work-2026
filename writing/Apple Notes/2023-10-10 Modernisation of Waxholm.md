# Modernisation of Waxholm

To bring the audio to modern standards, the audio of the current release of the Waxholm dataset was converted to WAV format: that is, a RIFF header replaces the SMP text header, and the bytes are converted from big-endian to little-endian. No other processing was performed on the audio.

For the text: FIXME: stuff happened. More than the above. Code is available.

The existing test set was retained, which consists of a set of sessions, which correspond to individual speakers. An additional validation set was created, by taking 5% of the sessions, to ensure that the speakers are distinct between training and validation sets.
TODO: actually do this again, and retrain, to be 100% certain that this is what happened.