# Anti-active learning

Starting with a known-good corpus, at each epoch slowly expand the data to less trusted: run an inference pass over the data, and whatever the model can recreate is added to the input.