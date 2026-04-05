from transformers import pipeline
pipe = pipeline(model=“some model name”)
output = pipe(“/my/file.wav", chunk_length_s=10, return_timestamps="word")