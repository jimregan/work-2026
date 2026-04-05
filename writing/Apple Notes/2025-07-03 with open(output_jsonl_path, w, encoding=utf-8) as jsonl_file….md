with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:
    for i, sample in enumerate(dataset):
        audio = sample\["audio"\]
        audio_path = os.path.join(output_audio_dir, f"{i}.wav")
        sf.write(audio_path, audio\["array"\], audio\["sampling_rate"\])
        sample_copy = dict(sample)
        sample_copy.pop("audio", None)
        sample_copy\["audio_filepath"\] = audio_path
        jsonl_file.write(json.dumps(sample_copy, ensure_ascii=False) + "\n")



with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:
    for i, sample in enumerate(dataset):
        audio = sample\["audio"\]
        audio_path = os.path.join(output_audio_dir, f"{i}.wav")
        sf.write(audio_path, audio\["array"\], audio\["sampling_rate"\])
        sample_copy = dict(sample)
        sample_copy.pop("audio", None)
        sample_copy\["audio_filepath"\] = audio_path
        serializable_sample = convert_for_json(sample_copy)
        jsonl_file.write(json.dumps(serializable_sample, ensure_ascii=False) + "\n")