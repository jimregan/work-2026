def read_reference_dir(ref_dir):
    pairs = {}
    if type(ref_dir) == str:
        ref_dir = Path(ref_dir)
        for wavfile in ref_dir.glob("*.wav"):
            name = str(wavfile)
            text = name.replace(".wav", ".txt")
            with open(text, "r") as f:
                data = f.readlines().strip()
            pairs\[wavfile\] = data
    return pairs



python /data/run_f5.py --json_file /data/numbered_all.json --ref_dir /data/audio_prompt_clips/ --output_dir /data/spk3 --speaker 3

docker run -it -v"$PWD/data:/data" --gpus '"device=3"' f5tts


for i in $(seq 20 26);do python test_ref_gen.py > /data/ggpt-output_$i.txt;done

export LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH
python /data/run_whisper.py /data/ggpt_spk5 /data/ggpt_spk5tsv


/data/audio_prompt_clips/hsi_7_0719_211_002_90_466.82_476.73.txt
“Look in the corner, THAT blue **garbage** can is always full.”



python  /data/run_f5.py  --ref_audio /data/audio_prompt_clips/hsi_7_0719_211_002_90_466.82_476.73.wav --ref_text "$(cat /data/audio_prompt_clips/hsi_7_0719_211_0_90_466.82_476.73.txt)" --gen_text “Look in the corner, *that* blue garbage can is always full.”