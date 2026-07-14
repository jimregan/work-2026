
```
from transformers import pipeline
import glob
import json

_SWE_MODEL = "KBLab/wav2vec2-large-voxrex-swedish"
OUTDIR="/home/joregan/storspigg-tbi-wav2vec"

pipe = pipeline(model=_SWE_MODEL, framework="pt")

for file in glob.glob("*.mp3"):
        jsonfile=file.replace("mp3", "json")
        output = pipe(file, chunk_length_s=10, return_timestamps="word")
        with open(f'{OUTDIR}/{jsonfile}', 'w') as f:
                json.dump(output, f)
```