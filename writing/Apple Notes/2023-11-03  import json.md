>>> import json
>>> for file in glob.glob("*.mp3"):
...     outname = file.replace(".mp3", ".json")
...     output = pipe(file, chunk_length_s=10, return_timestamps="word")
...     with open(outname, "w") as outf:
...             json.dump(output, outf)


import glob
import json
from pathlib import Path
for file in glob.glob("*.mp3"):
    jsonfile = file.replace(".mp3", ".json")
    if Path(jsonfile).exists() or file in SKIP:
            continue
    print(file)  
    output = pipe(file, chunk_length_s=10, return_timestamps="word")
    with open(jsonfile, "w") as outf:
            json.dump(output, outf)

SKIP = “””
ksiega-dzungli_001_czesc-i.mp3
“””