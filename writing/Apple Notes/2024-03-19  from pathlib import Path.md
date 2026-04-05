>>> from pathlib import Path
>>> TSVDIR = Path("/tmp/tsv")
>>> OUTDIR = Path("/tmp/ctm")
>>> for file in TSVDIR.glob("*.tsv"):
...     if ".interloctr." in file.name:
...             continue
...     OUTFILE = OUTDIR / f"{file.stem}.ctm"
...     a = GeneaTSV(filename=str(file))
...     a.write_ctm(str(OUTFILE))

20-20 twenty twenty
90s nineties
16 sixteen
1,**000 a thousand**
**20 twenty**