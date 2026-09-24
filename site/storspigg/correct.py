"""Apply the corrections the cross-source comparison supports.

Two classes of change are applied:

  reader  the metadata `narrator` column is a controlled registry field (no
          narrator has two spellings across 1000 rows), and every mismatch is a
          homophone the corrector could not have heard: Lövenborg/Löfvenborg,
          Sätterstad/Zetterstad (initial Z is /s/ in Swedish), Sjöld/Sköld,
          Pilblad/Pihlblad, Malmqvist/Malmkvist. Metadata wins on spelling.

  pages   ten files where the transcript reads "sjuttiotvå" (72) but messate,
          whisperx and wav2vec independently agree on 272.

Deliberately NOT applied:
  year    the transcript disagrees with media_year on 97 files, but media_year
          is the cataloguing year, not the year spoken: transcript, messate and
          both ASR systems agree against it every time.
  readers where the transcript names two people and the metadata lists one, and
          CA24294, whose narrator cell contradicts all four text sources.
"""
import csv
import difflib
import re
import sys

import slots
import sources

# Metadata narrator is authoritative for spelling, except where the transcript
# is demonstrably better informed.
SKIP_READER = {
    "CA24294",  # narrator cell says Godenius, Anna; all four texts say Bäckström
    "CA24623",  # transcript names two readers, metadata lists one
}
# Transcript names two readers; apply only the spelling fix, keep both names.
PARTIAL_READER = {
    "CA30220": ("Elisabeth Lindberg", "Elisabet Lindberg"),
}
REVIEW_YEAR = {"CA32579", "CA33620"}

READER_RE = re.compile(r"(Inläsare\s+är\s+)(.+?)(\s+(?:vid|för|hos|på)\s+)", re.IGNORECASE)


def spoken_name(narrator):
    if "," in narrator:
        last, first = narrator.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return narrator.strip()


UNITS = ["", "ett", "två", "tre", "fyra", "fem", "sex", "sju", "åtta", "nio"]
TEENS = ["tio", "elva", "tolv", "tretton", "fjorton", "femton", "sexton",
         "sjutton", "arton", "nitton"]
TENS = ["", "", "tjugo", "trettio", "fyrtio", "femtio", "sextio", "sjuttio",
        "åttio", "nittio"]


def to_words(n, common_gender=True):
    """Swedish cardinal as one run-together word, matching transcript style.

    `sida` is an en-word, so a count ending in 1 is "...en sidor" not "...ett
    sidor" — the transcript already writes femhundrafyrtioen, trehundrasjuttioen.
    """
    one = "en" if common_gender else "ett"
    if n < 10:
        return one if n == 1 else UNITS[n]
    if n < 20:
        return TEENS[n - 10]
    if n < 100:
        rest = n % 100 % 10
        return TENS[n // 10] + (one if rest == 1 else UNITS[rest])
    if n < 1000:
        # the hundreds multiplier keeps "ett-": etthundraen, not enhundraen
        head = "etthundra" if n // 100 == 1 else UNITS[n // 100] + "hundra"
        rest = n % 100
        return head + (to_words(rest, common_gender) if rest else "")
    raise ValueError(n)


def main(out_path, log_path):
    rows = {k: v for k, v in sources.load().items() if k}
    ext = {k: {s: slots.extract(v[s]) for s in sources.TEXT_SOURCES}
           for k, v in rows.items()}

    changes = []
    corrected = {}

    for key, v in rows.items():
        text = v["transcript_raw"]
        orig = text

        # --- reader spelling from the metadata registry
        tr = ext[key]["transcript"]["reader"]
        gt = v["narrator_tokens"]
        if key in PARTIAL_READER:
            old, new = PARTIAL_READER[key]
            if old in text:
                text = text.replace(old, new)
                changes.append((key, "reader-partial", old, new))
        elif key not in SKIP_READER and tr and gt and tr != gt:
            want = spoken_name(v["narrator"])
            m = READER_RE.search(text)
            if m:
                text = text[:m.start(2)] + want + text[m.end(2):]
                changes.append((key, "reader", m.group(2), want))
            else:
                changes.append((key, "reader-FAILED", " ".join(tr), want))

        # --- page count where three independent sources outvote the transcript
        t_pages = ext[key]["transcript"]["pages"]
        m_pages = ext[key]["messate"]["pages"]
        w_pages = ext[key]["whisperx"]["pages"]
        v_pages = ext[key]["wav2vec"]["pages"]
        if (t_pages is not None and t_pages != m_pages
                and m_pages == w_pages == v_pages and m_pages is not None):
            old_w = to_words(t_pages)
            new_w = to_words(m_pages)
            pat = re.compile(r"\b" + old_w + r"\b(?=\s+sidor)", re.IGNORECASE)
            if pat.search(text):
                text = pat.sub(new_w, text, count=1)
                changes.append((key, "pages", old_w, new_w))
            else:
                changes.append((key, "pages-FAILED", old_w, new_w))

        corrected[key] = text
        if text != orig:
            pass

    with open(out_path, "w") as fh:
        for key in sorted(corrected):
            fh.write(f"{key}\t{corrected[key]}\n")

    with open(log_path, "w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "kind", "from", "to"])
        for row in sorted(changes):
            w.writerow(row)
        for key in sorted(REVIEW_YEAR):
            w.writerow([key, "review-year",
                        str(ext[key]["transcript"]["year"]),
                        "whisperx/wav2vec say 2017 (low conf 0.30-0.33)"])

    kinds = {}
    for _, kind, _, _ in changes:
        kinds[kind] = kinds.get(kind, 0) + 1
    print("files:", len(corrected))
    print("changes:", len(changes), kinds)
    print("files changed:", sum(1 for k in corrected if corrected[k] != rows[k]["transcript_raw"]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
