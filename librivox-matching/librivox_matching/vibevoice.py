"""Parse VibeVoice JSON output and merge chunks at sentence boundaries."""

import json
from dataclasses import dataclass


@dataclass
class Chunk:
    start: float
    end: float
    speaker: int
    content: str


SENTENCE_ENDINGS = frozenset(".!?")


def parse_vibevoice(json_path: str) -> list[Chunk]:
    """Parse a VibeVoice JSON file into a list of Chunks."""
    with open(json_path) as f:
        data = json.load(f)
    return [
        Chunk(
            start=item["Start"],
            end=item["End"],
            speaker=item["Speaker"],
            content=item["Content"],
        )
        for item in data
    ]


def merge_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Merge consecutive chunks that don't end with sentence-ending punctuation.

    If a chunk's content does not end with '.', '!', or '?', it is merged
    with the following chunk (accumulating text, extending the end time).
    Speaker is taken from the first chunk in a merged group.
    """
    if not chunks:
        return []

    merged = []
    current = Chunk(
        start=chunks[0].start,
        end=chunks[0].end,
        speaker=chunks[0].speaker,
        content=chunks[0].content,
    )

    for chunk in chunks[1:]:
        text = current.content.rstrip()
        if text and text[-1] in SENTENCE_ENDINGS:
            merged.append(current)
            current = Chunk(
                start=chunk.start,
                end=chunk.end,
                speaker=chunk.speaker,
                content=chunk.content,
            )
        else:
            current.end = chunk.end
            current.content = current.content + " " + chunk.content

    merged.append(current)
    return merged
