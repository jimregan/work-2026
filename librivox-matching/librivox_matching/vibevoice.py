"""Parse VibeVoice JSON output and merge temporally contiguous chunks."""

import json
from dataclasses import dataclass


# Maximum allowed time gap (in seconds) to consider chunks contiguous
MERGE_EPSILON = 1e-3


@dataclass
class Chunk:
    start: float
    end: float
    speaker: int
    content: str


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
    """Merge temporally contiguous chunks.

    When one chunk's end time is approximately equal (within MERGE_EPSILON)
    to the next chunk's start time, they are merged (accumulating text,
    extending the end time). Speaker is taken from the first chunk in a
    merged group.
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
        if abs(chunk.start - current.end) < MERGE_EPSILON:
            current.end = chunk.end
            current.content = current.content + " " + chunk.content
        else:
            merged.append(current)
            current = Chunk(
                start=chunk.start,
                end=chunk.end,
                speaker=chunk.speaker,
                content=chunk.content,
            )

    merged.append(current)
    return merged
