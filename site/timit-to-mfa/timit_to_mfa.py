#!/usr/bin/env python3
"""
Convert TIMIT .phn files to MFA-compatible TextGrids with an identity dictionary.

Merging rules applied:
  - Closure phones (bcl, dcl, gcl, kcl, pcl, tcl) are merged with the
    following burst, spanning both segments under the burst label.
  - Silence phones (h#, pau, epi) are normalised to 'sil'.
  - Glottal stop (q) and all other phones are kept as-is.
"""

import argparse
import sys
from pathlib import Path

SAMPLE_RATE = 16000

CLOSURE_TO_BURST = {
    'bcl': 'b',
    'dcl': 'd',
    'gcl': 'g',
    'kcl': 'k',
    'pcl': 'p',
    'tcl': 't',
}

SILENCE_PHONES = {'h#', 'pau', 'epi'}
SILENCE_LABEL = 'sil'


def read_phn(path):
    segments = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            start, end, phone = int(parts[0]), int(parts[1]), parts[2]
            segments.append((start / SAMPLE_RATE, end / SAMPLE_RATE, phone))
    return segments


def merge_segments(segments, phn_path):
    result = []
    i = 0
    while i < len(segments):
        start, end, phone = segments[i]

        if phone in SILENCE_PHONES:
            # Merge adjacent silence runs produced by consecutive silence phones
            j = i + 1
            while j < len(segments) and segments[j][2] in SILENCE_PHONES:
                j += 1
            result.append((start, segments[j - 1][1], SILENCE_LABEL))
            i = j
            continue

        if phone in CLOSURE_TO_BURST:
            burst = CLOSURE_TO_BURST[phone]
            if i + 1 < len(segments) and segments[i + 1][2] == burst:
                result.append((start, segments[i + 1][1], burst))
                i += 2
                continue
            # Orphan closure (unreleased stop): keep as-is
            result.append((start, end, phone))
            i += 1
            continue

        result.append((start, end, phone))
        i += 1

    return result


def write_textgrid(path, segments, tier_name):
    if not segments:
        return

    xmin = segments[0][0]
    xmax = segments[-1][1]

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        '',
        f'xmin = {xmin}',
        f'xmax = {xmax}',
        'tiers? <exists>',
        'size = 1',
        'item []:',
        '    item [1]:',
        '        class = "IntervalTier"',
        f'        name = "{tier_name}"',
        f'        xmin = {xmin}',
        f'        xmax = {xmax}',
        f'        intervals: size = {len(segments)}',
    ]

    for idx, (seg_start, seg_end, phone) in enumerate(segments, 1):
        lines += [
            f'        intervals [{idx}]:',
            f'            xmin = {seg_start}',
            f'            xmax = {seg_end}',
            f'            text = "{phone}"',
        ]

    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def collect_phones(all_segments):
    phones = set()
    for segments in all_segments:
        for _, _, phone in segments:
            if phone != SILENCE_LABEL:
                phones.add(phone)
    return sorted(phones)


def write_dictionary(path, phones):
    lines = [f'{phone} {phone}' for phone in phones]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('timit_root', type=Path,
                        help='Root of TIMIT corpus (directory containing TRAIN/TEST or train/test)')
    parser.add_argument('--dict', type=Path, default=Path('timit_identity.dict'),
                        help='Output dictionary path (default: timit_identity.dict)')
    parser.add_argument('--tier', default='words',
                        help='TextGrid tier name (default: words)')
    args = parser.parse_args()

    phn_files = sorted(args.timit_root.rglob('*.PHN'))
    if not phn_files:
        phn_files = sorted(args.timit_root.rglob('*.phn'))

    if not phn_files:
        print(f'No .phn files found under {args.timit_root}', file=sys.stderr)
        sys.exit(1)

    print(f'Found {len(phn_files)} utterances')

    all_segments = []
    for phn_path in phn_files:
        raw = read_phn(phn_path)
        merged = merge_segments(raw, phn_path)
        all_segments.append(merged)
        tg_path = phn_path.with_suffix('.TextGrid')
        write_textgrid(tg_path, merged, tier_name=args.tier)

    phones = collect_phones(all_segments)
    write_dictionary(args.dict, phones)

    print(f'Wrote dictionary with {len(phones)} phones to {args.dict}')
    print(f'Phone set ({len(phones)}): {" ".join(phones)}')


if __name__ == '__main__':
    main()
