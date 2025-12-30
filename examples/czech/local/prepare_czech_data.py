#!/usr/bin/env python3
"""Prepare Czech dataset for CosyVoice3 training.

Converts CSV dataset format to CosyVoice Kaldi-style files.
Micromamba env: cosyvoice

Input CSV format: audio_file|text|speaker|duration|segments_merged
Output: wav.scp, text, utt2spk, spk2utt, instruct
"""
import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

INSTRUCT_PREFIX = "You are a helpful assistant.<|endofprompt|>"


def parse_csv_line(line: str, delimiter: str = '|') -> dict:
    """Parse a single CSV line into components.

    Args:
        line: Raw CSV line
        delimiter: Field separator

    Returns:
        Dict with audio_file, text, speaker, duration keys
    """
    parts = line.strip().split(delimiter)
    if len(parts) < 3:
        return None
    return {
        'audio_file': parts[0],
        'text': parts[1],
        'speaker': parts[2],
        'duration': float(parts[3]) if len(parts) > 3 else 0.0
    }


def generate_utt_id(audio_path: str, index: int) -> str:
    """Generate unique utterance ID from audio path.

    Args:
        audio_path: Path to audio file
        index: Sequential index for uniqueness

    Returns:
        Utterance ID string
    """
    basename = Path(audio_path).stem
    return f"{basename}_{index:06d}"


def main(args):
    """Main data preparation function."""
    os.makedirs(args.des_dir, exist_ok=True)

    # Read CSV file
    logger.info(f"Reading dataset from {args.src_csv}")
    with open(args.src_csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Skip header if present
    if lines and 'audio_file' in lines[0].lower():
        lines = lines[1:]

    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    skipped = 0

    for idx, line in enumerate(tqdm(lines, desc="Processing entries")):
        entry = parse_csv_line(line, args.delimiter)
        if entry is None:
            skipped += 1
            continue

        # Verify audio file exists
        if not os.path.exists(entry['audio_file']):
            logger.warning(f"Audio file not found: {entry['audio_file']}")
            skipped += 1
            continue

        # Generate utterance ID
        utt = generate_utt_id(entry['audio_file'], idx)
        spk = entry['speaker'].replace(' ', '_').replace('/', '_')

        utt2wav[utt] = entry['audio_file']
        utt2text[utt] = entry['text']
        utt2spk[utt] = spk

        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)

    logger.info(f"Processed {len(utt2wav)} utterances, skipped {skipped}")
    logger.info(f"Found {len(spk2utt)} unique speakers")

    # Write output files
    with open(f'{args.des_dir}/wav.scp', 'w', encoding='utf-8') as f:
        for k, v in utt2wav.items():
            f.write(f'{k} {v}\n')

    with open(f'{args.des_dir}/text', 'w', encoding='utf-8') as f:
        for k, v in utt2text.items():
            f.write(f'{k} {v}\n')

    with open(f'{args.des_dir}/utt2spk', 'w', encoding='utf-8') as f:
        for k, v in utt2spk.items():
            f.write(f'{k} {v}\n')

    with open(f'{args.des_dir}/spk2utt', 'w', encoding='utf-8') as f:
        for k, v in spk2utt.items():
            f.write(f'{k} {" ".join(v)}\n')

    # Write instruct file for CosyVoice3
    with open(f'{args.des_dir}/instruct', 'w', encoding='utf-8') as f:
        for k in utt2text.keys():
            f.write(f'{k} {INSTRUCT_PREFIX}\n')

    logger.info(f"Output written to {args.des_dir}")
    return len(utt2wav), len(spk2utt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Czech dataset for CosyVoice3")
    parser.add_argument('--src_csv', type=str, required=True,
                        help='Source CSV file path')
    parser.add_argument('--des_dir', type=str, required=True,
                        help='Destination directory for output files')
    parser.add_argument('--delimiter', type=str, default='|',
                        help='CSV delimiter (default: |)')
    args = parser.parse_args()
    main(args)
