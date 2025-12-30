#!/usr/bin/env python3
"""Split dataset into train and eval sets.

Creates 99% train / 1% eval split, stratified by speaker.
Micromamba env: cosyvoice
"""
import argparse
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def load_kaldi_file(filepath: str) -> dict:
    """Load Kaldi-style file into dictionary."""
    result = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) >= 2:
                result[parts[0]] = parts[1]
            elif len(parts) == 1:
                result[parts[0]] = ''
    return result


def save_kaldi_file(filepath: str, data: dict):
    """Save dictionary to Kaldi-style file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for k, v in data.items():
            f.write(f'{k} {v}\n')


def main(args):
    """Split dataset into train and eval sets."""
    random.seed(args.seed)

    # Load source files
    src_dir = args.src_dir
    wav_scp = load_kaldi_file(f'{src_dir}/wav.scp')
    text = load_kaldi_file(f'{src_dir}/text')
    utt2spk = load_kaldi_file(f'{src_dir}/utt2spk')
    instruct = load_kaldi_file(f'{src_dir}/instruct') if os.path.exists(f'{src_dir}/instruct') else {}

    # Group utterances by speaker
    spk2utts = defaultdict(list)
    for utt, spk in utt2spk.items():
        spk2utts[spk].append(utt)

    # Split each speaker's utterances
    train_utts, eval_utts = [], []
    eval_ratio = args.eval_percent / 100.0

    for spk, utts in spk2utts.items():
        random.shuffle(utts)
        n_eval = max(1, int(len(utts) * eval_ratio))
        eval_utts.extend(utts[:n_eval])
        train_utts.extend(utts[n_eval:])

    logger.info(f"Train: {len(train_utts)} utterances, Eval: {len(eval_utts)} utterances")

    # Create output directories
    train_dir = f'{args.des_dir}/train'
    eval_dir = f'{args.des_dir}/eval'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Write train files
    for split_name, split_utts, split_dir in [
        ('train', train_utts, train_dir),
        ('eval', eval_utts, eval_dir)
    ]:
        save_kaldi_file(f'{split_dir}/wav.scp', {u: wav_scp[u] for u in split_utts})
        save_kaldi_file(f'{split_dir}/text', {u: text[u] for u in split_utts})
        save_kaldi_file(f'{split_dir}/utt2spk', {u: utt2spk[u] for u in split_utts})

        # Build spk2utt
        spk2utt = defaultdict(list)
        for u in split_utts:
            spk2utt[utt2spk[u]].append(u)
        save_kaldi_file(f'{split_dir}/spk2utt', {k: ' '.join(v) for k, v in spk2utt.items()})

        if instruct:
            save_kaldi_file(f'{split_dir}/instruct', {u: instruct[u] for u in split_utts if u in instruct})

        logger.info(f"Written {split_name} to {split_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/eval")
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--des_dir', type=str, required=True)
    parser.add_argument('--eval_percent', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
