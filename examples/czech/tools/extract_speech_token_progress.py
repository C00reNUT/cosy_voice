#!/usr/bin/env python3
"""Speech token extraction with nohup-friendly progress bars and resume capability.

Extracts speech tokens using speech_tokenizer ONNX model for CosyVoice fine-tuning.
Supports resuming from interrupted runs - already processed samples are skipped.
Progress is logged to stdout with periodic updates suitable for log files.

Environment: micromamba activate fish-speech

Usage:
    python extract_speech_token_progress.py --dir /path/to/data --onnx_path /path/to/speech_tokenizer_v3.onnx
"""
import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torchaudio
import whisper

logging.basicConfig(level=logging.WARNING)


def single_job(utt: str, utt2wav: dict, ort_session) -> tuple:
    """Extract speech tokens for a single utterance.

    Args:
        utt: Utterance ID.
        utt2wav: Mapping of utterance IDs to wav file paths.
        ort_session: ONNX runtime session.

    Returns:
        Tuple of (utterance_id, speech_token_list).
    """
    audio, sample_rate = torchaudio.load(utt2wav[utt], backend='soundfile')
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    # Convert audio to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if audio.shape[1] / 16000 > 30:
        logging.warning(f'Skipping {utt}: audio longer than 30s')
        speech_token = []
    else:
        feat = whisper.log_mel_spectrogram(audio, n_mels=128)
        speech_token = ort_session.run(
            None,
            {
                ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)
            }
        )[0].flatten().tolist()
    return utt, speech_token


def log_progress(current: int, total: int, start_time: float, skipped: int = 0, prefix: str = "") -> None:
    """Log progress with ETA to stdout (flushes immediately).

    Args:
        current: Current progress count (processed in this run).
        total: Total items to process in this run.
        start_time: Start time from time.time().
        skipped: Number of items skipped (already processed).
        prefix: Optional prefix string.
    """
    elapsed = time.time() - start_time
    pct = (current / total) * 100 if total > 0 else 100
    rate = current / elapsed if elapsed > 0 else 0
    eta_seconds = (total - current) / rate if rate > 0 else 0
    eta_min = int(eta_seconds // 60)
    eta_sec = int(eta_seconds % 60)

    bar_width = 30
    filled = int(bar_width * current // total) if total > 0 else bar_width
    bar = '=' * filled + '>' + '.' * (bar_width - filled - 1) if filled < bar_width else '=' * bar_width

    skip_info = f" (skipped {skipped:,})" if skipped > 0 else ""
    print(f"\r{prefix}[{bar}] {current:,}/{total:,} ({pct:.1f}%){skip_info} | {rate:.1f} it/s | ETA: {eta_min}m {eta_sec}s    ",
          end='', flush=True)


def main(args: argparse.Namespace) -> None:
    """Main extraction function with resume capability.

    Args:
        args: Command line arguments.
    """
    token_path = Path(args.dir) / "utt2speech_token.pt"
    checkpoint_path = Path(args.dir) / "speech_token_checkpoint.pt"

    # Load data mappings
    utt2wav = {}
    with open(f'{args.dir}/wav.scp') as f:
        for line in f:
            parts = line.strip().split()
            utt2wav[parts[0]] = parts[1]

    total_utts = len(utt2wav)

    # Check for existing progress (checkpoint or final output)
    utt2speech_token = {}
    if checkpoint_path.exists():
        print(f"Found checkpoint, loading...", flush=True)
        utt2speech_token = torch.load(checkpoint_path)
        print(f"Loaded {len(utt2speech_token):,} tokens from checkpoint", flush=True)
    elif token_path.exists() and not args.no_resume:
        print(f"Found existing utt2speech_token.pt, loading...", flush=True)
        utt2speech_token = torch.load(token_path)
        print(f"Loaded {len(utt2speech_token):,} tokens from previous run", flush=True)

    # Filter out already processed utterances
    utts_to_process = [utt for utt in utt2wav.keys() if utt not in utt2speech_token]
    skipped = total_utts - len(utts_to_process)

    if len(utts_to_process) == 0:
        print(f"All {total_utts:,} utterances already processed. Skipping extraction.", flush=True)
        return

    print(f"Processing {len(utts_to_process):,} utterances ({skipped:,} already done) with {args.num_thread} threads...", flush=True)

    # Setup ONNX session
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1

    # Try CUDA first, fall back to CPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
    active_provider = ort_session.get_providers()[0]
    print(f"Using provider: {active_provider}", flush=True)

    # Process with thread pool
    executor = ThreadPoolExecutor(max_workers=args.num_thread)
    all_tasks = [executor.submit(single_job, utt, utt2wav, ort_session) for utt in utts_to_process]

    start_time = time.time()
    completed = 0
    last_log = 0
    last_checkpoint = time.time()
    checkpoint_interval = 300  # Save checkpoint every 5 minutes

    total_to_process = len(utts_to_process)

    for future in as_completed(all_tasks):
        utt, speech_token = future.result()
        utt2speech_token[utt] = speech_token

        completed += 1

        # Log progress every 100 items
        if completed - last_log >= 100:
            log_progress(completed, total_to_process, start_time, skipped)
            last_log = completed

        # Save checkpoint periodically
        if time.time() - last_checkpoint >= checkpoint_interval:
            torch.save(utt2speech_token, checkpoint_path)
            last_checkpoint = time.time()
            print(f"\n[Checkpoint saved: {len(utt2speech_token):,} tokens]", flush=True)

    # Final progress
    log_progress(completed, total_to_process, start_time, skipped)
    elapsed = time.time() - start_time
    print(f"\nCompleted {total_to_process:,} utterances in {elapsed/60:.1f} minutes ({total_to_process/elapsed:.1f} it/s)", flush=True)

    # Save final results
    torch.save(utt2speech_token, token_path)
    print(f"Saved utt2speech_token.pt ({len(utt2speech_token):,} utterances)", flush=True)

    # Remove checkpoint file after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Removed checkpoint file", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract speech tokens with progress tracking and resume")
    parser.add_argument("--dir", type=str, required=True, help="Directory with wav.scp")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to speech_tokenizer_v3.onnx")
    parser.add_argument("--num_thread", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--no_resume", action="store_true", help="Disable auto-resume (reprocess all)")
    args = parser.parse_args()

    main(args)
