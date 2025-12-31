#!/usr/bin/env python3
"""Speaker embedding extraction with nohup-friendly progress bars and resume capability.

Extracts speaker embeddings using CAM++ model for CosyVoice fine-tuning.
Supports resuming from interrupted runs - already processed samples are skipped.
Progress is logged to stdout with periodic updates suitable for log files.

Environment: micromamba activate fish-speech

Usage:
    python extract_embedding_progress.py --dir /path/to/data --onnx_path /path/to/campplus.onnx
"""
import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def single_job(utt: str, utt2wav: dict, ort_session) -> tuple:
    """Extract embedding for a single utterance.

    Args:
        utt: Utterance ID.
        utt2wav: Mapping of utterance IDs to wav file paths.
        ort_session: ONNX runtime session.

    Returns:
        Tuple of (utterance_id, embedding_list or None if file missing/error, error_msg or None).
    """
    wav_path = utt2wav[utt]

    # Pre-check file existence
    if not os.path.exists(wav_path):
        return utt, None, f"File not found: {wav_path}"

    try:
        audio, sample_rate = torchaudio.load(wav_path)
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = ort_session.run(
            None,
            {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()}
        )[0].flatten().tolist()
        return utt, embedding, None
    except Exception as e:
        error_type = type(e).__name__
        return utt, None, f"{error_type}: {str(e)} ({wav_path})"


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


def write_failed_log(failed_list: List[Tuple[str, str]], log_path: Path) -> None:
    """Write failed utterances to log file.

    Args:
        failed_list: List of (utt_id, error_msg) tuples.
        log_path: Path to write the log file.
    """
    with open(log_path, 'w') as f:
        f.write(f"# Failed utterances log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total failed: {len(failed_list)}\n")
        f.write("#" + "=" * 79 + "\n")
        for utt, error in failed_list:
            f.write(f"{utt}\t{error}\n")


def main(args: argparse.Namespace) -> int:
    """Main extraction function with resume capability.

    Args:
        args: Command line arguments.

    Returns:
        Exit code (0 for success, 1 for critical errors).
    """
    # Validate ONNX model exists
    if not os.path.exists(args.onnx_path):
        print(f"ERROR: ONNX model not found: {args.onnx_path}", flush=True)
        return 1

    utt_emb_path = Path(args.dir) / "utt2embedding.pt"
    spk_emb_path = Path(args.dir) / "spk2embedding.pt"
    checkpoint_path = Path(args.dir) / "embedding_checkpoint.pt"
    failed_log_path = Path(args.dir) / "failed_embedding.log"

    # Load data mappings
    wav_scp_path = Path(args.dir) / "wav.scp"
    utt2spk_path = Path(args.dir) / "utt2spk"

    if not wav_scp_path.exists():
        print(f"ERROR: wav.scp not found: {wav_scp_path}", flush=True)
        return 1
    if not utt2spk_path.exists():
        print(f"ERROR: utt2spk not found: {utt2spk_path}", flush=True)
        return 1

    utt2wav, utt2spk = {}, {}
    with open(wav_scp_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt2wav[parts[0]] = parts[1]
    with open(utt2spk_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt2spk[parts[0]] = parts[1]

    total_utts = len(utt2wav)
    print(f"Loaded {total_utts:,} utterances from wav.scp", flush=True)

    # Check for existing progress (checkpoint or final output)
    utt2embedding = {}
    if checkpoint_path.exists():
        print(f"Found checkpoint, loading...", flush=True)
        utt2embedding = torch.load(checkpoint_path)
        print(f"Loaded {len(utt2embedding):,} embeddings from checkpoint", flush=True)
    elif utt_emb_path.exists() and not args.no_resume:
        print(f"Found existing utt2embedding.pt, loading...", flush=True)
        utt2embedding = torch.load(utt_emb_path)
        print(f"Loaded {len(utt2embedding):,} embeddings from previous run", flush=True)

    # Filter out already processed utterances
    utts_to_process = [utt for utt in utt2wav.keys() if utt not in utt2embedding]
    skipped = total_utts - len(utts_to_process)

    if len(utts_to_process) == 0:
        print(f"All {total_utts:,} utterances already processed. Skipping extraction.", flush=True)
        # Still need to compute spk2embedding if not exists
        if not spk_emb_path.exists():
            print("Computing speaker embeddings...", flush=True)
            spk2embedding = {}
            for utt, emb in utt2embedding.items():
                spk = utt2spk[utt]
                if spk not in spk2embedding:
                    spk2embedding[spk] = []
                spk2embedding[spk].append(emb)
            for k, v in spk2embedding.items():
                spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()
            torch.save(spk2embedding, spk_emb_path)
            print(f"Saved spk2embedding.pt ({len(spk2embedding):,} speakers)", flush=True)
        return 0

    print(f"Processing {len(utts_to_process):,} utterances ({skipped:,} already done) with {args.num_thread} threads...", flush=True)

    # Setup ONNX session
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)

    # Process with thread pool
    executor = ThreadPoolExecutor(max_workers=args.num_thread)
    all_tasks = [executor.submit(single_job, utt, utt2wav, ort_session) for utt in utts_to_process]

    start_time = time.time()
    completed = 0
    failed = 0
    failed_items: List[Tuple[str, str]] = []  # (utt_id, error_msg)
    last_log = 0
    last_checkpoint = time.time()
    checkpoint_interval = 300  # Save checkpoint every 5 minutes

    total_to_process = len(utts_to_process)

    for future in as_completed(all_tasks):
        utt, embedding, error_msg = future.result()

        # Skip failed extractions (missing/corrupted files)
        if embedding is None:
            failed += 1
            failed_items.append((utt, error_msg or "Unknown error"))
            continue

        utt2embedding[utt] = embedding
        completed += 1

        # Log progress every 100 items
        if completed - last_log >= 100:
            log_progress(completed, total_to_process - failed, start_time, skipped)
            last_log = completed

        # Save checkpoint periodically
        if time.time() - last_checkpoint >= checkpoint_interval:
            torch.save(utt2embedding, checkpoint_path)
            last_checkpoint = time.time()
            print(f"\n[Checkpoint saved: {len(utt2embedding):,} embeddings]", flush=True)

    # Final progress
    log_progress(completed, total_to_process - failed, start_time, skipped)
    elapsed = time.time() - start_time
    print(f"\nCompleted {completed:,} utterances in {elapsed/60:.1f} minutes ({completed/elapsed:.1f} it/s)", flush=True)

    # Write failed utterances log
    if failed_items:
        write_failed_log(failed_items, failed_log_path)
        print(f"WARNING: {failed:,} files failed - details in {failed_log_path}", flush=True)
        # Print first 5 failures
        for utt, err in failed_items[:5]:
            print(f"  - {utt}: {err}", flush=True)
        if len(failed_items) > 5:
            print(f"  ... and {len(failed_items) - 5} more", flush=True)

    # Compute speaker embeddings
    print("Computing speaker embeddings...", flush=True)
    spk2embedding = {}
    for utt, emb in utt2embedding.items():
        spk = utt2spk[utt]
        if spk not in spk2embedding:
            spk2embedding[spk] = []
        spk2embedding[spk].append(emb)
    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()

    # Save final results
    torch.save(utt2embedding, utt_emb_path)
    torch.save(spk2embedding, spk_emb_path)
    print(f"Saved utt2embedding.pt ({len(utt2embedding):,} utts) and spk2embedding.pt ({len(spk2embedding):,} spks)", flush=True)

    # Remove checkpoint file after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Removed checkpoint file", flush=True)

    return 0


if __name__ == "__main__":
    print("=== USING CUSTOM PROGRESS SCRIPT ===", flush=True)
    parser = argparse.ArgumentParser(description="Extract speaker embeddings with progress tracking and resume")
    parser.add_argument("--dir", type=str, required=True, help="Directory with wav.scp and utt2spk")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to campplus.onnx")
    parser.add_argument("--num_thread", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--no_resume", action="store_true", help="Disable auto-resume (reprocess all)")
    args = parser.parse_args()

    sys.exit(main(args))
