#!/usr/bin/env python3
"""Dataset validation script for CosyVoice Czech fine-tuning pipeline.

Pre-flight check to validate audio files before running extraction stages.
Catches missing/corrupted files early to prevent silent failures.

Micromamba env: fish-speech

Usage:
    python validate_dataset.py --dir /path/to/data [--strict] [--quick]
"""
import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


def validate_audio_file(audio_path: str, quick: bool = False) -> tuple:
    """Validate a single audio file.

    Args:
        audio_path: Path to audio file.
        quick: If True, only check file exists. If False, try to load it.

    Returns:
        Tuple of (path, is_valid, error_message or None).
    """
    if not os.path.exists(audio_path):
        return audio_path, False, "File not found"

    if quick:
        return audio_path, True, None

    try:
        # Try to load the audio file
        audio, sample_rate = torchaudio.load(audio_path, backend='soundfile')

        # Check for empty audio
        if audio.shape[1] == 0:
            return audio_path, False, "Empty audio (0 samples)"

        # Check for very short audio (< 0.1s)
        duration = audio.shape[1] / sample_rate
        if duration < 0.1:
            return audio_path, False, f"Audio too short ({duration:.3f}s)"

        return audio_path, True, None

    except Exception as e:
        return audio_path, False, str(e)


def log_progress(current: int, total: int, start_time: float) -> None:
    """Log progress to stdout."""
    elapsed = time.time() - start_time
    pct = (current / total) * 100 if total > 0 else 100
    rate = current / elapsed if elapsed > 0 else 0
    eta_seconds = (total - current) / rate if rate > 0 else 0
    eta_min = int(eta_seconds // 60)
    eta_sec = int(eta_seconds % 60)

    bar_width = 30
    filled = int(bar_width * current // total) if total > 0 else bar_width
    bar = '=' * filled + '>' + '.' * (bar_width - filled - 1) if filled < bar_width else '=' * bar_width

    print(f"\r[{bar}] {current:,}/{total:,} ({pct:.1f}%) | {rate:.1f} it/s | ETA: {eta_min}m {eta_sec}s    ",
          end='', flush=True)


def main(args: argparse.Namespace) -> int:
    """Main validation function.

    Args:
        args: Command line arguments.

    Returns:
        Exit code (0 for success, 1 for failures found).
    """
    data_dir = Path(args.dir)

    # Check required files exist
    wav_scp_path = data_dir / "wav.scp"
    if not wav_scp_path.exists():
        logger.error(f"wav.scp not found at {wav_scp_path}")
        return 1

    # Load wav.scp
    utt2wav = {}
    with open(wav_scp_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt2wav[parts[0]] = parts[1]
            else:
                logger.warning(f"Malformed line in wav.scp: {line.strip()}")

    total_utts = len(utt2wav)
    logger.info(f"Loaded {total_utts:,} utterances from wav.scp")

    # Check utt2spk if it exists
    utt2spk_path = data_dir / "utt2spk"
    if utt2spk_path.exists():
        utt2spk = {}
        with open(utt2spk_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utt2spk[parts[0]] = parts[1]

        # Check for missing mappings
        missing_spk = [utt for utt in utt2wav if utt not in utt2spk]
        if missing_spk:
            logger.warning(f"{len(missing_spk)} utterances missing from utt2spk")
            for utt in missing_spk[:5]:
                logger.warning(f"  Missing: {utt}")
            if len(missing_spk) > 5:
                logger.warning(f"  ... and {len(missing_spk) - 5} more")
    else:
        logger.info("utt2spk not found, skipping speaker mapping check")

    # Validate audio files
    mode = "quick (existence only)" if args.quick else "full (load audio)"
    logger.info(f"Validating audio files ({mode}) with {args.num_threads} threads...")

    valid_files = []
    invalid_files = []

    executor = ThreadPoolExecutor(max_workers=args.num_threads)
    all_tasks = [
        executor.submit(validate_audio_file, wav_path, args.quick)
        for wav_path in utt2wav.values()
    ]

    start_time = time.time()
    completed = 0
    last_log = 0

    for future in as_completed(all_tasks):
        audio_path, is_valid, error_msg = future.result()

        if is_valid:
            valid_files.append(audio_path)
        else:
            invalid_files.append((audio_path, error_msg))

        completed += 1

        # Log progress every 100 items
        if completed - last_log >= 100:
            log_progress(completed, total_utts, start_time)
            last_log = completed

    # Final progress
    log_progress(completed, total_utts, start_time)
    print()  # Newline after progress bar

    elapsed = time.time() - start_time
    logger.info(f"Validation complete in {elapsed:.1f}s")
    logger.info(f"Valid files: {len(valid_files):,} ({100 * len(valid_files) / total_utts:.1f}%)")
    logger.info(f"Invalid files: {len(invalid_files):,} ({100 * len(invalid_files) / total_utts:.1f}%)")

    # Write reports
    if invalid_files:
        invalid_report = data_dir / "invalid_files.txt"
        with open(invalid_report, 'w') as f:
            for path, error in invalid_files:
                f.write(f"{path}\t{error}\n")
        logger.info(f"Invalid files report: {invalid_report}")

        # Print first few invalid files
        logger.warning("First 10 invalid files:")
        for path, error in invalid_files[:10]:
            logger.warning(f"  {path}: {error}")
        if len(invalid_files) > 10:
            logger.warning(f"  ... and {len(invalid_files) - 10} more")

    if args.write_valid:
        valid_report = data_dir / "valid_files.txt"
        with open(valid_report, 'w') as f:
            for path in valid_files:
                f.write(f"{path}\n")
        logger.info(f"Valid files report: {valid_report}")

    # Exit code based on results
    if invalid_files:
        if args.strict:
            logger.error(f"STRICT MODE: {len(invalid_files)} invalid files found. Exiting with error.")
            return 1
        else:
            logger.warning(f"{len(invalid_files)} invalid files found. Use --strict to fail on errors.")
            return 0
    else:
        logger.info("All files validated successfully!")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate audio dataset before extraction"
    )
    parser.add_argument(
        "--dir", type=str, required=True,
        help="Directory containing wav.scp"
    )
    parser.add_argument(
        "--num_threads", type=int, default=8,
        help="Number of worker threads (default: 8)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: only check file existence, don't load audio"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with error if any invalid files found"
    )
    parser.add_argument(
        "--write_valid", action="store_true",
        help="Write list of valid files to valid_files.txt"
    )
    args = parser.parse_args()

    sys.exit(main(args))
