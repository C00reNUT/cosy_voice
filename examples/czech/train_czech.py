#!/usr/bin/env python3
"""Czech fine-tuning script for Fun-CosyVoice3-0.5B-2512.

Features:
- LLM + Flow joint SFT training
- Rolling checkpoints (max 3, every 500 steps)
- Evaluation every 5000 steps with TTS generation
- Best model tracking by eval loss
- Full resume capability

Micromamba env: cosyvoice_training
Environment location: $MAMBA_ROOT_PREFIX/envs/cosyvoice_training

TTS Evaluation Sentence Files:
- Czech: examples/czech/local/eval_sentences.json
- German: examples/czech/local/eval_sentences_german.json

Usage:
    torchrun --nproc_per_node=1 train_czech.py \\
        --config conf/cosyvoice3_czech.yaml \\
        --train_data data/train/parquet/data.list \\
        --cv_data data/eval/parquet/data.list \\
        --model_dir /mnt/8TB/TRAINING_RUNS/cosyvoice/...
"""
from __future__ import print_function
import argparse
import datetime
import logging
import os
import sys
import functools
import json
import math
import random
import glob
import shutil
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torchaudio
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from torch.distributed.elastic.multiprocessing.errors import record

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
matcha_root = PROJECT_ROOT / "third_party" / "Matcha-TTS"
if matcha_root.exists():
    sys.path.insert(0, str(matcha_root))

from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config,
    batch_forward, log_per_step
)
from cosyvoice.cli.cosyvoice import CosyVoice3

# Import local eval sentences
sys.path.insert(0, str(Path(__file__).parent))
from local.eval_sentences import get_eval_sentences

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Czech CosyVoice3 Fine-tuning')
    parser.add_argument('--train_engine', default='torch_ddp',
                        choices=['torch_ddp'], help='Training engine')
    parser.add_argument('--model', default='llm', choices=['llm', 'flow'],
                        help='Model to train')
    parser.add_argument('--config', required=True, help='Config file')
    parser.add_argument('--train_data', required=True, help='Train data list')
    parser.add_argument('--cv_data', required=True, help='CV data list')
    parser.add_argument('--qwen_pretrain_path', required=True, help='Qwen tokenizer path')
    parser.add_argument('--checkpoint', help='Checkpoint to load')
    parser.add_argument('--model_dir', required=True, help='Output model directory')
    parser.add_argument('--tensorboard_dir', default=None, help='TensorBoard log dir')
    parser.add_argument('--dataset_csv', required=True, help='Dataset CSV for speaker refs')
    parser.add_argument('--ddp.dist_backend', dest='dist_backend', default='nccl')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--prefetch', default=100, type=int)
    parser.add_argument('--pin_memory', action='store_true', default=True)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--save_per_step', default=500, type=int, help='Save every N steps')
    parser.add_argument('--eval_per_step', default=5000, type=int, help='Eval every N steps (0 to disable step eval)')
    parser.add_argument('--eval_per_epoch', default=1, type=int,
                        help='Eval every N epochs (0 to disable epoch eval)')
    parser.add_argument('--max_checkpoints', default=3, type=int, help='Max rolling checkpoints (total)')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint dir')
    parser.add_argument('--timeout', default=60, type=int, help='Timeout for joins')
    parser.add_argument('--tts_eval_per_step', default=500, type=int,
                        help='Generate TTS samples every N steps (0 to disable step TTS)')
    parser.add_argument('--tts_eval_per_epoch', nargs='?', const=1, default=0, type=int,
                        help='Generate TTS samples every N epochs (0 to disable; flag means every epoch)')
    parser.add_argument('--llm_eval_checkpoint', default=None,
                        help='LLM checkpoint path for TTS eval during flow training')
    parser.add_argument('--flow_eval_checkpoint', '--flow-eval-checkpoint', default=None,
                        help='Flow checkpoint path for TTS eval during LLM training')
    parser.add_argument('--flow_mel_l1_utts', default=1, type=int,
                        help='Number of CV utterances to compute flow mel L1 (0 to disable)')
    parser.add_argument('--max-frames-in-batch', dest='max_frames_in_batch', default=None, type=int,
                        help='Override dynamic batch max_frames_in_batch in config')
    parser.add_argument('--lr', dest='lr', default=None, type=float,
                        help='Override optimizer lr (train_conf.optim_conf.lr)')
    parser.add_argument('--max-epoch', dest='max_epoch', default=None, type=int,
                        help='Override max_epoch from config (applies to both LLM and Flow)')
    parser.add_argument('--max-steps', dest='max_steps', default=0, type=int,
                        help='Stop training after this many global steps (0 to disable)')
    parser.add_argument('--accum-grad', dest='accum_grad', default=None, type=int,
                        help='Override train_conf.accum_grad (default: config)')
    parser.add_argument('--reset_checkpoint_state', action='store_true',
                        help='Ignore step/epoch/best_loss from --checkpoint (use weights only)')
    return parser.parse_args()


def _filter_state_dict_tensors(state_dict):
    """Remove non-tensor items from checkpoint state_dict."""
    return {k: v for k, v in state_dict.items() if torch.is_tensor(v)}


def _resolve_llm_eval_checkpoint(args) -> str:
    """Resolve LLM checkpoint for flow-phase TTS eval."""
    if args.llm_eval_checkpoint and os.path.exists(args.llm_eval_checkpoint):
        return args.llm_eval_checkpoint
    if args.model != 'flow':
        return ''

    llm_dir = Path(args.model_dir).parent / 'llm'
    candidates = [
        llm_dir / 'best_model.pt',
        llm_dir / 'epoch_1_whole.pt',
        llm_dir / 'epoch_0_whole.pt',
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Fall back to the latest step checkpoint if present.
    step_ckpts = sorted(llm_dir.glob('epoch_*_step_*.pt'))
    if step_ckpts:
        return str(step_ckpts[-1])

    return ''


def estimate_total_steps_from_tarlist(data_list_path: str, accum_grad: int, max_epoch: int) -> tuple[int, int]:
    """Estimate total training steps from tar list count (heuristic).

    Args:
        data_list_path: Path to data.list file
        accum_grad: Gradient accumulation steps
        max_epoch: Maximum epochs

    Returns:
        Estimated total steps
    """
    try:
        with open(data_list_path, 'r') as f:
            tar_count = sum(1 for _ in f)
        # Each tar has ~1000 samples, dynamic batching yields ~142 batches per 1000 samples
        samples_per_tar = 1000
        batches_per_1000_samples = 142  # empirical from previous runs
        batches_per_epoch = tar_count * batches_per_1000_samples
        steps_per_epoch = math.ceil(batches_per_epoch / accum_grad)
        return steps_per_epoch, steps_per_epoch * max_epoch
    except Exception:
        return 0, 0  # Return 0s if estimation fails


def estimate_steps_from_duration(train_data_list: str,
                                 dataset_csv: str,
                                 max_frames_in_batch: int,
                                 accum_grad: int,
                                 max_epoch: int,
                                 sample_rate: int,
                                 hop_size: int,
                                 shuffle_size: int = 1000,
                                 sort_size: int = 500,
                                 seed: int = 0,
                                 cache_path: str | None = None) -> tuple[int, int, dict]:
    """Estimate steps from total audio duration in the train split.

    This reads the train parquet list, maps wav paths to durations from the dataset CSV,
    and computes steps per epoch based on total frames.
    """
    if not os.path.isfile(train_data_list) or not os.path.isfile(dataset_csv):
        return 0, 0, {}

    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            if cache.get('train_data_list') == train_data_list and \
               cache.get('dataset_csv') == dataset_csv and \
               cache.get('max_frames_in_batch') == max_frames_in_batch and \
               cache.get('accum_grad') == accum_grad and \
               cache.get('max_epoch') == max_epoch and \
               cache.get('sample_rate') == sample_rate and \
               cache.get('hop_size') == hop_size and \
               cache.get('shuffle_size') == shuffle_size and \
               cache.get('sort_size') == sort_size and \
               cache.get('seed') == seed and \
               cache.get('train_data_mtime') == os.path.getmtime(train_data_list) and \
               cache.get('dataset_csv_mtime') == os.path.getmtime(dataset_csv):
                return cache.get('steps_per_epoch', 0), cache.get('total_steps', 0), cache
        except Exception:
            cache = {}

    try:
        import csv
        import pyarrow.parquet as pq
    except Exception:
        return 0, 0, {}

    dur_map = {}
    with open(dataset_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            try:
                dur_map[row['audio_file']] = float(row['duration'])
            except Exception:
                continue

    total_seconds = 0.0
    missing = 0
    rows = 0
    lengths = []
    with open(train_data_list, 'r', encoding='utf-8') as f:
        parquet_paths = [line.strip() for line in f if line.strip()]

    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            wavs = pf.read_row_group(rg, columns=['wav']).to_pydict().get('wav', [])
            for wav in wavs:
                rows += 1
                dur = dur_map.get(wav)
                if dur is None:
                    missing += 1
                    continue
                total_seconds += dur
                if hop_size:
                    frames = max(1, int(round(dur * sample_rate / hop_size)))
                    lengths.append(frames)

    frames_per_sec = sample_rate / hop_size if hop_size else 0
    total_frames = total_seconds * frames_per_sec
    if max_frames_in_batch and lengths:
        rng = random.Random(seed)
        # Shuffle buffer
        shuffled = []
        buf = []
        for length in lengths:
            buf.append(length)
            if len(buf) >= shuffle_size:
                rng.shuffle(buf)
                shuffled.extend(buf)
                buf = []
        if buf:
            rng.shuffle(buf)
            shuffled.extend(buf)

        # Sort buffer by length
        sorted_lengths = []
        buf = []
        for length in shuffled:
            buf.append(length)
            if len(buf) >= sort_size:
                buf.sort()
                sorted_lengths.extend(buf)
                buf = []
        if buf:
            buf.sort()
            sorted_lengths.extend(buf)

        # Dynamic batch simulation
        batches_per_epoch = 0
        buf = []
        longest = 0
        for length in sorted_lengths:
            new_longest = length if length > longest else longest
            frames_after = new_longest * (len(buf) + 1)
            if frames_after > max_frames_in_batch:
                if buf:
                    batches_per_epoch += 1
                buf = [length]
                longest = length
            else:
                buf.append(length)
                longest = new_longest
        if buf:
            batches_per_epoch += 1
    else:
        batches_per_epoch = math.ceil(total_frames / max_frames_in_batch) if max_frames_in_batch else 0
    steps_per_epoch = math.ceil(batches_per_epoch / accum_grad) if accum_grad else 0
    total_steps = steps_per_epoch * max_epoch

    cache = {
        'train_data_list': train_data_list,
        'dataset_csv': dataset_csv,
        'train_data_mtime': os.path.getmtime(train_data_list),
        'dataset_csv_mtime': os.path.getmtime(dataset_csv),
        'max_frames_in_batch': max_frames_in_batch,
        'accum_grad': accum_grad,
        'max_epoch': max_epoch,
        'sample_rate': sample_rate,
        'hop_size': hop_size,
        'rows': rows,
        'missing': missing,
        'total_seconds': total_seconds,
        'total_hours': total_seconds / 3600.0,
        'frames_per_sec': frames_per_sec,
        'total_frames': int(total_frames),
        'batches_per_epoch': batches_per_epoch,
        'shuffle_size': shuffle_size,
        'sort_size': sort_size,
        'seed': seed,
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
    }
    if cache_path:
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2)
        except Exception:
            pass

    return steps_per_epoch, total_steps, cache


DEFAULT_INSTRUCT = "You are a helpful assistant.<|endofprompt|>"


def load_dataset_speakers(csv_path: str, delimiter: str = '|') -> list:
    """Load speaker audio references from dataset CSV.

    Args:
        csv_path: Path to dataset CSV
        delimiter: Field separator

    Returns:
        List of (audio_path, speaker_name) tuples
    """
    speakers = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Skip header
    if lines and 'audio_file' in lines[0].lower():
        lines = lines[1:]

    for line in lines:
        parts = line.strip().split(delimiter)
        if len(parts) >= 3 and os.path.exists(parts[0]):
            speakers.append((parts[0], parts[2]))  # audio_path, speaker

    return speakers


def get_random_speaker_ref(speakers: list, rng: random.Random) -> tuple:
    """Get random speaker reference audio.

    Args:
        speakers: List of (audio_path, speaker) tuples
        rng: Random number generator

    Returns:
        (audio_path, speaker_name) tuple
    """
    return rng.choice(speakers)


def format_prompt_text(text: str) -> str:
    prompt_text = (text or '').strip()
    if not prompt_text:
        return '<|endofprompt|>'
    if '<|endofprompt|>' not in prompt_text:
        prompt_text = f'{prompt_text} <|endofprompt|>'
    return prompt_text


def find_latest_step_checkpoint(resume_path: str) -> str | None:
    if os.path.isfile(resume_path):
        return resume_path

    candidates = []
    patterns = [
        os.path.join(resume_path, 'epoch_*_step_*.pt'),
        os.path.join(resume_path, 'checkpoints', 'epoch_*_step_*.pt'),
        os.path.join(resume_path, 'step_*.pt'),
    ]
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    if not candidates:
        return None

    def get_step(path: str) -> int:
        parts = Path(path).stem.split('_')
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return -1

    return max(candidates, key=get_step)


def manage_rolling_checkpoints(checkpoint_dir: str, max_keep: int = 3):
    """Manage rolling checkpoints, keeping only the most recent checkpoints in total.

    Args:
        checkpoint_dir: Directory containing checkpoints
        max_keep: Maximum checkpoints to keep (includes epoch, step, and best_model)
    """
    patterns = [
        os.path.join(checkpoint_dir, '*.pt'),
        os.path.join(checkpoint_dir, 'checkpoints', '*.pt'),
    ]
    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(glob.glob(pattern))

    protected = set()
    best_model = Path(checkpoint_dir) / 'best_model.pt'
    if best_model.exists():
        protected.add(str(best_model))

    checkpoints = [
        ckpt for ckpt in checkpoints
        if Path(ckpt).name not in ('init.pt',) and ckpt not in protected
    ]

    checkpoints = sorted(checkpoints, key=os.path.getmtime)

    keep_count = max(max_keep - len(protected), 0)
    keep = set(checkpoints[-keep_count:]) if keep_count > 0 else set()
    keep |= protected

    for old_ckpt in list(checkpoints):
        if old_ckpt in keep:
            continue
        logger.info(f"Removing old checkpoint: {old_ckpt}")
        try:
            os.remove(old_ckpt)
            yaml_file = old_ckpt.replace('.pt', '.yaml')
            if os.path.exists(yaml_file):
                os.remove(yaml_file)
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint: {e}")


def save_checkpoint(model, step: int, epoch: int, save_dir: str,
                    prefix: str = "step", best_loss: float = None):
    """Save a checkpoint with training state.

    Args:
        model: Model (DDP wrapped)
        step: Current step
        epoch: Current epoch
        save_dir: Directory to save checkpoint
        prefix: Checkpoint name prefix
        best_loss: Best eval loss so far
    """
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, f'{prefix}_{step}.pt')

    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    state_dict['step'] = step
    state_dict['epoch'] = epoch
    if best_loss is not None:
        state_dict['best_loss'] = best_loss

    torch.save(state_dict, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


class CzechExecutor(Executor):
    """Extended executor with Czech-specific evaluation and TTS generation."""

    def __init__(self, args, speakers: list, sample_rate: int = 24000):
        """Initialize Czech executor.

        Args:
            args: Command line arguments
            speakers: List of (audio_path, speaker) tuples for TTS refs
            sample_rate: Audio sample rate
        """
        super().__init__(gan=False)
        self.args = args
        self.speakers = speakers
        self.sample_rate = sample_rate
        self.best_loss = float('inf')
        self.eval_step_count = 0
        self.cosyvoice_model = None  # Set externally
        self.model_type = 'llm'  # Set externally

    def sync_weights_to_inference(self, training_model):
        """Sync training weights to inference model for TTS eval."""
        if self.cosyvoice_model is None:
            return

        # Get training state dict
        if hasattr(training_model, 'module'):
            state_dict = training_model.module.state_dict()
        else:
            state_dict = training_model.state_dict()

        # Load into inference model's component
        try:
            if self.model_type == 'llm':
                self.cosyvoice_model.model.llm.load_state_dict(state_dict, strict=False)
            elif self.model_type == 'flow':
                self.cosyvoice_model.model.flow.load_state_dict(state_dict, strict=False)
            logger.info(f"Synced {self.model_type} weights to inference model")
        except Exception as e:
            logger.warning(f"Failed to sync weights: {e}")

    def save_best_model(self, model, cv_loss: float):
        """Save model if it's the best so far."""
        if cv_loss < self.best_loss:
            self.best_loss = cv_loss
            save_path = os.path.join(self.args.model_dir, 'best_model.pt')

            if hasattr(model, 'module'):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict['best_loss'] = cv_loss
            state_dict['step'] = self.step
            state_dict['epoch'] = self.epoch
            torch.save(state_dict, save_path)
            logger.info(f"New best model saved! Loss: {cv_loss:.6f} at step {self.step}")

    @torch.inference_mode()
    def save_step_checkpoint(self, model, info_dict):
        super().save_step_checkpoint(model, info_dict)
        if self.rank == 0:
            max_keep = getattr(self.args, 'max_checkpoints', 3)
            manage_rolling_checkpoints(self.args.model_dir, max_keep)

    def on_step_end(self, model, info_dict):
        """Check and run TTS evaluation at configured intervals.

        This hook is called after each training step, independent of cv().
        """
        if self.rank != 0 or self.cosyvoice_model is None:
            return

        tts_interval = getattr(self.args, 'tts_eval_per_step', 0)
        if tts_interval > 0 and self.step > 0 and (self.step + 1) % tts_interval == 0:
            self.run_evaluation_tts(model, self.step)

    def run_evaluation_tts(self, model, step: int):
        """Run TTS generation during evaluation.

        Args:
            model: Training model (used for sync)
            step: Current training step
        """
        if self.rank != 0:
            return

        if self.cosyvoice_model is None:
            logger.warning("CosyVoice model not loaded, skipping TTS evaluation")
            return

        # Sync training weights to inference model
        self.sync_weights_to_inference(model)

        output_dir = os.path.join(
            self.args.model_dir, 'audio_samples', f'eval_step_{step}'
        )
        os.makedirs(output_dir, exist_ok=True)

        rng = random.Random(step)  # Different random refs each eval

        sentences = get_eval_sentences()
        logger.info(f"Generating {len(sentences)} TTS samples at step {step}...")

        for idx, sentence in enumerate(sentences):
            try:
                # Get random speaker reference
                ref_audio, ref_speaker = get_random_speaker_ref(self.speakers, rng)
                ref_speaker_safe = ref_speaker.replace(' ', '_').replace('/', '_')[:30]
                ref_basename = Path(ref_audio).stem[:20]
                prompt_text = format_prompt_text(DEFAULT_INSTRUCT)

                # Use instruct2 to avoid passing prompt speech tokens into the LLM,
                # which is out-of-distribution for SFT-style fine-tuning.
                for result in self.cosyvoice_model.inference_instruct2(
                    tts_text=sentence,
                    instruct_text=prompt_text,
                    prompt_wav=ref_audio,
                    stream=False,
                    speed=1.0,
                    # Keep the sentence intact (no Czech comma-based splitting).
                    text_frontend=False
                ):
                    tts_speech = result['tts_speech']
                    break  # Just get first result

                # Save audio
                output_name = f"sample_{idx+1:02d}_{ref_speaker_safe}_ref-{ref_basename}.wav"
                output_path = os.path.join(output_dir, output_name)
                torchaudio.save(output_path, tts_speech.cpu(), self.sample_rate)
                logger.info(f"Generated: {output_name}")

            except Exception as e:
                logger.error(f"Failed to generate sample {idx}: {e}")
                continue

        # Clean up GPU memory after TTS
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"TTS samples saved to {output_dir}")

    @torch.inference_mode()
    def compute_flow_mel_l1(self, model, cv_data_loader):
        """Compute mean L1 mel error for flow on a small CV subset."""
        if self.model_type != 'flow' or self.rank != 0:
            return None
        max_utts = max(0, getattr(self.args, 'flow_mel_l1_utts', 0))
        if max_utts == 0:
            return None

        flow_model = model.module if hasattr(model, 'module') else model
        flow_model.eval()

        try:
            device = next(flow_model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

        total_l1 = 0.0
        processed = 0

        for batch in cv_data_loader:
            if processed >= max_utts:
                break
            if 'speech_token' not in batch or 'speech_feat' not in batch:
                logger.warning("CV batch missing speech_token/speech_feat; skipping L1 mel metric")
                break

            token = batch['speech_token'][:1].to(device)
            token_len = batch['speech_token_len'][:1].to(device)
            feat = batch['speech_feat'][:1].to(device)
            feat_len = batch['speech_feat_len'][:1].to(device)
            embedding = batch['embedding'][:1].to(device)

            tlen = int(token_len[0].item())
            flen = int(feat_len[0].item())
            if tlen <= 0 or flen <= 0:
                continue

            token = token[:, :tlen]
            feat = feat[:, :flen, :]
            token_len = torch.tensor([tlen], dtype=torch.int32, device=device)
            feat_len = torch.tensor([flen], dtype=torch.int32, device=device)

            prompt_token = token.new_zeros((1, 0))
            prompt_token_len = token_len.new_zeros((1,))
            prompt_feat = feat.new_zeros((1, 0, feat.shape[2]))
            prompt_feat_len = feat_len.new_zeros((1,))

            pred_feat, _ = flow_model.inference(
                token=token,
                token_len=token_len,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
                streaming=False,
                finalize=True
            )
            pred_feat = pred_feat.transpose(1, 2)  # [B, T, 80]

            min_len = min(pred_feat.shape[1], feat.shape[1])
            if min_len <= 0:
                continue
            l1 = torch.mean(torch.abs(pred_feat[:, :min_len, :] - feat[:, :min_len, :]))
            total_l1 += l1.item()
            processed += 1

        if processed == 0:
            return None
        return total_l1 / processed

    @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True):
        """Override cv() to add TTS generation, best model saving, and checkpoint cleanup.

        This is called by train_one_epoc at evaluation checkpoints.
        """
        # Call parent cv() method for loss computation and checkpoint saving
        super().cv(model, cv_data_loader, writer, info_dict, on_batch_end)

        # Get eval loss from info_dict
        cv_loss = info_dict['loss_dict'].get('loss', float('inf'))
        if hasattr(cv_loss, 'item'):
            cv_loss = cv_loss.item()

        # Save best model if loss improved
        if self.rank == 0:
            self.save_best_model(model, cv_loss)

            # Clean up old checkpoints (keep only max_checkpoints total)
            max_keep = getattr(self.args, 'max_checkpoints', 3)
            checkpoint_dir = self.args.model_dir
            manage_rolling_checkpoints(checkpoint_dir, max_keep)

        # Optional flow mel L1 metric
        if self.model_type == 'flow' and self.rank == 0:
            flow_l1 = self.compute_flow_mel_l1(model, cv_data_loader)
            if flow_l1 is not None:
                if writer is not None:
                    writer.add_scalar('CV/flow_mel_l1', flow_l1, self.step + 1)
                logger.info(f"Flow mel L1 (avg over {getattr(self.args, 'flow_mel_l1_utts', 0)} utts): {flow_l1:.6f}")

        # Run TTS generation at epoch end if configured.
        # Note: Step-based TTS is handled by on_step_end() hook, not here.
        tts_epoch_interval = int(getattr(self.args, 'tts_eval_per_epoch', 0) or 0)
        run_tts_epoch = tts_epoch_interval > 0 and on_batch_end and \
            ((self.epoch + 1) % tts_epoch_interval == 0)
        if run_tts_epoch and self.rank == 0:
            self.run_evaluation_tts(model, self.step)


@record
def main():
    """Main training function."""
    args = get_args()
    mamba_env = os.environ.get('COSYVOICE_MAMBA_ENV')
    if mamba_env:
        logger.info(f"Micromamba env: {mamba_env}")

    # Set up tensorboard dir
    if args.tensorboard_dir is None:
        args.tensorboard_dir = os.path.join(args.model_dir, 'tensorboard')

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.join(args.model_dir, 'checkpoints'), exist_ok=True)

    # Override dict to only load the model we're training
    override_dict = {k: None for k in ['llm', 'flow', 'hift'] if k != args.model}

    # Load config
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={
            **override_dict,
            'qwen_pretrain_path': args.qwen_pretrain_path
        })

    if args.max_frames_in_batch is not None and args.max_frames_in_batch > 0:
        batch_cfg = configs.get('batch')
        if isinstance(batch_cfg, functools.partial):
            if batch_cfg.keywords is None:
                batch_cfg.keywords = {}
            batch_cfg.keywords['max_frames_in_batch'] = args.max_frames_in_batch
        elif isinstance(batch_cfg, dict):
            batch_cfg['max_frames_in_batch'] = args.max_frames_in_batch

    if args.max_epoch is not None and args.max_epoch > 0:
        configs['train_conf']['max_epoch'] = args.max_epoch

    if args.lr is not None and args.lr > 0:
        optim_conf = configs['train_conf'].get('optim_conf', {})
        if isinstance(optim_conf, functools.partial):
            if optim_conf.keywords is None:
                optim_conf.keywords = {}
            optim_conf.keywords['lr'] = args.lr
        elif isinstance(optim_conf, dict):
            optim_conf['lr'] = args.lr
        else:
            configs['train_conf']['optim_conf'] = {'lr': args.lr}

    if args.accum_grad is not None and args.accum_grad > 0:
        configs['train_conf']['accum_grad'] = args.accum_grad

    # Override save_per_step from args
    configs['train_conf']['save_per_step'] = args.save_per_step
    args_dict = vars(args).copy()
    if args.accum_grad is None:
        args_dict.pop('accum_grad', None)
    configs['train_conf'].update(args_dict)

    # Init distributed
    init_distributed(args)
    rank = int(os.environ.get('RANK', 0))

    # Load dataset speakers for TTS eval
    speakers = load_dataset_speakers(args.dataset_csv)
    logger.info(f"Loaded {len(speakers)} speaker references for TTS eval")

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan=False, dpo=False)

    # Check and save config
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard
    writer = init_summarywriter(args)

    # Load model
    model = configs[args.model]
    start_step, start_epoch = 0, -1
    best_loss = float('inf')

    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        if args.reset_checkpoint_state:
            logger.info(f"Loaded checkpoint weights from {args.checkpoint} (reset step/epoch/best_loss)")
        else:
            if 'step' in state_dict:
                start_step = state_dict['step']
            if 'epoch' in state_dict:
                start_epoch = state_dict['epoch']
            if 'best_loss' in state_dict:
                best_loss = state_dict['best_loss']
            logger.info(f"Loaded checkpoint from {args.checkpoint}, step={start_step}, epoch={start_epoch}")

    # Resume from directory if specified
    if args.resume and os.path.exists(args.resume):
        latest = find_latest_step_checkpoint(args.resume)
        if latest:
            state_dict = torch.load(latest, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            start_step = state_dict.get('step', 0)
            start_epoch = state_dict.get('epoch', -1)
            best_loss = state_dict.get('best_loss', float('inf'))
            logger.info(f"Resumed from {latest}, step={start_step}, epoch={start_epoch}")

    # Wrap model
    model = wrap_cuda_model(args, model)

    # Optimizer and scheduler
    model, optimizer, scheduler, _, _ = init_optimizer_and_scheduler(args, configs, model, gan=False)
    scheduler.set_step(start_step)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # Save initial checkpoint
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    def _cfg_kwargs(cfg_val):
        if isinstance(cfg_val, dict):
            return cfg_val
        if isinstance(cfg_val, functools.partial):
            return cfg_val.keywords or {}
        return {}

    batch_kw = _cfg_kwargs(configs.get('batch'))
    max_frames = batch_kw.get('max_frames_in_batch', 0)

    feat_kw = _cfg_kwargs(configs.get('feat_extractor'))
    hop_size = feat_kw.get('hop_size', 0)

    shuffle_kw = _cfg_kwargs(configs.get('shuffle'))
    shuffle_size = shuffle_kw.get('shuffle_size', 1000)

    sort_kw = _cfg_kwargs(configs.get('sort'))
    sort_size = sort_kw.get('sort_size', 500)
    cache_path = os.path.join(args.model_dir, 'steps_estimate.json')
    steps_per_epoch_est, total_steps_est, est_meta = estimate_steps_from_duration(
        args.train_data,
        args.dataset_csv,
        max_frames,
        configs['train_conf']['accum_grad'],
        configs['train_conf']['max_epoch'],
        configs.get('sample_rate', 0),
        hop_size,
        shuffle_size=shuffle_size,
        sort_size=sort_size,
        cache_path=cache_path,
    )
    if total_steps_est == 0:
        steps_per_epoch_est, total_steps_est = estimate_total_steps_from_tarlist(
            args.train_data, configs['train_conf']['accum_grad'], configs['train_conf']['max_epoch']
        )
        est_meta = {}
        est_source = 'tarlist'
    else:
        est_source = 'duration'

    info_dict['steps_per_epoch_est'] = steps_per_epoch_est
    info_dict['total_steps'] = total_steps_est
    info_dict['total_steps_est'] = total_steps_est
    info_dict['steps_estimate_source'] = est_source
    if est_meta:
        info_dict['steps_estimate_rows'] = est_meta.get('rows', 0)
        info_dict['steps_estimate_missing'] = est_meta.get('missing', 0)
    save_model(model, 'init', info_dict)

    # Create executor
    executor = CzechExecutor(args, speakers, configs['sample_rate'])
    executor.step = start_step
    executor.best_loss = best_loss
    executor.model_type = args.model

    # Load CosyVoice3 model for TTS evaluation (only on rank 0)
    # CosyVoice3 expects the model dir with cosyvoice3.yaml, which is parent of qwen_pretrain_path
    if rank == 0 and (args.tts_eval_per_step > 0 or args.tts_eval_per_epoch > 0):
        cosyvoice_model_dir = str(Path(args.qwen_pretrain_path).parent)
        logger.info(f"Loading CosyVoice3 model from {cosyvoice_model_dir} for TTS evaluation...")
        try:
            cosyvoice_model = CosyVoice3(cosyvoice_model_dir)
            executor.cosyvoice_model = cosyvoice_model
            if args.model == 'flow':
                llm_eval_ckpt = _resolve_llm_eval_checkpoint(args)
                if llm_eval_ckpt:
                    logger.info(f"Loading LLM eval checkpoint for flow TTS: {llm_eval_ckpt}")
                    state_dict = torch.load(llm_eval_ckpt, map_location='cpu')
                    state_dict = _filter_state_dict_tensors(state_dict)
                    missing, unexpected = executor.cosyvoice_model.model.llm.load_state_dict(
                        state_dict, strict=False
                    )
                    if missing:
                        logger.info(f"LLM eval checkpoint missing keys: {len(missing)}")
                    if unexpected:
                        logger.info(f"LLM eval checkpoint unexpected keys: {len(unexpected)}")
                else:
                    logger.warning("No LLM eval checkpoint found for flow TTS; using base LLM.")
            if args.model == 'llm' and args.flow_eval_checkpoint:
                if os.path.exists(args.flow_eval_checkpoint):
                    logger.info(f"Loading Flow eval checkpoint for LLM TTS: {args.flow_eval_checkpoint}")
                    state_dict = torch.load(args.flow_eval_checkpoint, map_location='cpu')
                    state_dict = _filter_state_dict_tensors(state_dict)
                    missing, unexpected = executor.cosyvoice_model.model.flow.load_state_dict(
                        state_dict, strict=False
                    )
                    if missing:
                        logger.info(f"Flow eval checkpoint missing keys: {len(missing)}")
                    if unexpected:
                        logger.info(f"Flow eval checkpoint unexpected keys: {len(unexpected)}")
                else:
                    logger.warning(f"Flow eval checkpoint not found: {args.flow_eval_checkpoint}")
            logger.info("CosyVoice3 model loaded successfully for TTS evaluation")
        except Exception as e:
            logger.warning(f"Failed to load CosyVoice3 for TTS eval: {e}")
            logger.warning("TTS evaluation will be disabled")

    logger.info(f'Starting training from step {start_step}, epoch {start_epoch + 1}')
    if est_meta:
        logger.info(
            "Estimated steps/epoch from duration: %s (total_steps %s, hours %.2f, missing %s, rows %s)",
            info_dict["steps_per_epoch_est"],
            info_dict["total_steps"],
            est_meta.get('total_hours', 0.0),
            est_meta.get('missing', 0),
            est_meta.get('rows', 0),
        )
    else:
        logger.info(
            "Estimated steps/epoch from tarlist heuristic: %s (total_steps %s)",
            info_dict["steps_per_epoch_est"],
            info_dict["total_steps"],
        )
    logger.info(f'Save every {args.save_per_step} steps, eval every {args.eval_per_step} steps')
    logger.info(f'Eval every {args.eval_per_epoch} epochs')
    logger.info(f'TTS eval every {args.tts_eval_per_step} steps')
    logger.info(f'TTS eval every {args.tts_eval_per_epoch} epochs')
    logger.info(f'Max rolling checkpoints (total): {args.max_checkpoints}')
    if args.max_steps and args.max_steps > 0:
        logger.info(f"Max steps enabled: {args.max_steps}")
    if args.accum_grad is not None and args.accum_grad > 0:
        logger.info(f"Accum grad override: {args.accum_grad}")

    # Initial evaluation at step 0
    if start_step == 0:
        logger.info("Running initial evaluation at step 0...")
        dist.barrier()
        executor.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
        # Run initial TTS if enabled
        if rank == 0 and executor.cosyvoice_model is not None and \
                (args.tts_eval_per_step > 0 or args.tts_eval_per_epoch > 0):
            executor.run_evaluation_tts(model, 0)

    # Training loop
    for epoch in range(start_epoch + 1, configs['train_conf']['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()

        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))

        # Use existing train method but add our checkpoint management
        executor.train_one_epoc(
            model, optimizer, scheduler,
            train_data_loader, cv_data_loader,
            writer, info_dict, scaler, group_join
        )

        dist.destroy_process_group(group_join)

        # End of epoch - save checkpoint
        if rank == 0:
            ckpt_dir = os.path.join(args.model_dir, 'checkpoints')
            save_checkpoint(model, executor.step, epoch, ckpt_dir,
                          prefix=f"epoch_{epoch}", best_loss=executor.best_loss)
            manage_rolling_checkpoints(args.model_dir, args.max_checkpoints)
        if getattr(executor, 'stop_training', False):
            logger.info("Stopping training early due to max_steps.")
            break

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
