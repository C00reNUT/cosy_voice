#!/usr/bin/env python3
"""Czech fine-tuning script for Fun-CosyVoice3-0.5B-2512.

Features:
- LLM + Flow joint SFT training
- Rolling checkpoints (max 3, every 500 steps)
- Evaluation every 5000 steps with TTS generation
- Best model tracking by eval loss
- Full resume capability

Micromamba env: cosyvoice

Usage:
    torchrun --nproc_per_node=1 train_czech.py \\
        --config conf/cosyvoice3_czech.yaml \\
        --train_data data/train/parquet/data.list \\
        --cv_data data/eval/parquet/data.list \\
        --model_dir /mnt/8TB/TRAINING_OUTPUTS/...
"""
from __future__ import print_function
import argparse
import datetime
import logging
import os
import sys
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

from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config,
    batch_forward, log_per_step
)

# Import local eval sentences
sys.path.insert(0, str(Path(__file__).parent))
from local.eval_sentences import get_eval_sentences

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

EVAL_SENTENCES = get_eval_sentences()


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
    parser.add_argument('--eval_per_step', default=5000, type=int, help='Eval every N steps')
    parser.add_argument('--max_checkpoints', default=3, type=int, help='Max rolling checkpoints')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint dir')
    parser.add_argument('--timeout', default=60, type=int, help='Timeout for joins')
    return parser.parse_args()


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


def manage_rolling_checkpoints(checkpoint_dir: str, max_keep: int = 3):
    """Manage rolling checkpoints, keeping only the most recent.

    Args:
        checkpoint_dir: Directory containing checkpoints
        max_keep: Maximum checkpoints to keep
    """
    # Find step checkpoints (not eval or best)
    pattern = os.path.join(checkpoint_dir, 'step_*.pt')
    checkpoints = sorted(glob.glob(pattern), key=lambda x: int(Path(x).stem.split('_')[-1]))

    # Remove old checkpoints
    while len(checkpoints) > max_keep:
        old_ckpt = checkpoints.pop(0)
        logger.info(f"Removing old checkpoint: {old_ckpt}")
        try:
            os.remove(old_ckpt)
            # Also remove yaml file if exists
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

    def run_evaluation_tts(self, model, cosyvoice_model, step: int):
        """Run TTS generation during evaluation.

        Args:
            model: Training model
            cosyvoice_model: Full CosyVoice model for inference
            step: Current training step
        """
        if self.rank != 0:
            return

        output_dir = os.path.join(
            self.args.model_dir, 'audio_samples', f'eval_step_{step}'
        )
        os.makedirs(output_dir, exist_ok=True)

        rng = random.Random(step)  # Different random refs each eval

        logger.info(f"Generating {len(EVAL_SENTENCES)} TTS samples...")

        for idx, sentence in enumerate(EVAL_SENTENCES):
            try:
                # Get random speaker reference
                ref_audio, ref_speaker = get_random_speaker_ref(self.speakers, rng)
                ref_speaker_safe = ref_speaker.replace(' ', '_').replace('/', '_')[:30]
                ref_basename = Path(ref_audio).stem[:20]

                # Use the inference method
                for result in cosyvoice_model.inference_zero_shot(
                    tts_text=sentence,
                    prompt_text="",  # No prompt text needed
                    prompt_wav=ref_audio,
                    stream=False,
                    speed=1.0,
                    text_frontend=True
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

        logger.info(f"TTS samples saved to {output_dir}")


@record
def main():
    """Main training function."""
    args = get_args()

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

    # Override save_per_step from args
    configs['train_conf']['save_per_step'] = args.save_per_step
    configs['train_conf'].update(vars(args))

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
        if 'step' in state_dict:
            start_step = state_dict['step']
        if 'epoch' in state_dict:
            start_epoch = state_dict['epoch']
        if 'best_loss' in state_dict:
            best_loss = state_dict['best_loss']
        logger.info(f"Loaded checkpoint from {args.checkpoint}, step={start_step}, epoch={start_epoch}")

    # Resume from directory if specified
    if args.resume and os.path.exists(args.resume):
        # Find latest checkpoint
        checkpoints = glob.glob(os.path.join(args.resume, 'step_*.pt'))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(Path(x).stem.split('_')[-1]))
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
    save_model(model, 'init', info_dict)

    # Create executor
    executor = CzechExecutor(args, speakers, configs['sample_rate'])
    executor.step = start_step
    executor.best_loss = best_loss

    logger.info(f'Starting training from step {start_step}, epoch {start_epoch + 1}')
    logger.info(f'Save every {args.save_per_step} steps, eval every {args.eval_per_step} steps')
    logger.info(f'Max rolling checkpoints: {args.max_checkpoints}')

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

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
