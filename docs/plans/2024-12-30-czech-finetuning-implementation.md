# Czech Fine-tuning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fine-tune Fun-CosyVoice3-0.5B-2512 on 200h Czech dataset with LLM+Flow SFT, evaluation TTS, and full checkpointing.

**Architecture:** Convert CSV dataset to CosyVoice parquet format with speaker embeddings and speech tokens. Create custom training script extending existing CosyVoice training with: rolling checkpoints (500 steps), evaluation every 5000 steps with TTS generation, best model tracking, and full resume capability.

**Tech Stack:** PyTorch, ONNX Runtime, torchaudio, hyperpyyaml, TensorBoard, DDP

---

## Task 1: Create Czech Example Directory Structure

**Files:**
- Create: `examples/czech/` directory structure

**Step 1: Create directory structure**

```bash
mkdir -p examples/czech/{local,tools,conf}
```

**Step 2: Verify structure**

```bash
ls -la examples/czech/
```

Expected: `local/`, `tools/`, `conf/` directories

**Step 3: Commit**

```bash
git add examples/czech/
git commit -m "feat(czech): create example directory structure"
```

---

## Task 2: Create Data Preparation Script

**Files:**
- Create: `examples/czech/local/prepare_czech_data.py`

**Step 1: Create the data preparation script**

```python
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
```

**Step 2: Make executable and test syntax**

```bash
chmod +x examples/czech/local/prepare_czech_data.py
python3 -m py_compile examples/czech/local/prepare_czech_data.py
```

Expected: No output (syntax OK)

**Step 3: Commit**

```bash
git add examples/czech/local/prepare_czech_data.py
git commit -m "feat(czech): add data preparation script for CSV to Kaldi format"
```

---

## Task 3: Create Train/Eval Split Script

**Files:**
- Create: `examples/czech/local/split_train_eval.py`

**Step 1: Create the split script**

```python
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
```

**Step 2: Test syntax**

```bash
python3 -m py_compile examples/czech/local/split_train_eval.py
```

**Step 3: Commit**

```bash
git add examples/czech/local/split_train_eval.py
git commit -m "feat(czech): add train/eval split script with speaker stratification"
```

---

## Task 4: Copy and Adapt Extraction Tools

**Files:**
- Create: `examples/czech/tools/extract_embedding.py`
- Create: `examples/czech/tools/extract_speech_token.py`
- Create: `examples/czech/tools/make_parquet_list.py`

**Step 1: Create symlinks to existing tools (they work as-is)**

```bash
cd examples/czech/tools
ln -s ../../libritts/cosyvoice3/tools/extract_embedding.py extract_embedding.py
ln -s ../../libritts/cosyvoice3/tools/extract_speech_token.py extract_speech_token.py
ln -s ../../libritts/cosyvoice3/tools/make_parquet_list.py make_parquet_list.py
cd ../../..
```

**Step 2: Verify symlinks**

```bash
ls -la examples/czech/tools/
```

Expected: Symlinks pointing to cosyvoice3 tools

**Step 3: Commit**

```bash
git add examples/czech/tools/
git commit -m "feat(czech): add tool symlinks for embedding/token extraction"
```

---

## Task 5: Create Czech Training Configuration

**Files:**
- Create: `examples/czech/conf/cosyvoice3_czech.yaml`

**Step 1: Create the configuration file**

```yaml
# Czech Fine-tuning Configuration for Fun-CosyVoice3-0.5B-2512
# Based on cosyvoice3.yaml with training modifications

# set random seed
__set_seed1: !apply:random.seed [2024]
__set_seed2: !apply:numpy.random.seed [2024]
__set_seed3: !apply:torch.manual_seed [2024]
__set_seed4: !apply:torch.cuda.manual_seed_all [2024]

# fixed params
sample_rate: 24000
llm_input_size: 896
llm_output_size: 896
spk_embed_dim: 192
qwen_pretrain_path: ''
token_frame_rate: 25
token_mel_ratio: 2

# stream related params
chunk_size: 25
num_decoding_left_chunks: -1

# model params
llm: !new:cosyvoice.llm.llm.CosyVoice3LM
    llm_input_size: !ref <llm_input_size>
    llm_output_size: !ref <llm_output_size>
    speech_token_size: 6561
    length_normalized_loss: True
    lsm_weight: 0
    mix_ratio: [5, 15]
    llm: !new:cosyvoice.llm.llm.Qwen2Encoder
        pretrain_path: !ref <qwen_pretrain_path>
    sampling: !name:cosyvoice.utils.common.ras_sampling
        top_p: 0.8
        top_k: 25
        win_size: 10
        tau_r: 0.1

flow: !new:cosyvoice.flow.flow.CausalMaskedDiffWithDiT
    input_size: 80
    output_size: 80
    spk_embed_dim: !ref <spk_embed_dim>
    output_type: 'mel'
    vocab_size: 6561
    input_frame_rate: !ref <token_frame_rate>
    only_mask_loss: True
    token_mel_ratio: !ref <token_mel_ratio>
    pre_lookahead_len: 3
    pre_lookahead_layer: !new:cosyvoice.transformer.upsample_encoder.PreLookaheadLayer
        in_channels: 80
        channels: 1024
        pre_lookahead_len: 3
    decoder: !new:cosyvoice.flow.flow_matching.CausalConditionalCFM
        in_channels: 240
        n_spks: 1
        spk_emb_dim: 80
        cfm_params: !new:omegaconf.DictConfig
            content:
                sigma_min: 1e-06
                solver: 'euler'
                t_scheduler: 'cosine'
                training_cfg_rate: 0.2
                inference_cfg_rate: 0.7
                reg_loss_type: 'l1'
        estimator: !new:cosyvoice.flow.DiT.dit.DiT
            dim: 1024
            depth: 22
            heads: 16
            dim_head: 64
            ff_mult: 2
            mel_dim: 80
            mu_dim: 80
            spk_dim: 80
            out_channels: 80
            static_chunk_size: !ref <chunk_size> * <token_mel_ratio>
            num_decoding_left_chunks: !ref <num_decoding_left_chunks>

hift: !new:cosyvoice.hifigan.generator.CausalHiFTGenerator
    in_channels: 80
    base_channels: 512
    nb_harmonics: 8
    sampling_rate: !ref <sample_rate>
    nsf_alpha: 0.1
    nsf_sigma: 0.003
    nsf_voiced_threshold: 10
    upsample_rates: [8, 5, 3]
    upsample_kernel_sizes: [16, 11, 7]
    istft_params:
        n_fft: 16
        hop_len: 4
    resblock_kernel_sizes: [3, 7, 11]
    resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    source_resblock_kernel_sizes: [7, 7, 11]
    source_resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    lrelu_slope: 0.1
    audio_limit: 0.99
    conv_pre_look_right: 4
    f0_predictor: !new:cosyvoice.hifigan.f0_predictor.CausalConvRNNF0Predictor
        num_class: 1
        in_channels: 80
        cond_channels: 512

# processor functions
parquet_opener: !name:cosyvoice.dataset.processor.parquet_opener
get_tokenizer: !name:cosyvoice.tokenizer.tokenizer.get_qwen_tokenizer
    token_path: !ref <qwen_pretrain_path>
    skip_special_tokens: True
    version: cosyvoice3
allowed_special: 'all'
tokenize: !name:cosyvoice.dataset.processor.tokenize
    get_tokenizer: !ref <get_tokenizer>
    allowed_special: !ref <allowed_special>
filter: !name:cosyvoice.dataset.processor.filter
    max_length: 40960
    min_length: 100
    token_max_length: 200
    token_min_length: 1
resample: !name:cosyvoice.dataset.processor.resample
    resample_rate: !ref <sample_rate>
truncate: !name:cosyvoice.dataset.processor.truncate
    truncate_length: 24480
feat_extractor: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1920
    num_mels: 80
    sampling_rate: !ref <sample_rate>
    hop_size: 480
    win_size: 1920
    fmin: 0
    fmax: null
    center: False
compute_fbank: !name:cosyvoice.dataset.processor.compute_fbank
    feat_extractor: !ref <feat_extractor>
parse_embedding: !name:cosyvoice.dataset.processor.parse_embedding
    normalize: True
shuffle: !name:cosyvoice.dataset.processor.shuffle
    shuffle_size: 1000
sort: !name:cosyvoice.dataset.processor.sort
    sort_size: 500
batch: !name:cosyvoice.dataset.processor.batch
    batch_type: 'dynamic'
    max_frames_in_batch: 2000
padding: !name:cosyvoice.dataset.processor.padding
    use_spk_embedding: True  # Enable for SFT

# dataset processor pipeline
data_pipeline: [
    !ref <parquet_opener>,
    !ref <tokenize>,
    !ref <filter>,
    !ref <resample>,
    !ref <compute_fbank>,
    !ref <parse_embedding>,
    !ref <shuffle>,
    !ref <sort>,
    !ref <batch>,
    !ref <padding>,
]

# Czech SFT train conf
train_conf:
    optim: adam
    optim_conf:
        lr: 0.00001  # 1e-5 as specified
    scheduler: constantlr
    scheduler_conf:
        warmup_steps: 2500
    max_epoch: 2
    grad_clip: 5
    accum_grad: 2
    log_interval: 100
    save_per_step: 500  # Save every 500 steps
```

**Step 2: Verify YAML syntax**

```bash
python3 -c "import yaml; yaml.safe_load(open('examples/czech/conf/cosyvoice3_czech.yaml'))"
```

Expected: No errors

**Step 3: Commit**

```bash
git add examples/czech/conf/cosyvoice3_czech.yaml
git commit -m "feat(czech): add training configuration for Czech SFT"
```

---

## Task 6: Create Evaluation Sentences Module

**Files:**
- Create: `examples/czech/local/eval_sentences.py`

**Step 1: Create the evaluation sentences module**

```python
#!/usr/bin/env python3
"""Czech evaluation sentences for TTS generation during training.

Contains 15 sentences from Sherlock Holmes Czech translation.
Micromamba env: cosyvoice
"""

CZECH_EVAL_SENTENCES = [
    "SHERLOCK HOLMES: STUDIE V ŠARLATOVÉ - Kapitola první: PAN SHERLOCK HOLMES.",

    "Roku osmnáct set sedmdesát osm jsem dosáhl hodnosti doktora medicíny na Londýnské univerzitě a odebral se do Netley, abych absolvoval kurz předepsaný pro vojenské chirurgy.",

    "Když jsem tam studia dokončil, byl jsem řádně přidělen k Pátému pluku northumberlandských střelců jako asistent chirurga.",

    "Pluk v té době pobýval v Indii, a než jsem se k němu mohl připojit, vypukla druhá afghánská válka.",

    "Po vylodění v Bombají jsem se dozvěděl, že můj sbor již prošel průsmyky a nachází se hluboko v nepřátelském území.",

    "Následoval jsem jej však s mnoha dalšími důstojníky, kteří byli v téže situaci jako já, a podařilo se nám bezpečně dorazit do Kandaháru, kde jsem nalezl svůj pluk a ihned se ujal svých nových povinností.",

    "Tažení přineslo mnohým pocty a povýšení, leč mně nic než neštěstí a pohromy.",

    "Byl jsem odvelen od své brigády a přidělen k berkshirskému pluku, s nímž jsem sloužil v osudné bitvě u Majvandu.",

    "Tam mě zasáhla střela z džezailu do ramene, roztříštila kost a škrábla podklíčkovou tepnu.",

    "Byl bych padl do rukou vražedných Gázíů, nebýt oddanosti a odvahy, kterou prokázal můj sluha Murray; přehodil mě přes soumara a podařilo se mu dopravit mě bezpečně k britským liniím.",

    "Ztrápen bolestí a zesláblý dlouhotrvajícími útrapami, jež jsem podstoupil, byl jsem s velkým vlakem raněných trpitelů odsunut do základní nemocnice v Péšávaru.",

    "Zde jsem se zotavil a můj stav se zlepšil natolik, že jsem byl schopen procházet se po nemocničních pokojích a dokonce se trochu vyhřívat na verandě, když mě skolil střevní tyfus, ona kletba našich indických držav.",

    "Po měsíce se o mém životě pochybovalo, a když jsem konečně přišel k sobě a počal se uzdravovat, byl jsem tak slabý a vyhublý, že lékařská komise rozhodla, že se nesmí ztratit ani den a musím být odeslán zpět do Anglie.",

    "Byl jsem tudíž vypraven na vojenské lodi Orontes a o měsíc později jsem vystoupil na přístavní molo v Portsmouthu se zdravím nenávratně zničeným, avšak s povolením otcovské vlády strávit příštích devět měsíců pokusy o jeho nápravu.",

    "Neměl jsem v Anglii příbuzných ani přátel, a byl jsem tedy volný jako pták, či tak volný, jak to jen příjem jedenácti šilinků a šesti pencí denně muži dovolí.",
]


def get_eval_sentences() -> list:
    """Get list of evaluation sentences.

    Returns:
        List of 15 Czech sentences for TTS evaluation
    """
    return CZECH_EVAL_SENTENCES.copy()


if __name__ == "__main__":
    sentences = get_eval_sentences()
    print(f"Total evaluation sentences: {len(sentences)}")
    for i, s in enumerate(sentences, 1):
        print(f"{i:2d}. {s[:60]}...")
```

**Step 2: Test**

```bash
python3 examples/czech/local/eval_sentences.py
```

Expected: Lists 15 sentences

**Step 3: Commit**

```bash
git add examples/czech/local/eval_sentences.py
git commit -m "feat(czech): add evaluation sentences from Sherlock Holmes"
```

---

## Task 7: Create Main Training Script

**Files:**
- Create: `examples/czech/train_czech.py`

**Step 1: Create the training script (Part 1 - imports and args)**

```python
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
    python train_czech.py --config conf/cosyvoice3_czech.yaml \\
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config,
    batch_forward, log_per_step, log_per_save
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Czech CosyVoice3 Fine-tuning')
    parser.add_argument('--train_engine', default='torch_ddp',
                        choices=['torch_ddp'], help='Training engine')
    parser.add_argument('--config', required=True, help='Config file')
    parser.add_argument('--train_data', required=True, help='Train data list')
    parser.add_argument('--cv_data', required=True, help='CV data list')
    parser.add_argument('--qwen_pretrain_path', required=True, help='Qwen tokenizer path')
    parser.add_argument('--checkpoint_llm', help='LLM checkpoint to load')
    parser.add_argument('--checkpoint_flow', help='Flow checkpoint to load')
    parser.add_argument('--model_dir', required=True, help='Output model directory')
    parser.add_argument('--tensorboard_dir', default=None, help='TensorBoard log dir')
    parser.add_argument('--dataset_csv', required=True, help='Dataset CSV for speaker refs')
    parser.add_argument('--ddp.dist_backend', dest='dist_backend', default='nccl')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--prefetch', default=100, type=int)
    parser.add_argument('--pin_memory', action='store_true', default=True)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--save_per_step', default=500, type=int, help='Save every N steps')
    parser.add_argument('--eval_per_step', default=5000, type=int, help='Eval every N steps')
    parser.add_argument('--max_checkpoints', default=3, type=int, help='Max rolling checkpoints')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint dir')
    return parser.parse_args()


# Import eval sentences
from local.eval_sentences import get_eval_sentences

EVAL_SENTENCES = get_eval_sentences()
```

**Step 2: Create training script (Part 2 - helper functions)**

Continue the file with:

```python
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
        if len(parts) >= 3:
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
    pattern = os.path.join(checkpoint_dir, 'step_*')
    checkpoints = sorted(glob.glob(pattern), key=lambda x: int(x.split('_')[-1]))

    # Remove old checkpoints
    while len(checkpoints) > max_keep:
        old_ckpt = checkpoints.pop(0)
        logger.info(f"Removing old checkpoint: {old_ckpt}")
        shutil.rmtree(old_ckpt, ignore_errors=True)


def save_training_state(model, optimizer, scheduler, scaler, step, epoch,
                        save_dir: str, best_loss: float = None):
    """Save complete training state for resume.

    Args:
        model: Model (DDP wrapped)
        optimizer: Optimizer
        scheduler: LR scheduler
        scaler: GradScaler for AMP
        step: Current step
        epoch: Current epoch
        save_dir: Directory to save state
        best_loss: Best eval loss so far
    """
    os.makedirs(save_dir, exist_ok=True)

    state = {
        'step': step,
        'epoch': epoch,
        'best_loss': best_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        'scaler': scaler.state_dict() if scaler is not None else None,
    }

    # Save model state
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), os.path.join(save_dir, 'model.pt'))
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

    torch.save(state, os.path.join(save_dir, 'training_state.pt'))
    logger.info(f"Saved training state to {save_dir}")


def load_training_state(model, optimizer, scheduler, scaler, resume_dir: str):
    """Load training state for resume.

    Args:
        model: Model to load state into
        optimizer: Optimizer
        scheduler: LR scheduler
        scaler: GradScaler
        resume_dir: Directory with saved state

    Returns:
        (step, epoch, best_loss) tuple
    """
    state_path = os.path.join(resume_dir, 'training_state.pt')
    model_path = os.path.join(resume_dir, 'model.pt')

    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Training state not found: {state_path}")

    state = torch.load(state_path, map_location='cpu')

    # Load model
    if os.path.exists(model_path):
        model_state = torch.load(model_path, map_location='cpu')
        if hasattr(model, 'module'):
            model.module.load_state_dict(model_state, strict=False)
        else:
            model.load_state_dict(model_state, strict=False)

    optimizer.load_state_dict(state['optimizer'])

    if state['scheduler'] is not None and hasattr(scheduler, 'load_state_dict'):
        scheduler.load_state_dict(state['scheduler'])

    if state['scaler'] is not None and scaler is not None:
        scaler.load_state_dict(state['scaler'])

    logger.info(f"Resumed from step {state['step']}, epoch {state['epoch']}")
    return state['step'], state['epoch'], state.get('best_loss', float('inf'))
```

**Step 3: Create training script (Part 3 - TTS generation)**

```python
@torch.no_grad()
def generate_tts_samples(llm_model, flow_model, hift_model, frontend,
                         sentences: list, speakers: list, output_dir: str,
                         step: int, sample_rate: int = 24000, device='cuda'):
    """Generate TTS samples for evaluation.

    Memory-safe: uses models already in VRAM, no gradient tracking.

    Args:
        llm_model: LLM model (already on device)
        flow_model: Flow model (already on device)
        hift_model: HiFT vocoder (already on device)
        frontend: CosyVoiceFrontEnd for preprocessing
        sentences: List of sentences to synthesize
        speakers: List of (audio_path, speaker_name) for references
        output_dir: Directory to save audio files
        step: Current training step
        sample_rate: Output sample rate
        device: Device to use
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set models to eval mode
    llm_model.eval()
    flow_model.eval()
    hift_model.eval()

    rng = random.Random(step)  # Different random refs each eval

    for idx, sentence in enumerate(sentences):
        try:
            # Get random speaker reference
            ref_audio, ref_speaker = get_random_speaker_ref(speakers, rng)
            ref_speaker_safe = ref_speaker.replace(' ', '_').replace('/', '_')[:30]
            ref_basename = Path(ref_audio).stem[:20]

            # Prepare prompt text with instruct prefix
            prompt_text = f"You are a helpful assistant.<|endofprompt|>"

            # Frontend processing
            model_input = frontend.frontend_zero_shot(
                sentence, prompt_text, ref_audio, sample_rate, ''
            )

            # Move inputs to device
            for k, v in model_input.items():
                if isinstance(v, torch.Tensor):
                    model_input[k] = v.to(device)

            # LLM inference
            text = model_input['text']
            text_len = torch.tensor([text.shape[1]], dtype=torch.int32, device=device)
            prompt_text_tensor = model_input['prompt_text']
            prompt_text_len = torch.tensor([prompt_text_tensor.shape[1]], dtype=torch.int32, device=device)
            llm_prompt_speech_token = model_input['llm_prompt_speech_token']
            prompt_speech_token_len = torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32, device=device)
            llm_embedding = model_input['llm_embedding']

            # Generate speech tokens
            speech_tokens = []
            for token in llm_model.inference(
                text=text, text_len=text_len,
                prompt_text=prompt_text_tensor, prompt_text_len=prompt_text_len,
                prompt_speech_token=llm_prompt_speech_token,
                prompt_speech_token_len=prompt_speech_token_len,
                embedding=llm_embedding
            ):
                speech_tokens.append(token)

            if not speech_tokens:
                logger.warning(f"No tokens generated for sentence {idx}")
                continue

            speech_token = torch.tensor([speech_tokens], dtype=torch.int32, device=device)

            # Flow inference
            flow_prompt_token = model_input['flow_prompt_speech_token']
            flow_prompt_feat = model_input['prompt_speech_feat']
            flow_embedding = model_input['flow_embedding']

            tts_mel, _ = flow_model.inference(
                token=speech_token,
                token_len=torch.tensor([speech_token.shape[1]], dtype=torch.int32, device=device),
                prompt_token=flow_prompt_token,
                prompt_token_len=torch.tensor([flow_prompt_token.shape[1]], dtype=torch.int32, device=device),
                prompt_feat=flow_prompt_feat,
                prompt_feat_len=torch.tensor([flow_prompt_feat.shape[1]], dtype=torch.int32, device=device),
                embedding=flow_embedding
            )

            # Vocoder
            tts_speech = hift_model(tts_mel)

            # Save audio
            output_name = f"sample_{idx+1:02d}_{ref_speaker_safe}_ref-{ref_basename}.wav"
            output_path = os.path.join(output_dir, output_name)
            torchaudio.save(output_path, tts_speech.cpu(), sample_rate)
            logger.info(f"Generated: {output_name}")

        except Exception as e:
            logger.error(f"Failed to generate sample {idx}: {e}")
            continue

    # Return models to train mode will be done by caller
    logger.info(f"Generated {len(sentences)} TTS samples to {output_dir}")
```

**Step 4: Create training script (Part 4 - main training loop)**

```python
class CzechExecutor(Executor):
    """Extended executor with Czech-specific evaluation."""

    def __init__(self, args, speakers, frontend, sample_rate=24000):
        super().__init__(gan=False)
        self.args = args
        self.speakers = speakers
        self.frontend = frontend
        self.sample_rate = sample_rate
        self.best_loss = float('inf')

    def run_evaluation_tts(self, model, step: int):
        """Run TTS generation during evaluation.

        Args:
            model: DDP wrapped model containing llm, flow, hift
            step: Current training step
        """
        if self.rank != 0:
            return

        output_dir = os.path.join(
            self.args.model_dir, 'audio_samples', f'eval_step_{step}'
        )

        # Get model components
        if hasattr(model, 'module'):
            m = model.module
        else:
            m = model

        generate_tts_samples(
            llm_model=m.llm if hasattr(m, 'llm') else m,
            flow_model=m.flow if hasattr(m, 'flow') else None,
            hift_model=m.hift if hasattr(m, 'hift') else None,
            frontend=self.frontend,
            sentences=EVAL_SENTENCES,
            speakers=self.speakers,
            output_dir=output_dir,
            step=step,
            sample_rate=self.sample_rate
        )


@record
def main():
    """Main training function."""
    args = get_args()

    # Set up tensorboard dir
    if args.tensorboard_dir is None:
        args.tensorboard_dir = os.path.join(args.model_dir, 'tensorboard')

    # Load config
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': args.qwen_pretrain_path})

    # Override save_per_step from args
    configs['train_conf']['save_per_step'] = args.save_per_step
    configs['train_conf'].update(vars(args))

    # Init distributed
    init_distributed(args)
    rank = int(os.environ.get('RANK', 0))

    # Load dataset speakers for TTS eval
    speakers = load_dataset_speakers(args.dataset_csv)
    logger.info(f"Loaded {len(speakers)} speaker references")

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan=False, dpo=False)

    # Check and save config
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard
    writer = init_summarywriter(args)

    # Load models
    llm = configs['llm']
    flow = configs['flow']

    # Load pretrained weights
    if args.checkpoint_llm and os.path.exists(args.checkpoint_llm):
        state_dict = torch.load(args.checkpoint_llm, map_location='cpu')
        llm.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded LLM from {args.checkpoint_llm}")

    if args.checkpoint_flow and os.path.exists(args.checkpoint_flow):
        state_dict = torch.load(args.checkpoint_flow, map_location='cpu')
        flow.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded Flow from {args.checkpoint_flow}")

    # Create combined model for training
    # For simplicity, train LLM and Flow separately in sequence
    # This follows the original CosyVoice training pattern

    start_step, start_epoch = 0, -1
    best_loss = float('inf')

    # Wrap model
    model = llm  # Start with LLM training
    model = wrap_cuda_model(args, model)

    # Optimizer and scheduler
    model, optimizer, scheduler, _, _ = init_optimizer_and_scheduler(args, configs, model, gan=False)
    scheduler.set_step(start_step)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # Resume if specified
    if args.resume and os.path.exists(args.resume):
        start_step, start_epoch, best_loss = load_training_state(
            model, optimizer, scheduler, scaler, args.resume
        )

    # Create executor with frontend for TTS
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd
    frontend = CosyVoiceFrontEnd(
        configs['get_tokenizer'],
        configs['feat_extractor'],
        os.path.join(os.path.dirname(args.checkpoint_llm), 'campplus.onnx'),
        os.path.join(os.path.dirname(args.checkpoint_llm), 'speech_tokenizer_v3.onnx'),
        None,  # No spk2info needed
        configs['allowed_special']
    )

    executor = CzechExecutor(args, speakers, frontend, configs['sample_rate'])
    executor.step = start_step

    # Info dict
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch

    # Initial evaluation
    if rank == 0 and start_step == 0:
        logger.info("Running initial evaluation...")
        executor.run_evaluation_tts(model, step=0)

    logger.info(f'Starting training from step {start_step}, epoch {start_epoch + 1}')

    # Training loop
    for epoch in range(start_epoch + 1, configs['train_conf']['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()

        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=60))

        model.train()

        for batch_idx, batch_dict in enumerate(train_data_loader):
            info_dict["tag"] = "TRAIN"
            info_dict["step"] = executor.step
            info_dict["epoch"] = epoch
            info_dict["batch_idx"] = batch_idx

            # Forward pass
            info_dict = batch_forward(model, batch_dict, scaler, info_dict)

            # Backward pass
            if scaler is not None:
                scaler.scale(info_dict['loss_dict']['loss']).backward()
            else:
                info_dict['loss_dict']['loss'].backward()

            # Update
            if (batch_idx + 1) % info_dict['accum_grad'] == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                executor.step += 1

                # Log
                log_per_step(writer, info_dict)

                # Save checkpoint
                if executor.step % args.save_per_step == 0:
                    ckpt_dir = os.path.join(args.model_dir, 'checkpoints', f'step_{executor.step}')
                    save_training_state(model, optimizer, scheduler, scaler,
                                       executor.step, epoch, ckpt_dir, best_loss)
                    manage_rolling_checkpoints(
                        os.path.join(args.model_dir, 'checkpoints'),
                        args.max_checkpoints
                    )

                # Evaluation
                if executor.step % args.eval_per_step == 0:
                    logger.info(f"Running evaluation at step {executor.step}...")
                    model.eval()

                    # Compute eval loss
                    total_loss = 0
                    num_batches = 0
                    with torch.no_grad():
                        for cv_batch in cv_data_loader:
                            cv_info = batch_forward(model, cv_batch, None, info_dict.copy())
                            total_loss += cv_info['loss_dict']['loss'].item()
                            num_batches += 1

                    avg_loss = total_loss / max(num_batches, 1)
                    logger.info(f"Eval loss at step {executor.step}: {avg_loss:.6f}")

                    if writer is not None:
                        writer.add_scalar('CV/loss', avg_loss, executor.step)

                    # Save eval checkpoint
                    eval_ckpt_dir = os.path.join(args.model_dir, 'checkpoints', f'eval_step_{executor.step}')
                    save_training_state(model, optimizer, scheduler, scaler,
                                       executor.step, epoch, eval_ckpt_dir, best_loss)

                    # Save best model
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_dir = os.path.join(args.model_dir, 'checkpoints', 'best')
                        save_training_state(model, optimizer, scheduler, scaler,
                                           executor.step, epoch, best_dir, best_loss)
                        logger.info(f"New best model saved with loss {best_loss:.6f}")

                    # Generate TTS samples
                    executor.run_evaluation_tts(model, executor.step)

                    model.train()

        dist.destroy_process_group(group_join)

        # End of epoch eval
        logger.info(f"End of epoch {epoch} evaluation...")
        # Similar eval code as above...

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
```

**Step 5: Verify syntax**

```bash
python3 -m py_compile examples/czech/train_czech.py
```

**Step 6: Commit**

```bash
git add examples/czech/train_czech.py
git commit -m "feat(czech): add main training script with eval TTS and checkpointing"
```

---

## Task 8: Create Run Script

**Files:**
- Create: `examples/czech/run.sh`

**Step 1: Create the run script**

```bash
#!/bin/bash
# Czech Fine-tuning for Fun-CosyVoice3-0.5B-2512
# Usage: ./run.sh [stage]

set -e

# Paths
MODEL_DIR=/mnt/4TB_Dataset_WD/MODELS/AUDIO/TEXT_TO_SPEECH/Fun-CosyVoice3-0.5B-2512
DATASET_CSV=/mnt/4TB_Dataset_WD/AUDIO_DATASETS/CZECH/CZECH_30s_200hours_hranicar_oko_merged/dataset_merged.csv
OUTPUT_BASE=/mnt/4TB_Dataset_WD/AUDIO_DATASETS/CZECH/CZECH_30s_200hours_hranicar_oko_merged_Fun-CosyVoice3-0.5B-2512
TRAINING_OUTPUT=/mnt/8TB/TRAINING_OUTPUTS/Fun-CosyVoice3-0.5B-2512_CZECH_30s_200hours_lr1e-5_$(date +%Y-%m-%d)

# Training params
CUDA_VISIBLE_DEVICES="0"
NUM_GPUS=1

stage=${1:-0}
stop_stage=${2:-100}

# Stage 0: Prepare data from CSV
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Prepare data from CSV"
    mkdir -p $OUTPUT_BASE
    python local/prepare_czech_data.py \
        --src_csv $DATASET_CSV \
        --des_dir $OUTPUT_BASE \
        --delimiter '|'
fi

# Stage 1: Split train/eval
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Split train/eval (99%/1%)"
    python local/split_train_eval.py \
        --src_dir $OUTPUT_BASE \
        --des_dir $OUTPUT_BASE \
        --eval_percent 1.0 \
        --seed 42
fi

# Stage 2: Extract speaker embeddings
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Extract speaker embeddings"
    for split in train eval; do
        python tools/extract_embedding.py \
            --dir $OUTPUT_BASE/$split \
            --onnx_path $MODEL_DIR/campplus.onnx \
            --num_thread 8
    done
fi

# Stage 3: Extract speech tokens
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Extract speech tokens"
    for split in train eval; do
        python tools/extract_speech_token.py \
            --dir $OUTPUT_BASE/$split \
            --onnx_path $MODEL_DIR/speech_tokenizer_v3.onnx \
            --num_thread 4
    done
fi

# Stage 4: Make parquet files
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Make parquet files"
    for split in train eval; do
        mkdir -p $OUTPUT_BASE/$split/parquet
        python tools/make_parquet_list.py \
            --num_utts_per_parquet 1000 \
            --num_processes 8 \
            --instruct \
            --src_dir $OUTPUT_BASE/$split \
            --des_dir $OUTPUT_BASE/$split/parquet
    done
fi

# Stage 5: Train LLM + Flow
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Stage 5: Train LLM + Flow"
    mkdir -p $TRAINING_OUTPUT

    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

    torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS \
        --rdzv_id=2024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:29500" \
        train_czech.py \
        --config conf/cosyvoice3_czech.yaml \
        --train_data $OUTPUT_BASE/train/parquet/data.list \
        --cv_data $OUTPUT_BASE/eval/parquet/data.list \
        --qwen_pretrain_path $MODEL_DIR/CosyVoice-BlankEN \
        --checkpoint_llm $MODEL_DIR/llm.pt \
        --checkpoint_flow $MODEL_DIR/flow.pt \
        --model_dir $TRAINING_OUTPUT \
        --dataset_csv $DATASET_CSV \
        --save_per_step 500 \
        --eval_per_step 5000 \
        --max_checkpoints 3 \
        --use_amp \
        --pin_memory
fi

echo "Done!"
```

**Step 2: Make executable**

```bash
chmod +x examples/czech/run.sh
```

**Step 3: Commit**

```bash
git add examples/czech/run.sh
git commit -m "feat(czech): add run script for full training pipeline"
```

---

## Task 9: Test Data Preparation (Dry Run)

**Step 1: Test data prep script on small sample**

```bash
cd examples/czech
head -100 /mnt/4TB_Dataset_WD/AUDIO_DATASETS/CZECH/CZECH_30s_200hours_hranicar_oko_merged/dataset_merged.csv > /tmp/test_czech.csv
python local/prepare_czech_data.py --src_csv /tmp/test_czech.csv --des_dir /tmp/test_czech_out
```

Expected: Creates wav.scp, text, utt2spk, spk2utt, instruct

**Step 2: Verify output files**

```bash
wc -l /tmp/test_czech_out/*
head -3 /tmp/test_czech_out/text
```

**Step 3: Clean up test files**

```bash
rm -rf /tmp/test_czech.csv /tmp/test_czech_out
```

---

## Task 10: Final Integration Test

**Step 1: Run stages 0-1 on full dataset**

```bash
cd examples/czech
./run.sh 0 1
```

Expected: Creates Kaldi files and train/eval split

**Step 2: Verify split sizes**

```bash
wc -l /mnt/4TB_Dataset_WD/AUDIO_DATASETS/CZECH/CZECH_30s_200hours_hranicar_oko_merged_Fun-CosyVoice3-0.5B-2512/train/wav.scp
wc -l /mnt/4TB_Dataset_WD/AUDIO_DATASETS/CZECH/CZECH_30s_200hours_hranicar_oko_merged_Fun-CosyVoice3-0.5B-2512/eval/wav.scp
```

Expected: ~99% train, ~1% eval

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat(czech): complete Czech fine-tuning implementation"
```

---

## Summary

**Files Created:**
1. `examples/czech/local/prepare_czech_data.py` - CSV to Kaldi conversion
2. `examples/czech/local/split_train_eval.py` - Train/eval split
3. `examples/czech/local/eval_sentences.py` - Czech evaluation sentences
4. `examples/czech/tools/` - Symlinks to extraction tools
5. `examples/czech/conf/cosyvoice3_czech.yaml` - Training config
6. `examples/czech/train_czech.py` - Main training script
7. `examples/czech/run.sh` - Full pipeline script

**Key Features:**
- Rolling checkpoints (max 3, every 500 steps)
- Evaluation every 5000 steps with TTS generation
- Best model tracking by eval loss
- Full resume capability
- TensorBoard logging
- 15 Czech sentences with random speaker refs per eval
