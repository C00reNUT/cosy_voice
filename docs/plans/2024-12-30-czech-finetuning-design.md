# Czech Language Fine-tuning for Fun-CosyVoice3-0.5B-2512

**Date:** 2024-12-30
**Status:** Approved
**Model:** Fun-CosyVoice3-0.5B-2512
**Dataset:** CZECH_30s_200hours_hranicar_oko_merged (~200 hours)

---

## Overview

Fine-tune Fun-CosyVoice3-0.5B-2512 to support Czech language using LLM+Flow SFT training approach.

## Key Finding: Tokenizer Already Supports Czech

The Qwen-based tokenizer natively handles all Czech characters through GPT-2 byte-level BPE:

| Czech Char | Token ID | Verified |
|------------|----------|----------|
| š | 11645 | ✓ |
| Š | 144456 | ✓ |
| ř | 28332 | ✓ |
| Ř | 146092 | ✓ |
| ž | 12176 | ✓ |
| Ž | 144768 | ✓ |
| ů | 51253 | ✓ |
| Ů | 148941 | ✓ |
| ť | 74801 | ✓ |
| Ť | 146148 | ✓ |
| ň | 145082 | ✓ |

**No tokenizer modification needed.**

---

## Paths

### Input

- **Model:** `/mnt/4TB_Dataset_WD/MODELS/AUDIO/TEXT_TO_SPEECH/Fun-CosyVoice3-0.5B-2512/`
- **Dataset CSV:** `/mnt/4TB_Dataset_WD/AUDIO_DATASETS/CZECH/CZECH_30s_200hours_hranicar_oko_merged/dataset_merged.csv`
- **Dataset Format:** `audio_file|text|speaker|duration|segments_merged`

### Output

- **Prepared Data:** `/mnt/4TB_Dataset_WD/AUDIO_DATASETS/CZECH/CZECH_30s_200hours_hranicar_oko_merged_Fun-CosyVoice3-0.5B-2512/`
- **Training Output:** `/mnt/8TB/TRAINING_OUTPUTS/Fun-CosyVoice3-0.5B-2512_CZECH_30s_200hours_lr1e-5_2024-12-30/`

---

## Data Preparation Pipeline

### Step 1: Convert CSV to Kaldi-style files

```
{dataset}_Fun-CosyVoice3-0.5B-2512/
├── wav.scp          # utt_id → audio_path
├── text             # utt_id → transcription (with instruct prefix)
├── utt2spk          # utt_id → speaker_id
└── spk2utt          # speaker_id → utt_ids
```

### Step 2: Extract speaker embeddings

Using `campplus.onnx`:
- `utt2embedding.pt` - per-utterance embeddings
- `spk2embedding.pt` - per-speaker averaged embeddings

### Step 3: Extract speech tokens

Using `speech_tokenizer_v3.onnx`:
- `utt2speech_token.pt` - discrete speech tokens

### Step 4: Convert to parquet

```
parquet/
├── train/
│   └── data.list
└── eval/
    └── data.list    # 1% of data
```

### Instruct Prefix

All text prefixed with: `"You are a helpful assistant.<|endofprompt|>"`

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Mode | LLM + Flow SFT (joint) |
| Epochs | 2 |
| Optimizer | Adam |
| Learning Rate | 1e-5 |
| Scheduler | constantlr |
| Warmup Steps | 2500 |
| Gradient Clip | 5 |
| AMP | Enabled |

### Checkpointing

| Type | Frequency | Max Kept |
|------|-----------|----------|
| Regular | Every 500 steps | 3 (rolling) |
| Eval | Every 5000 steps + start | All |
| Best | On best eval loss | 1 |

### Resume Support

`training_state.pt` contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Current step
- Current epoch
- GradScaler state (if AMP)

---

## Evaluation

### Schedule

- Step 0 (before training)
- Every 5000 steps
- End of training

### Metrics

- Eval loss (logged to TensorBoard)
- Generated audio samples

### TTS Generation

**15 Czech reference sentences** (Sherlock Holmes excerpt):

1. SHERLOCK HOLMES: STUDIE V ŠARLATOVÉ - Kapitola první: PAN SHERLOCK HOLMES.
2. Roku osmnáct set sedmdesát osm jsem dosáhl hodnosti doktora medicíny...
3. Když jsem tam studia dokončil, byl jsem řádně přidělen k Pátému pluku...
4. Pluk v té době pobýval v Indii, a než jsem se k němu mohl připojit...
5. Po vylodění v Bombají jsem se dozvěděl, že můj sbor již prošel průsmyky...
6. Následoval jsem jej však s mnoha dalšími důstojníky...
7. Tažení přineslo mnohým pocty a povýšení, leč mně nic než neštěstí...
8. Byl jsem odvelen od své brigády a přidělen k berkshirskému pluku...
9. Tam mě zasáhla střela z džezailu do ramene...
10. Byl bych padl do rukou vražedných Gázíů...
11. Ztrápen bolestí a zesláblý dlouhotrvajícími útrapami...
12. Zde jsem se zotavil a můj stav se zlepšil natolik...
13. Po měsíce se o mém životě pochybovalo...
14. Byl jsem tudíž vypraven na vojenské lodi „Orontes"...
15. Neměl jsem v Anglii příbuzných ani přátel...

**Speaker Reference:**
- Random speaker from dataset for each sample
- Different random selection for each evaluation run
- Reference audio path included in output filename

**Memory-Safe Inference:**
```python
@torch.no_grad()
def generate_eval_samples(model, ...):
    model.eval()
    # Uses model already in VRAM
    # No gradient tracking
    for sentence in eval_sentences:
        # Generate with random speaker reference
        ...
    model.train()
```

---

## Output Structure

```
/mnt/8TB/TRAINING_OUTPUTS/Fun-CosyVoice3-0.5B-2512_CZECH_30s_200hours_lr1e-5_2024-12-30/
├── checkpoints/
│   ├── step_500/
│   │   ├── llm.pt
│   │   ├── flow.pt
│   │   └── training_state.pt
│   ├── step_1000/
│   ├── step_1500/
│   ├── eval_step_0/
│   ├── eval_step_5000/
│   └── best/
│
├── audio_samples/
│   ├── eval_step_0/
│   │   ├── sample_01_SHERLOCK_ref-John_Flanagan_000123.wav
│   │   └── ... (15 samples)
│   └── eval_step_5000/
│
├── tensorboard/
│   └── events.out.tfevents.*
│
└── config.yaml
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `examples/czech/local/prepare_czech_data.py` | CSV → Kaldi format conversion |
| `examples/czech/tools/extract_embedding.py` | Speaker embedding extraction |
| `examples/czech/tools/extract_speech_token.py` | Speech token extraction |
| `examples/czech/tools/make_parquet_list.py` | Parquet conversion |
| `examples/czech/train_czech.py` | Main training script |
| `examples/czech/conf/cosyvoice3_czech.yaml` | Training config |
| `examples/czech/eval_sentences.py` | Evaluation sentences and TTS |

---

## Implementation Order

1. Data preparation scripts
2. Training script with checkpointing
3. Evaluation integration with TTS
4. TensorBoard logging
5. Resume functionality
6. End-to-end testing

---

## Notes

- Use micromamba environment for all Python execution
- All scripts include module docstrings with env name
- Follow existing CosyVoice code patterns
- No new markdown files beyond this plan
