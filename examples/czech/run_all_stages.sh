#!/bin/bash
# Czech CosyVoice Fine-tuning Pipeline - Stages 2-6
# Features: Progress bars, auto-resume, checkpoint saving
#
# Run with: nohup bash run_all_stages.sh > /tmp/czech_cosyvoice_pipeline.log 2>&1 &
# Monitor with: tail -f /tmp/czech_cosyvoice_pipeline.log

set -e

# Configuration
OUTPUT_BASE=/mnt/4TB_Dataset_WD/AUDIO_DATASETS/CZECH/CZECH_30s_200hours_hranicar_oko_merged_Fun-CosyVoice3-0.5B-2512
MODEL_DIR=/mnt/4TB_Dataset_WD/MODELS/AUDIO/TEXT_TO_SPEECH/Fun-CosyVoice3-0.5B-2512
COSYVOICE_DIR=/mnt/8TB/AUDIO/TEXT_TO_SPEECH/CosyVoice
CZECH_DIR=$COSYVOICE_DIR/examples/czech
TOOLS_DIR=$CZECH_DIR/tools
TRAINING_OUTPUT=/mnt/8TB/TRAINING_OUTPUTS/Fun-CosyVoice3-0.5B-2512_CZECH_30s_200hours_lr1e-5_$(date +%Y-%m-%d)
CONDA_ENV=fish-speech
NUM_THREADS=8
# Set to "gpu" for faster Stage 2 (~10 min instead of ~115 min)
# GPU version uses CUDAExecutionProvider instead of CPU threading
STAGE2_MODE="cpu"  # Options: "cpu" (current), "gpu" (5-20x faster)

echo "=========================================="
echo "Czech CosyVoice Fine-tuning Pipeline"
echo "Started: $(date)"
echo "=========================================="
echo "OUTPUT_BASE: $OUTPUT_BASE"
echo "MODEL_DIR: $MODEL_DIR"
echo "TRAINING_OUTPUT: $TRAINING_OUTPUT"
echo "Auto-resume: ENABLED (use --no_resume to disable)"
echo ""
echo "[STAGE: INIT] Pipeline initialization complete"
echo ""

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate $CONDA_ENV

# Add CosyVoice and Matcha-TTS to PYTHONPATH
export PYTHONPATH="$COSYVOICE_DIR:$COSYVOICE_DIR/third_party/Matcha-TTS:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# ========== Pre-validation (Optional but recommended) ==========
echo ""
echo "========== Pre-validation: Checking dataset =========="
echo "[PRE-CHECK] Validating audio files exist and are readable..."
echo "Started: $(date)"

VALIDATION_FAILED=0
for split in train eval; do
    echo ""
    echo "--- Validating $split split ---"

    # Check required files exist
    if [ ! -f "$OUTPUT_BASE/$split/wav.scp" ]; then
        echo "ERROR: wav.scp not found for $split split"
        VALIDATION_FAILED=1
        continue
    fi

    # Run validation (quick mode - just check existence)
    # Use --strict to fail on missing files, remove for warning-only mode
    python $TOOLS_DIR/validate_dataset.py \
        --dir $OUTPUT_BASE/$split \
        --quick \
        --num_threads $NUM_THREADS || {
            echo "WARNING: Some files failed validation in $split split"
            echo "Check $OUTPUT_BASE/$split/invalid_files.txt for details"
        }
done

if [ $VALIDATION_FAILED -eq 1 ]; then
    echo "FATAL: Pre-validation failed. Fix issues before running pipeline."
    exit 1
fi

echo ""
echo "Pre-validation complete: $(date)"

# Add NVIDIA library paths to LD_LIBRARY_PATH
NVIDIA_LIB_BASE="/mnt/BigStorage/MAMBA_CACHE_DIR/envs/fish-speech/lib/python3.11/site-packages/nvidia"
for lib_dir in $(find $NVIDIA_LIB_BASE -name "lib" -type d 2>/dev/null); do
    export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
done
echo "LD_LIBRARY_PATH set with NVIDIA libraries"

# ========== Stage 2: Extract speaker embeddings ==========
echo ""
echo "========== Stage 2: Extract speaker embeddings =========="
if [ "$STAGE2_MODE" = "gpu" ]; then
    echo "[STAGE: 2/6] Extracting speaker embeddings (GPU, ~10-15 min)"
else
    echo "[STAGE: 2/6] Extracting speaker embeddings (CPU, ~115 min)"
fi
echo "Started: $(date)"

for split in train eval; do
    echo ""
    echo "--- Processing $split split ---"
    if [ "$STAGE2_MODE" = "gpu" ]; then
        python $TOOLS_DIR/extract_embedding_gpu.py \
            --dir $OUTPUT_BASE/$split \
            --onnx_path $MODEL_DIR/campplus.onnx
    else
        python $TOOLS_DIR/extract_embedding_progress.py \
            --dir $OUTPUT_BASE/$split \
            --onnx_path $MODEL_DIR/campplus.onnx \
            --num_thread $NUM_THREADS
    fi
done

echo ""
echo "Stage 2 complete: $(date)"

# Post-Stage 2 validation
echo ""
echo "[CHECK] Validating Stage 2 outputs..."
STAGE2_VALID=1
for split in train eval; do
    if [ ! -f "$OUTPUT_BASE/$split/utt2embedding.pt" ]; then
        echo "ERROR: utt2embedding.pt not found for $split"
        STAGE2_VALID=0
    fi
    if [ ! -f "$OUTPUT_BASE/$split/spk2embedding.pt" ]; then
        echo "ERROR: spk2embedding.pt not found for $split"
        STAGE2_VALID=0
    fi
    # Check for failed files log
    if [ -f "$OUTPUT_BASE/$split/failed_embedding.log" ]; then
        FAILED_COUNT=$(grep -v "^#" "$OUTPUT_BASE/$split/failed_embedding.log" | wc -l)
        if [ "$FAILED_COUNT" -gt 0 ]; then
            echo "WARNING: $FAILED_COUNT files failed embedding extraction in $split"
        fi
    fi
done
if [ $STAGE2_VALID -eq 0 ]; then
    echo "FATAL: Stage 2 validation failed. Cannot continue."
    exit 1
fi
echo "[CHECK] Stage 2 outputs validated successfully"

# ========== Stage 3: Extract speech tokens ==========
echo ""
echo "========== Stage 3: Extract speech tokens =========="
echo "[STAGE: 3/6] Extracting speech tokens (GPU, ~15 min)"
echo "Started: $(date)"

for split in train eval; do
    echo ""
    echo "--- Processing $split split ---"
    python $TOOLS_DIR/extract_speech_token_progress.py \
        --dir $OUTPUT_BASE/$split \
        --onnx_path $MODEL_DIR/speech_tokenizer_v3.onnx \
        --num_thread $NUM_THREADS
done

echo ""
echo "Stage 3 complete: $(date)"

# Post-Stage 3 validation
echo ""
echo "[CHECK] Validating Stage 3 outputs..."
STAGE3_VALID=1
for split in train eval; do
    if [ ! -f "$OUTPUT_BASE/$split/utt2speech_token.pt" ]; then
        echo "ERROR: utt2speech_token.pt not found for $split"
        STAGE3_VALID=0
    fi
    # Check for failed files log
    if [ -f "$OUTPUT_BASE/$split/failed_speech_token.log" ]; then
        FAILED_COUNT=$(grep -v "^#" "$OUTPUT_BASE/$split/failed_speech_token.log" | wc -l)
        if [ "$FAILED_COUNT" -gt 0 ]; then
            echo "WARNING: $FAILED_COUNT files failed speech token extraction in $split"
        fi
    fi
done
if [ $STAGE3_VALID -eq 0 ]; then
    echo "FATAL: Stage 3 validation failed. Cannot continue."
    exit 1
fi
echo "[CHECK] Stage 3 outputs validated successfully"

# ========== Stage 4: Make parquet files ==========
echo ""
echo "========== Stage 4: Make parquet files =========="
echo "[STAGE: 4/6] Creating parquet files (~5 min)"
echo "Started: $(date)"

cd $COSYVOICE_DIR/examples/libritts/cosyvoice3

for split in train eval; do
    echo ""
    echo "--- Processing $split split ---"

    # Check if parquet already exists
    if [ -f "$OUTPUT_BASE/${split}_parquet/data.list" ]; then
        echo "Parquet files already exist for $split, skipping..."
        continue
    fi

    python tools/make_parquet_list.py \
        --num_utts_per_parquet 1000 \
        --num_processes $NUM_THREADS \
        --src_dir $OUTPUT_BASE/$split \
        --des_dir $OUTPUT_BASE/${split}_parquet
done

echo ""
echo "Stage 4 complete: $(date)"

# Post-Stage 4 validation and data.list sanitization
echo ""
echo "[CHECK] Validating and sanitizing Stage 4 outputs..."
STAGE4_VALID=1
for split in train eval; do
    PARQUET_DIR="$OUTPUT_BASE/${split}_parquet"
    DATA_LIST="$PARQUET_DIR/data.list"

    # Check if parquet files exist
    PARQUET_COUNT=$(find "$PARQUET_DIR" -name "parquet_*.tar" -type f 2>/dev/null | wc -l)
    if [ "$PARQUET_COUNT" -eq 0 ]; then
        echo "ERROR: No parquet files found in $PARQUET_DIR"
        STAGE4_VALID=0
        continue
    fi
    echo "Found $PARQUET_COUNT parquet files for $split"

    # Regenerate data.list to ensure no ANSI color codes
    echo "Regenerating data.list for $split (sanitized)..."
    ls --color=never "$PARQUET_DIR"/parquet_*.tar > "$DATA_LIST"

    # Verify data.list is clean (no escape sequences)
    if grep -qP '\x1b\[' "$DATA_LIST" 2>/dev/null; then
        echo "WARNING: ANSI codes detected in $DATA_LIST, cleaning..."
        sed -i 's/\x1b\[[0-9;]*m//g' "$DATA_LIST"
    fi

    # Count entries
    ENTRIES=$(wc -l < "$DATA_LIST")
    echo "data.list for $split contains $ENTRIES entries"
done

if [ $STAGE4_VALID -eq 0 ]; then
    echo "FATAL: Stage 4 validation failed. Cannot continue."
    exit 1
fi
echo "[CHECK] Stage 4 outputs validated and sanitized successfully"

# ========== Stage 5: LLM Training ==========
echo ""
echo "========== Stage 5: LLM Training =========="
echo "[STAGE: 5/6] LLM fine-tuning (~12-36 hours)"
echo "Started: $(date)"

# Create training output directory
mkdir -p $TRAINING_OUTPUT/llm
mkdir -p $TRAINING_OUTPUT/flow
mkdir -p $TRAINING_OUTPUT/eval_samples

cd $CZECH_DIR

# Check if LLM training already completed (max_epoch=2 means epoch_1_whole.pt is final)
if [ -f "$TRAINING_OUTPUT/llm/epoch_1_whole.pt" ]; then
    echo "LLM training already completed (epoch_1_whole.pt exists), skipping..."
else
    # Run LLM training with torchrun for distributed training
    echo "Starting LLM training..."
    torchrun --nproc_per_node=1 --master_port=29501 train_czech.py \
        --train_data $OUTPUT_BASE/train_parquet/data.list \
        --cv_data $OUTPUT_BASE/eval_parquet/data.list \
        --model_dir $TRAINING_OUTPUT/llm \
        --config conf/cosyvoice3_czech.yaml \
        --checkpoint $MODEL_DIR/llm.pt \
        --model llm \
        --qwen_pretrain_path $MODEL_DIR/CosyVoice-BlankEN \
        --dataset_csv /mnt/4TB_Dataset_WD/AUDIO_DATASETS/CZECH/CZECH_30s_200hours_hranicar_oko_merged/dataset_merged.csv \
        --num_workers 4 \
        --prefetch 100
fi

echo ""
echo "Stage 5 (LLM training) complete: $(date)"

# ========== Stage 6: Flow Training ==========
echo ""
echo "========== Stage 6: Flow Training =========="
echo "[STAGE: 6/6] Flow fine-tuning (~12-36 hours)"
echo "Started: $(date)"

# Get latest LLM checkpoint
LATEST_LLM_CKPT=$(ls -t $TRAINING_OUTPUT/llm/*.pt 2>/dev/null | head -1)
if [ -z "$LATEST_LLM_CKPT" ]; then
    echo "Warning: No LLM checkpoint found, using original model"
    LATEST_LLM_CKPT=$MODEL_DIR/llm.pt
fi
echo "Using LLM checkpoint: $LATEST_LLM_CKPT"

# Run Flow training with torchrun for distributed training
echo "Starting Flow training..."
torchrun --nproc_per_node=1 --master_port=29502 train_czech.py \
    --train_data $OUTPUT_BASE/train_parquet/data.list \
    --cv_data $OUTPUT_BASE/eval_parquet/data.list \
    --model_dir $TRAINING_OUTPUT/flow \
    --config conf/cosyvoice3_czech.yaml \
    --checkpoint $MODEL_DIR/flow.pt \
    --model flow \
    --qwen_pretrain_path $MODEL_DIR/CosyVoice-BlankEN \
    --dataset_csv /mnt/4TB_Dataset_WD/AUDIO_DATASETS/CZECH/CZECH_30s_200hours_hranicar_oko_merged/dataset_merged.csv \
    --num_workers 4 \
    --prefetch 100

echo ""
echo "Stage 6 (Flow training) complete: $(date)"

echo ""
echo "=========================================="
echo "[STAGE: DONE] Pipeline Complete!"
echo "Finished: $(date)"
echo "Training outputs: $TRAINING_OUTPUT"
echo "=========================================="
