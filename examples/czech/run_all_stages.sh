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

echo "=========================================="
echo "Czech CosyVoice Fine-tuning Pipeline"
echo "Started: $(date)"
echo "=========================================="
echo "OUTPUT_BASE: $OUTPUT_BASE"
echo "MODEL_DIR: $MODEL_DIR"
echo "TRAINING_OUTPUT: $TRAINING_OUTPUT"
echo "Auto-resume: ENABLED (use --no_resume to disable)"
echo ""

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate $CONDA_ENV

# ========== Stage 2: Extract speaker embeddings ==========
echo ""
echo "========== Stage 2: Extract speaker embeddings =========="
echo "Started: $(date)"

for split in train eval; do
    echo ""
    echo "--- Processing $split split ---"
    python $TOOLS_DIR/extract_embedding_progress.py \
        --dir $OUTPUT_BASE/$split \
        --onnx_path $MODEL_DIR/campplus.onnx \
        --num_thread $NUM_THREADS
done

echo ""
echo "Stage 2 complete: $(date)"

# ========== Stage 3: Extract speech tokens ==========
echo ""
echo "========== Stage 3: Extract speech tokens =========="
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

# ========== Stage 4: Make parquet files ==========
echo ""
echo "========== Stage 4: Make parquet files =========="
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

# ========== Stage 5: LLM Training ==========
echo ""
echo "========== Stage 5: LLM Training =========="
echo "Started: $(date)"

# Create training output directory
mkdir -p $TRAINING_OUTPUT/llm
mkdir -p $TRAINING_OUTPUT/flow
mkdir -p $TRAINING_OUTPUT/eval_samples

cd $CZECH_DIR

# Run LLM training
echo "Starting LLM training..."
python train_czech.py \
    --train_data $OUTPUT_BASE/train_parquet/data.list \
    --cv_data $OUTPUT_BASE/eval_parquet/data.list \
    --model_dir $MODEL_DIR \
    --config conf/cosyvoice3_czech.yaml \
    --checkpoint $MODEL_DIR/llm.pt \
    --train_type llm \
    --output_dir $TRAINING_OUTPUT/llm \
    --eval_output_dir $TRAINING_OUTPUT/eval_samples \
    --eval_data_dir $OUTPUT_BASE/eval \
    --num_workers 4 \
    --prefetch 100

echo ""
echo "Stage 5 (LLM training) complete: $(date)"

# ========== Stage 6: Flow Training ==========
echo ""
echo "========== Stage 6: Flow Training =========="
echo "Started: $(date)"

# Get latest LLM checkpoint
LATEST_LLM_CKPT=$(ls -t $TRAINING_OUTPUT/llm/*.pt 2>/dev/null | head -1)
if [ -z "$LATEST_LLM_CKPT" ]; then
    echo "Warning: No LLM checkpoint found, using original model"
    LATEST_LLM_CKPT=$MODEL_DIR/llm.pt
fi
echo "Using LLM checkpoint: $LATEST_LLM_CKPT"

# Run Flow training
echo "Starting Flow training..."
python train_czech.py \
    --train_data $OUTPUT_BASE/train_parquet/data.list \
    --cv_data $OUTPUT_BASE/eval_parquet/data.list \
    --model_dir $MODEL_DIR \
    --config conf/cosyvoice3_czech.yaml \
    --checkpoint $MODEL_DIR/flow.pt \
    --train_type flow \
    --output_dir $TRAINING_OUTPUT/flow \
    --eval_output_dir $TRAINING_OUTPUT/eval_samples \
    --eval_data_dir $OUTPUT_BASE/eval \
    --num_workers 4 \
    --prefetch 100

echo ""
echo "Stage 6 (Flow training) complete: $(date)"

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "Finished: $(date)"
echo "Training outputs: $TRAINING_OUTPUT"
echo "=========================================="
