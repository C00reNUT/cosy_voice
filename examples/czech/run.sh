#!/bin/bash
# Czech Fine-tuning for Fun-CosyVoice3-0.5B-2512
# Usage: ./run.sh [stage] [stop_stage]
# Micromamba env: cosyvoice

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

echo "=============================================="
echo "Czech Fine-tuning Pipeline"
echo "Model: Fun-CosyVoice3-0.5B-2512"
echo "Dataset: CZECH_30s_200hours"
echo "Output: $OUTPUT_BASE"
echo "Training Output: $TRAINING_OUTPUT"
echo "=============================================="

# Stage 0: Prepare data from CSV
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo ""
    echo "========== Stage 0: Prepare data from CSV =========="
    mkdir -p $OUTPUT_BASE
    python local/prepare_czech_data.py \
        --src_csv $DATASET_CSV \
        --des_dir $OUTPUT_BASE \
        --delimiter '|'
    echo "Stage 0 complete. Output: $OUTPUT_BASE"
fi

# Stage 1: Split train/eval
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo ""
    echo "========== Stage 1: Split train/eval (99%/1%) =========="
    python local/split_train_eval.py \
        --src_dir $OUTPUT_BASE \
        --des_dir $OUTPUT_BASE \
        --eval_percent 1.0 \
        --seed 42
    echo "Stage 1 complete. Train: $OUTPUT_BASE/train, Eval: $OUTPUT_BASE/eval"
fi

# Stage 2: Extract speaker embeddings
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo ""
    echo "========== Stage 2: Extract speaker embeddings =========="
    for split in train eval; do
        echo "Processing $split split..."
        python tools/extract_embedding.py \
            --dir $OUTPUT_BASE/$split \
            --onnx_path $MODEL_DIR/campplus.onnx \
            --num_thread 8
    done
    echo "Stage 2 complete."
fi

# Stage 3: Extract speech tokens
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo ""
    echo "========== Stage 3: Extract speech tokens =========="
    for split in train eval; do
        echo "Processing $split split..."
        python tools/extract_speech_token.py \
            --dir $OUTPUT_BASE/$split \
            --onnx_path $MODEL_DIR/speech_tokenizer_v3.onnx \
            --num_thread 4
    done
    echo "Stage 3 complete."
fi

# Stage 4: Make parquet files
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo ""
    echo "========== Stage 4: Make parquet files =========="
    for split in train eval; do
        echo "Processing $split split..."
        mkdir -p $OUTPUT_BASE/$split/parquet
        python tools/make_parquet_list.py \
            --num_utts_per_parquet 1000 \
            --num_processes 8 \
            --instruct \
            --src_dir $OUTPUT_BASE/$split \
            --des_dir $OUTPUT_BASE/$split/parquet
    done
    echo "Stage 4 complete."
fi

# Stage 5: Train LLM
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo ""
    echo "========== Stage 5: Train LLM =========="
    mkdir -p $TRAINING_OUTPUT

    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

    torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS \
        --rdzv_id=2024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:29500" \
        train_czech.py \
        --model llm \
        --config conf/cosyvoice3_czech.yaml \
        --train_data $OUTPUT_BASE/train/parquet/data.list \
        --cv_data $OUTPUT_BASE/eval/parquet/data.list \
        --qwen_pretrain_path $MODEL_DIR/CosyVoice-BlankEN \
        --checkpoint $MODEL_DIR/llm.pt \
        --model_dir $TRAINING_OUTPUT/llm \
        --dataset_csv $DATASET_CSV \
        --save_per_step 500 \
        --eval_per_step 5000 \
        --max_checkpoints 3 \
        --use_amp \
        --pin_memory
    echo "Stage 5 complete. LLM checkpoints: $TRAINING_OUTPUT/llm"
fi

# Stage 6: Train Flow
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo ""
    echo "========== Stage 6: Train Flow =========="
    mkdir -p $TRAINING_OUTPUT

    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

    torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS \
        --rdzv_id=2025 --rdzv_backend="c10d" --rdzv_endpoint="localhost:29501" \
        train_czech.py \
        --model flow \
        --config conf/cosyvoice3_czech.yaml \
        --train_data $OUTPUT_BASE/train/parquet/data.list \
        --cv_data $OUTPUT_BASE/eval/parquet/data.list \
        --qwen_pretrain_path $MODEL_DIR/CosyVoice-BlankEN \
        --checkpoint $MODEL_DIR/flow.pt \
        --model_dir $TRAINING_OUTPUT/flow \
        --dataset_csv $DATASET_CSV \
        --save_per_step 500 \
        --eval_per_step 5000 \
        --max_checkpoints 3 \
        --use_amp \
        --pin_memory
    echo "Stage 6 complete. Flow checkpoints: $TRAINING_OUTPUT/flow"
fi

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "LLM checkpoints: $TRAINING_OUTPUT/llm"
echo "Flow checkpoints: $TRAINING_OUTPUT/flow"
echo "=============================================="
