#!/usr/bin/env bash
set -euo pipefail

stage=${stage:--1}
stop_stage=${stop_stage:-4}

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GRPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COSYVOICE3_MODEL_DIR="${COSYVOICE3_MODEL_DIR:-$ROOT_DIR/pretrained_models/Fun-CosyVoice3-0.5B}"
HF_LLM_PATH="${HF_LLM_PATH:-}"
CONVERT_LLM_CKPT="${CONVERT_LLM_CKPT:-}"
CONVERT_OUTPUT_DIR="${CONVERT_OUTPUT_DIR:-}"
CONVERT_CPU="${CONVERT_CPU:-0}"

DATA_DIR="${DATA_DIR:-$GRPO_DIR/data}"
PARQUET_DIR="${PARQUET_DIR:-$DATA_DIR/parquet}"
TRAIN_JSONL="${TRAIN_JSONL:-$DATA_DIR/train.jsonl}"
TEST_JSONL="${TEST_JSONL:-$DATA_DIR/test.jsonl}"
TEXT_TRAIN="${TEXT_TRAIN:-}"
TEXT_TEST="${TEXT_TEST:-}"

EXP_NAME="${EXP_NAME:-cosyvoice3_grpo}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$GRPO_DIR/checkpoints/cosyvoice3_grpo}"
MERGE_STEP="${MERGE_STEP:-100}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"

ASR_SERVER_LOG="${ASR_SERVER_LOG:-$GRPO_DIR/token2wav_asr_server.log}"
ASR_SERVER_PID="${ASR_SERVER_PID:-$GRPO_DIR/token2wav_asr_server.pid}"

PREPARE_PY="$ROOT_DIR/examples/grpo/cosyvoice2/prepare_data.py"
REWARD_PY="$ROOT_DIR/examples/grpo/cosyvoice2/reward_tts.py"
ASR_SERVER_PY="$GRPO_DIR/token2wav_asr_server.py"
PRETRAINED_CONVERT_PY="$GRPO_DIR/pretrained_to_huggingface.py"
HF_TO_PRETRAINED_PY="$GRPO_DIR/huggingface_to_pretrained.py"
TEXT_TO_JSONL_PY="$GRPO_DIR/text_to_jsonl.py"

ASR_BACKEND="${ASR_BACKEND:-whisper}"
ASR_MODEL="${ASR_MODEL:-small}"
ASR_LANGUAGE="${ASR_LANGUAGE:-cs}"
ASR_COMPUTE_TYPE="${ASR_COMPUTE_TYPE:-float16}"
ASR_BEAM_SIZE="${ASR_BEAM_SIZE:-1}"

PROMPT_WAV="${PROMPT_WAV:-}"
PROMPT_TEXT="${PROMPT_TEXT:-}"
PROMPT_MANIFEST="${PROMPT_MANIFEST:-}"

if [[ $stage -le -1 && $stop_stage -ge -1 ]]; then
  log "stage -1: validate prerequisites"
  if [[ -z "$HF_LLM_PATH" && -n "$CONVERT_LLM_CKPT" && -n "$CONVERT_OUTPUT_DIR" ]]; then
    log "converting CosyVoice3 LLM checkpoint to HuggingFace format"
    CONVERT_FLAGS=()
    if [[ "$CONVERT_CPU" != "0" ]]; then
      CONVERT_FLAGS+=(--cpu)
    fi
    python "$PRETRAINED_CONVERT_PY" \
      --pretrained-cosyvoice3-path "$COSYVOICE3_MODEL_DIR" \
      --llm-ckpt "$CONVERT_LLM_CKPT" \
      --save-path "$CONVERT_OUTPUT_DIR" \
      "${CONVERT_FLAGS[@]}"
    HF_LLM_PATH="$CONVERT_OUTPUT_DIR"
  fi
  if [[ -z "$HF_LLM_PATH" ]]; then
    echo "ERROR: HF_LLM_PATH is required for GRPO training."
    echo "Provide a HuggingFace-compatible LLM path for CosyVoice3."
    exit 1
  fi
  if [[ ! -d "$HF_LLM_PATH" ]]; then
    echo "ERROR: HF_LLM_PATH does not exist: $HF_LLM_PATH"
    exit 1
  fi
  if [[ ! -d "$COSYVOICE3_MODEL_DIR" ]]; then
    echo "ERROR: COSYVOICE3_MODEL_DIR does not exist: $COSYVOICE3_MODEL_DIR"
    exit 1
  fi
  if [[ ! -f "$PREPARE_PY" || ! -f "$REWARD_PY" || ! -f "$ASR_SERVER_PY" || ! -f "$TEXT_TO_JSONL_PY" ]]; then
    echo "ERROR: Required scripts not found."
    echo "prepare: $PREPARE_PY"
    echo "reward:  $REWARD_PY"
    echo "asr:     $ASR_SERVER_PY"
    echo "text:    $TEXT_TO_JSONL_PY"
    exit 1
  fi
  ASR_BACKEND_ENV="$ASR_BACKEND" python - <<'PY'
import importlib.util
import os
backend = os.environ.get("ASR_BACKEND_ENV", "faster-whisper")
for name in ["verl", "pytriton", "datasets", "scipy"]:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"Missing Python module: {name}")
if backend in ("auto", "faster-whisper"):
    if importlib.util.find_spec("faster_whisper") is None:
        if backend == "faster-whisper":
            raise SystemExit("Missing Python module: faster_whisper")
        if importlib.util.find_spec("whisper") is None:
            raise SystemExit("Missing Python module: faster_whisper or whisper")
elif backend == "whisper":
    if importlib.util.find_spec("whisper") is None:
        raise SystemExit("Missing Python module: whisper")
elif backend in ("nemo-parakeet", "nemo-canary"):
    if importlib.util.find_spec("nemo") is None:
        raise SystemExit("Missing Python module: nemo")
print("Prerequisites OK")
PY
fi

if [[ $stage -le 0 && $stop_stage -ge 0 ]]; then
  log "stage 0: prepare data into veRL parquet format"
  mkdir -p "$DATA_DIR" "$PARQUET_DIR"
  if [[ ! -f "$TRAIN_JSONL" || ! -f "$TEST_JSONL" ]]; then
    if [[ -n "$TEXT_TRAIN" && -n "$TEXT_TEST" ]]; then
      log "building JSONL from Kaldi text files"
      python "$TEXT_TO_JSONL_PY" --input-text "$TEXT_TRAIN" --output-jsonl "$TRAIN_JSONL"
      python "$TEXT_TO_JSONL_PY" --input-text "$TEXT_TEST" --output-jsonl "$TEST_JSONL"
    else
      log "missing train/test JSONL, downloading AISHELL3 example set"
      wget -O "$DATA_DIR/aishell-3.jsonl" \
        https://huggingface.co/datasets/SparkAudio/voxbox/resolve/main/metadata/aishell-3.jsonl
      head -n 80000 "$DATA_DIR/aishell-3.jsonl" > "$TRAIN_JSONL"
      tail -n 100 "$DATA_DIR/aishell-3.jsonl" > "$TEST_JSONL"
    fi
  fi
  python "$PREPARE_PY" \
    --train_file "$TRAIN_JSONL" \
    --test_file "$TEST_JSONL" \
    --local_dir "$PARQUET_DIR"
fi

if [[ $stage -le 1 && $stop_stage -ge 1 ]]; then
  log "stage 1: start token2wav ASR server for reward"
  if [[ -z "$PROMPT_WAV" && -z "$PROMPT_MANIFEST" ]]; then
    echo "ERROR: PROMPT_WAV or PROMPT_MANIFEST must be set for the ASR reward server."
    exit 1
  fi
  if [[ -f "$ASR_SERVER_PID" ]] && kill -0 "$(cat "$ASR_SERVER_PID")" 2>/dev/null; then
    log "ASR server already running (pid $(cat "$ASR_SERVER_PID"))"
  else
    nohup python "$ASR_SERVER_PY" \
      --cosyvoice3-path "$COSYVOICE3_MODEL_DIR" \
      --number-of-devices "$N_GPUS_PER_NODE" \
      --asr-backend "$ASR_BACKEND" \
      --asr-model "$ASR_MODEL" \
      --asr-language "$ASR_LANGUAGE" \
      --asr-compute-type "$ASR_COMPUTE_TYPE" \
      --asr-beam-size "$ASR_BEAM_SIZE" \
      ${PROMPT_WAV:+--prompt-wav "$PROMPT_WAV"} \
      ${PROMPT_TEXT:+--prompt-text "$PROMPT_TEXT"} \
      ${PROMPT_MANIFEST:+--prompt-manifest "$PROMPT_MANIFEST"} \
      > "$ASR_SERVER_LOG" 2>&1 &
    echo $! > "$ASR_SERVER_PID"
    log "ASR server started (pid $(cat "$ASR_SERVER_PID"))"
  fi
fi

if [[ $stage -le 2 && $stop_stage -ge 2 ]]; then
  log "stage 2: GRPO training (LLM only)"
  export MKL_SERVICE_FORCE_INTEL=TRUE
  python -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files="$PARQUET_DIR/train.parquet" \
      data.val_files="$PARQUET_DIR/test.parquet" \
      data.train_batch_size="$TRAIN_BATCH_SIZE" \
      data.max_prompt_length=1024 \
      data.max_response_length=256 \
      data.truncation='error' \
      actor_rollout_ref.model.use_remove_padding=False \
      actor_rollout_ref.model.path="$HF_LLM_PATH" \
      actor_rollout_ref.actor.optim.lr=5e-7 \
      actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
      actor_rollout_ref.actor.use_kl_loss=False \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
      actor_rollout_ref.rollout.do_sample=true \
      actor_rollout_ref.rollout.temperature=0.7 \
      actor_rollout_ref.rollout.top_p=0.95 \
      actor_rollout_ref.rollout.top_k=25 \
      actor_rollout_ref.rollout.n=4 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=true \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
      actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
      actor_rollout_ref.rollout.val_kwargs.top_k=25 \
      reward_model.reward_manager=prime \
      custom_reward_function.path="$REWARD_PY" \
      custom_reward_function.name=compute_score \
      trainer.project_name='cosyvoice3_grpo' \
      trainer.experiment_name="$EXP_NAME" \
      trainer.logger=['console','tensorboard'] \
      trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
      trainer.nnodes=1 \
      trainer.save_freq=100 \
      trainer.test_freq=100 \
      trainer.resume_mode='auto' \
      trainer.total_epochs=1 \
      trainer.val_before_train=False
fi

if [[ $stage -le 3 && $stop_stage -ge 3 ]]; then
  log "stage 3: merge model shards (FSDP -> HF)"
  STEP_DIR="$CHECKPOINT_ROOT/$EXP_NAME/global_step_${MERGE_STEP}"
  if [[ ! -d "$STEP_DIR/actor" ]]; then
    echo "ERROR: missing checkpoint: $STEP_DIR/actor"
    exit 1
  fi
  python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$STEP_DIR/actor" \
      --target_dir "$STEP_DIR/merged_hf_model"
  log "merged HF model at $STEP_DIR/merged_hf_model"
fi

if [[ $stage -le 4 && $stop_stage -ge 4 ]]; then
  log "stage 4: export to CosyVoice3 format"
  echo "Use $HF_TO_PRETRAINED_PY to convert merged HF checkpoints to llm.pt."
fi
