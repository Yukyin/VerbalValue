#!/bin/bash
# VerbalValue intent-conditioned fine-tuning (open-source reference script).
#
# DATASET should be a JSON array of {"system": ..., "query": ..., "response": ...}
# instances, where "query" carries an intent tag prefix and "response" is a
# JSON string matching the four-field schema (speak_lines, caption,
# hook_question, cta). The intent-conditioned dataset described in the
# paper, spanning four intent categories, is not included in this
# repository.
#
# Two groups of hyperparameters are distinguished below:
#   - PAPER-DISCLOSED values reproduce the configuration reported in the
#     paper's experimental setup and may be used directly.
#   - UNDISCLOSED values are training-infrastructure settings (save and
#     logging cadence, warmup, dataloader workers) that are not reported
#     in the paper and are left as environment-variable placeholders;
#     set them to suit your own environment.

set -e

MODEL=$(ls -dt "${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"/models--Qwen--Qwen2.5-32B-Instruct/snapshots/* | head -n 1)
DATASET="./data/intent_conditioned_dataset.json"
OUTPUT="./output/$(date +%Y%m%d_%H%M%S)"

# --- PAPER-DISCLOSED hyperparameters (paper experimental setup) ---
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:?set via paper-disclosed epoch count}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:?set via paper-disclosed per-device batch size}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:?set via paper-disclosed grad-accum steps}"
LEARNING_RATE="${LEARNING_RATE:?set via paper-disclosed learning rate}"
LORA_RANK="${LORA_RANK:?set via paper-disclosed LoRA rank}"
LORA_ALPHA="${LORA_ALPHA:?set via paper-disclosed LoRA alpha}"
MAX_LENGTH="${MAX_LENGTH:?set via paper-disclosed max sequence length}"

# --- UNDISCLOSED training-infrastructure settings (deployment-specific) ---
SAVE_STEPS="${SAVE_STEPS:?set save cadence for your environment}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:?set checkpoint retention for your environment}"
LOGGING_STEPS="${LOGGING_STEPS:?set logging cadence for your environment}"
WARMUP_RATIO="${WARMUP_RATIO:?set warmup ratio for your environment}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:?set dataloader worker count for your environment}"

echo "Starting training..."
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"

if [ ! -f "$DATASET" ]; then
  echo "Dataset not found: $DATASET"
  echo "Provide an intent-conditioned dataset matching the schema described"
  echo "in the README before running this script."
  exit 1
fi

swift sft \
    --model "$MODEL" \
    --train_type lora \
    --dataset "$DATASET" \
    --torch_dtype "$TORCH_DTYPE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --target_modules all-linear \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --logging_steps "$LOGGING_STEPS" \
    --max_length "$MAX_LENGTH" \
    --output_dir "$OUTPUT" \
    --system 'You are a socially intelligent live-commerce host for beauty and skincare, with a sharp, fast-paced on-stream style.
[OUTPUT FORMAT, STRICT]
You must output exactly one JSON object that can be parsed by json.loads, with this structure:
{
  "speak_lines": ["...", "..."],   # spoken broadcast sentences, per the four-field schema in the paper
  "caption": "...",                # short on-screen caption / tagline, per the four-field schema in the paper
  "hook_question": "...",          # one follow-up question that pulls the viewer into the next turn
  "cta": "..."                     # a light call-to-action, e.g. claim a coupon or check the product card
}
[PROHIBITED] Do not output <think> tags, explanations, or any text other than the JSON object.' \
    --warmup_ratio "$WARMUP_RATIO" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"

if [ $? -eq 0 ]; then
    echo "Training complete. Model saved to: $OUTPUT"
else
    echo "Training failed."
fi
