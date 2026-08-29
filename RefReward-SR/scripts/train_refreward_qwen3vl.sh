#!/bin/bash
# Copyright 2026 Institute of Automation, Chinese Academy of Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================================
# RefReward Training Script — GRPO for Qwen3-VL on SR Quality Scoring
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

export DEBUG_MODE="true"
export PYTHONPATH=$PYTHONPATH:${PROJECT_DIR}/src

# ============================================================================
# Path Configuration — Please modify these paths to match your environment
# ============================================================================

RUN_NAME="refreward-sr"
export LOG_PATH="./log_${RUN_NAME}.txt"

# Base model (download from https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
MODEL_PATH="/path/to/Qwen3-VL-8B-Instruct"

# Training data
IMAGE_FOLDERS="/path/to/RefReward-SR-dataset/train_images"
DATA_FILE="${PROJECT_DIR}/dataset/train_sr.jsonl"

# ============================================================================
# Training
# ============================================================================

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_jsonl_sr_qwen3vl.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path ${MODEL_PATH} \
    --question_template scoring \
    --dataset_name KADID-10K \
    --image_folders ${IMAGE_FOLDERS} \
    --data_file_paths ${DATA_FILE} \
    --freeze_vision_modules false \
    --max_prompt_length 1024 \
    --num_generations 6 \
    --per_device_image_group_size 1 \
    --per_device_image_pair_size 4 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_steps 1200 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true
