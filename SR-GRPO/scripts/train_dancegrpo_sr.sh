#!/bin/bash
# Derived from https://github.com/XueZeyue/DanceGRPO/blob/main/scripts/finetune/finetune_flux_grpo_8gpus_lora.sh.
# Modified by Institute of Automation, Chinese Academy of Sciences in 2026.

# ============================================================================
# GRPO Fine-tuning Script for Flux ControlNet Super-Resolution Model
# Uses DeQA-Score, RefReward (Qwen), and LPIPS as reward models
# ============================================================================

# Wandb configuration
export WANDB_DISABLED=false
export WANDB_API_KEY="your_wandb_api_key"  # Replace with your API Key from https://wandb.ai/settings
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
# For offline mode (no API Key needed):
# export WANDB_MODE=offline

export PYTHONPATH=$(pwd):$PYTHONPATH
# Optional: Hugging Face mirror (default is the official endpoint)
# export HF_ENDPOINT=https://hf-mirror.com

# ============================================================================
# Path Configuration — Please modify these paths to match your environment
# ============================================================================

# FLUX.1-dev model (download from https://huggingface.co/black-forest-labs/FLUX.1-dev)
FLUX_MODEL_PATH="/path/to/FLUX.1-dev"

# ControlNet checkpoint for SR
CONTROLNET_PATH="/path/to/controlnet/model.safetensors"

# Dataset paths
DATA_JSON_PATH="/path/to/sr_embeddings/videos2caption.json"
LR_IMAGE_ROOT="/path/to/train_datasets/sr_bicubic"
HR_IMAGE_ROOT="/path/to/train_datasets/gt"

# Reward model paths
DEQA_MODEL_PATH="/path/to/DeQA-Score/checkpoints/DeQA-Score-Mix3"
DEQA_PREPROCESSOR_PATH="/path/to/DeQA-Score/preprocessor"
QWEN_MODEL_PATH="/path/to/RefReward/checkpoint"  # Trained RefReward model

# Output directory
OUTPUT_DIR="./outputs/flux_controlnet_sr_grpo"
mkdir -p "${OUTPUT_DIR}/images"
mkdir -p "${OUTPUT_DIR}/images_lq"

# ============================================================================
# Training
# ============================================================================

torchrun --nproc_per_node=8 --master_port 19003 \
    fastvideo/train_grpo_flux_controlnet_sr_lora.py \
    --seed 42 \
    --pretrained_model_name_or_path ${FLUX_MODEL_PATH} \
    --controlnet_model_name_or_path ${CONTROLNET_PATH} \
    --vae_model_path ${FLUX_MODEL_PATH} \
    --cache_dir data/.cache \
    --data_json_path ${DATA_JSON_PATH} \
    --lr_image_root ${LR_IMAGE_ROOT} \
    --hr_image_root ${HR_IMAGE_ROOT} \
    --gradient_checkpointing \
    --train_batch_size 2 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 12 \
    --max_train_steps 601 \
    --learning_rate 3e-4 \
    --mixed_precision bf16 \
    --checkpointing_steps 10 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir ${OUTPUT_DIR} \
    --h 512 \
    --w 512 \
    --t 1 \
    --sampling_steps 25 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 12 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --lora_alpha 64 \
    --lora_rank 32 \
    --num_double_layers 4 \
    --num_single_layers 0 \
    --controlnet_conditioning_scale 1.0 \
    --guidance_scale 2.5 \
    --train_transformer \
    --use_deqa \
    --deqa_model_path ${DEQA_MODEL_PATH} \
    --deqa_preprocessor_path ${DEQA_PREPROCESSOR_PATH} \
    --use_qwen \
    --qwen_model_path ${QWEN_MODEL_PATH} \
    --use_lpips \
    --deqa_weight 1.0 \
    --qwen_weight 1.0 \
    --lpips_weight 1.0
