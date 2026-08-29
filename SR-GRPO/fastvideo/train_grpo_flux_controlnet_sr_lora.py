# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0]
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.
#
# Modified by Institute of Automation, Chinese Academy of Sciences in 2026.
# Adapted from train_grpo_flux_lora.py for Flux ControlNet SR GRPO fine-tuning.

import argparse
import math
import os
from pathlib import Path
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions

from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.dataset_utils import LengthGroupedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
# MODIFIED: Use SR-specific dataset that provides LR images as control input
from fastvideo.dataset.latent_flux_sr_rl_datasets import LatentSRDataset, latent_sr_collate_function
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_checkpoint,
    save_lora_checkpoint,
    resume_lora_optimizer,
)
from fastvideo.utils.logging_ import main_print
import cv2
from diffusers.image_processor import VaeImageProcessor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
import time
from collections import deque
import numpy as np
from einops import rearrange
import torch.distributed as dist
from torch.nn import functional as F
from typing import List
from PIL import Image
# MODIFIED: Import FluxControlNetModel for super-resolution
from diffusers import FluxTransformer2DModel, AutoencoderKL
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from contextlib import contextmanager
from safetensors.torch import save_file, load_file
import lpips
import re
import sys
import traceback
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)
    

def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    # dw~N(0, delta_t)
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
        
    if grpo:
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = ((
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
        )
        - math.log(std_dev_t)- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean,pred_original_sample



def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents


# MODIFIED: Added prepare_control_image function to process LR images for ControlNet
def prepare_control_image(control_image, vae, device, dtype, batch_size, height, width, train_vae=False):
    """
    Process control image (LR image) for ControlNet input.
    1. Encode with VAE
    2. Normalize
    3. Pack into sequence format
    
    Args:
        train_vae: If True, keep gradients for VAE training
    """
    # Ensure control_image is on correct device and dtype
    # Use float32 only when training VAE (VAE loaded in float32); otherwise match VAE dtype (bfloat16)
    vae_dtype = next(vae.parameters()).dtype
    control_image = control_image.to(device=device, dtype=vae_dtype)
    
    if train_vae:
        # Keep gradients for VAE training
        control_latents = vae.encode(control_image).latent_dist.sample()
    else:
        with torch.no_grad():
            control_latents = vae.encode(control_image).latent_dist.sample()
    
    # Normalize: (latents - shift_factor) * scaling_factor
    # For Flux VAE: shift_factor=0.1159, scaling_factor=0.3611
    control_latents = (control_latents - 0.1159) * 0.3611
    
    # Pack latents
    _, c, h, w = control_latents.shape
    control_latents = pack_latents(control_latents, batch_size, c, h, w)
        
    return control_latents.to(dtype=dtype)


# MODIFIED: Added ControlNet parameters (controlnet, control_image, controlnet_conditioning_scale)
def run_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        controlnet,  # ADDED: ControlNet model
        control_image,  # ADDED: Packed control image latents
        encoder_hidden_states, 
        pooled_prompt_embeds, 
        text_ids,
        image_ids, 
        grpo_sample,
    ):
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []
        for i in progress_bar:
            B = encoder_hidden_states.shape[0]
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
            transformer.eval()
            controlnet.eval()  # ADDED: Set ControlNet to eval mode
            
            with torch.autocast("cuda", torch.bfloat16):
                # ADDED: Get ControlNet outputs
                controlnet_block_samples, controlnet_single_block_samples = controlnet(
                    hidden_states=z,
                    controlnet_cond=control_image,
                    controlnet_mode=None,
                    conditioning_scale=args.controlnet_conditioning_scale,
                    timestep=timesteps/1000,
                    guidance=torch.tensor(
                        [args.guidance_scale],
                        device=z.device,
                        dtype=torch.bfloat16
                    ),
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1),
                    img_ids=image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )
                
                # MODIFIED: Pass ControlNet outputs to transformer
                pred = transformer(
                    hidden_states=z,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps/1000,
                    guidance=torch.tensor(
                        [args.guidance_scale],
                        device=z.device,
                        dtype=torch.bfloat16
                    ),
                    txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1), # B, L
                    pooled_projections=pooled_prompt_embeds,
                    img_ids=image_ids,
                    controlnet_block_samples=controlnet_block_samples,  # ADDED
                    controlnet_single_block_samples=controlnet_single_block_samples,  # ADDED
                    joint_attention_kwargs=None,
                    return_dict=False,
                    controlnet_blocks_repeat=False,  # ADDED: For instantx controlnet style
                )[0]
            z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True)
            z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
        latents = pred_original
        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
        return z, latents, all_latents, all_log_probs

        
# MODIFIED: Added ControlNet parameters
def grpo_one_step(
            args,
            latents,
            pre_latents,
            encoder_hidden_states, 
            pooled_prompt_embeds, 
            text_ids,
            image_ids,
            transformer,
            controlnet,  # ADDED
            control_image,  # ADDED
            timesteps,
            i,
            sigma_schedule,
):
    B = encoder_hidden_states.shape[0]
    transformer.train()
    # Note: ControlNet is typically frozen during GRPO training, only transformer LoRA is trained
    controlnet.eval()
    
    with torch.autocast("cuda", torch.bfloat16):
        # ADDED: Get ControlNet outputs (no gradient)
        with torch.no_grad():
            controlnet_block_samples, controlnet_single_block_samples = controlnet(
                hidden_states=latents,
                controlnet_cond=control_image,
                controlnet_mode=None,
                conditioning_scale=args.controlnet_conditioning_scale,
                timestep=timesteps/1000,
                guidance=torch.tensor(
                    [args.guidance_scale],
                    device=latents.device,
                    dtype=torch.bfloat16
                ),
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1),
                img_ids=image_ids.squeeze(0),
                joint_attention_kwargs=None,
                return_dict=False,
            )
        
        # MODIFIED: Pass ControlNet outputs to transformer
        pred = transformer(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps/1000,
            guidance=torch.tensor(
                [args.guidance_scale],
                device=latents.device,
                dtype=torch.bfloat16
            ),
            txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1), # B, L
            pooled_projections=pooled_prompt_embeds,
            img_ids=image_ids.squeeze(0),
            controlnet_block_samples=controlnet_block_samples,  # ADDED
            controlnet_single_block_samples=controlnet_single_block_samples,  # ADDED
            joint_attention_kwargs=None,
            return_dict=False,
            controlnet_blocks_repeat=False,  # ADDED
        )[0]
    z, pred_original, log_prob = flux_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True)
    return log_prob


def _normalize_device(device):
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)


def move_model_to_device(model, device, dtype=None):
    if model is None or not hasattr(model, "to"):
        return
    device = _normalize_device(device)
    if dtype is None:
        model.to(device)
    else:
        model.to(device=device, dtype=dtype)
    # Keep auxiliary tensors on the same device as the model.
    aux_weight = getattr(model, "weight_tensor", None)
    if aux_weight is not None:
        aux_weight = aux_weight.to(device=device)
        setattr(model, "weight_tensor", aux_weight)
    aux_input_ids = getattr(model, "input_ids", None)
    if aux_input_ids is not None:
        aux_input_ids = aux_input_ids.to(device=device)
        setattr(model, "input_ids", aux_input_ids)
    if device.type == "cpu":
        torch.cuda.empty_cache()


# MODIFIED: Added ControlNet and control_image parameters, and multi-reward support
def sample_reference_model(
    args,
    device, 
    transformer,
    controlnet,  # ADDED
    vae,
    control_image_raw,  # ADDED: Raw LR image tensor
    hr_image_raw,  # ADDED: Raw HR image tensor for LPIPS
    encoder_hidden_states, 
    pooled_prompt_embeds, 
    text_ids,
    caption,
    train_vae=False,  # ADDED: Whether to train VAE with LoRA
    deqa_reward_model=None,
    qwen_reward_model=None,
    qwen_processor=None,
    net_lpips=None,
):
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_deqa_rewards = []
    all_qwen_rewards = []
    all_lpips_rewards = []
    all_image_ids = []
    all_control_images = []  # ADDED: Store packed control images
    
    if args.init_same_noise:
        input_latents = torch.randn(
                (1, IN_CHANNELS, latent_h, latent_w),
                device=device,
                dtype=torch.bfloat16,
            )

    all_image_paths = []
    all_lq_image_paths = []
    all_hr_image_raws = []
    
    rank = int(os.environ["RANK"])
    image_root_dir = args.output_dir if args.output_dir is not None else "."
    images_dir = os.path.join(image_root_dir, "images")
    images_lq_dir = os.path.join(image_root_dir, "images_lq")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(images_lq_dir, exist_ok=True)
    image_processor = VaeImageProcessor(16)
    vae.enable_tiling()
    
    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_pooled_prompt_embeds = pooled_prompt_embeds[batch_idx]
        batch_text_ids = text_ids[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]
        batch_control_image_raw = control_image_raw[batch_idx]
        batch_hr_image_raw = hr_image_raw[batch_idx]
        
        if not args.init_same_noise:
            input_latents = torch.randn(
                    (len(batch_idx), IN_CHANNELS, latent_h, latent_w),
                    device=device,
                    dtype=torch.bfloat16,
                )
        
        # Prepare control image (encode with VAE and pack)
        control_image_packed = prepare_control_image(
            batch_control_image_raw, vae, device, torch.bfloat16, 
            len(batch_idx), latent_h, latent_w, train_vae=False
        )
        
        input_latents_new = pack_latents(input_latents, len(batch_idx), IN_CHANNELS, latent_h, latent_w)
        image_ids = prepare_latent_image_ids(len(batch_idx), latent_h // 2, latent_w // 2, device, torch.bfloat16)
        grpo_sample = True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        
        with torch.no_grad():
            z, latents, batch_latents, batch_log_probs = run_sample_step(
                args,
                input_latents_new,
                progress_bar,
                sigma_schedule,
                transformer,
                controlnet,
                control_image_packed,
                batch_encoder_hidden_states,
                batch_pooled_prompt_embeds,
                batch_text_ids,
                image_ids,
                grpo_sample,
            )
        
        all_image_ids.append(image_ids)
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        all_control_images.append(control_image_packed)
        
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents = unpack_latents(latents, h, w, 8)
                latents = (latents / 0.3611) + 0.1159
                image = vae.decode(latents, return_dict=False)[0]
                decoded_image = image_processor.postprocess(image)
        
        image_path = os.path.join(images_dir, f"flux_sr_{rank}_{index}.png")
        decoded_image[0].save(image_path)
        all_image_paths.append(image_path)
        
        lq_image_pil = Image.fromarray(
            ((batch_control_image_raw[0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).astype(np.uint8)
        )
        lq_image_path = os.path.join(images_lq_dir, f"flux_sr_lq_{rank}_{index}.png")
        lq_image_pil.save(lq_image_path)
        all_lq_image_paths.append(lq_image_path)
        
        all_hr_image_raws.append(batch_hr_image_raw[0:1].cpu())
        
        main_print(f"[Sampling {index}] Done sampling and saving images")
    
    main_print(f"[Stage 1 Done] Generated {len(all_image_paths)} images")
    torch.cuda.empty_cache()
    
    if args.use_deqa and deqa_reward_model is not None:
        main_print("[DEQA] Computing rewards...")
        move_model_to_device(deqa_reward_model, device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for idx, image_path in enumerate(all_image_paths):
                    img_list = [Image.open(image_path)]
                    deqa_score = deqa_reward_model(img_list)
                    all_deqa_rewards.append(deqa_score.to(device))
        move_model_to_device(deqa_reward_model, torch.device("cpu"))
        torch.cuda.empty_cache()
        main_print(f"[DEQA] Computed {len(all_deqa_rewards)} rewards")
    else:
        all_deqa_rewards = [torch.tensor([0.0], device=device) for _ in all_image_paths]
    
    if args.use_qwen and qwen_reward_model is not None and qwen_processor is not None:
        main_print("[QWEN] Computing rewards...")
        move_model_to_device(qwen_reward_model, device, dtype=torch.bfloat16)
        from qwen_vl_utils import process_vision_info
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for idx, (lq_path, hr_path) in enumerate(zip(all_lq_image_paths, all_image_paths)):
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": lq_path},
                                {"type": "image", "image": hr_path},
                                {"type": "text", "text": (
                                    "You are a highly discerning expert in image quality assessment."
                                    "Your task is to evaluate a super-resolution (SR) output by comparing it with the original low-resolution reference."
                                    "You will be given two images: the first (LR) shows the original scene, and the second (HR) is the super-resolution result."
                                    "Assume the HR image may contain problematic regions."
                                    "Carefully inspect both images and identify any areas in the HR image that look unnatural or distorted (e.g., warped regions, unrealistic textures, artifacts) or semantically inconsistent with the LR image (e.g., missing, added or altered objects, inconsistent texture)."
                                    "For each problematic region, state its approximate location in the HR image (e.g., 'top left', 'center lower') and briefly explain why it appears incorrect."
                                    "After inspection, assign an overall quality score from 1.00 to 5.00, with two decimal places (e.g., 1.31, 2.77, 4.53)."
                                    "First output your reasoning in <thinking>…</thinking> tags (a brief summary of observed defects and semantic issues)."
                                    "Then output only one numeric score in <answer>…</answer> tags (no extra text)."
                                )},
                            ],
                        }
                    ]
                    
                    text = qwen_processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    text += "<thinking>…</thinking><answer>"
                    
                    image_inputs, video_inputs, video_kwargs = process_vision_info(
                        messages,
                        return_video_kwargs=True,
                        image_patch_size=16,
                    )
                    
                    inputs = qwen_processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        **video_kwargs,
                        do_resize=False,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(device)
                    
                    generated_ids = qwen_reward_model.generate(**inputs, max_new_tokens=64, do_sample=False)
                    
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = qwen_processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    
                    text_for_parse = "<answer>" + output_text[0]
                    m = re.search(r"<answer>\s*([0-9]+(?:\.[0-9]+)?)\s*(?:</answer>)?", text_for_parse, re.IGNORECASE)
                    
                    if m:
                        qwen_score = float(m.group(1))
                    else:
                        qwen_score = torch.empty(1).uniform_(1.0, 5.0).item()
                        qwen_score = round(qwen_score, 2)
                        print(f"[Warning] QWEN parse failed, output: {output_text[0]}, random score: {qwen_score}")
                    
                    qwen_score = torch.tensor([qwen_score], dtype=torch.float32, device=device)
                    all_qwen_rewards.append(qwen_score)
        
        move_model_to_device(qwen_reward_model, torch.device("cpu"))
        torch.cuda.empty_cache()
        main_print(f"[QWEN] Computed {len(all_qwen_rewards)} rewards")
    else:
        all_qwen_rewards = [torch.tensor([0.0], device=device) for _ in all_image_paths]
    
    if args.use_lpips and net_lpips is not None:
        main_print("[LPIPS] Computing rewards...")
        with torch.no_grad():
            for idx, (image_path, hr_raw) in enumerate(zip(all_image_paths, all_hr_image_raws)):
                gen_image_pil = Image.open(image_path)
                gen_image_tensor = torch.from_numpy(
                    np.array(gen_image_pil).transpose(2, 0, 1)
                ).float() / 255.0
                gen_image_tensor = gen_image_tensor * 2.0 - 1.0
                gen_image_tensor = gen_image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
                
                hr_image_tensor = hr_raw.to(device=device, dtype=torch.float32)
                
                lpips_val = net_lpips(gen_image_tensor, hr_image_tensor).squeeze()
                lpips_reward = -lpips_val
                all_lpips_rewards.append(lpips_reward.unsqueeze(0).to(device))
        main_print(f"[LPIPS] Computed {len(all_lpips_rewards)} rewards")
    else:
        all_lpips_rewards = [torch.tensor([0.0], device=device) for _ in all_image_paths]
    
    main_print(f"[All stages done] latents: {len(all_latents)}, rewards: DEQA={len(all_deqa_rewards)}, QWEN={len(all_qwen_rewards)}, LPIPS={len(all_lpips_rewards)}")

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_deqa_rewards = torch.cat(all_deqa_rewards, dim=0) if all_deqa_rewards else None
    all_qwen_rewards = torch.cat(all_qwen_rewards, dim=0) if all_qwen_rewards else None
    all_lpips_rewards = torch.cat(all_lpips_rewards, dim=0) if all_lpips_rewards else None
    all_image_ids = torch.stack(all_image_ids, dim=0)
    all_control_images = torch.cat(all_control_images, dim=0)

    return all_deqa_rewards, all_qwen_rewards, all_lpips_rewards, all_latents, all_log_probs, sigma_schedule, all_image_ids, all_control_images


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


# MODIFIED: Added controlnet parameter and multi-reward support
def train_one_step(
    args,
    device,
    transformer,
    controlnet,
    vae,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    max_grad_norm,
    train_vae=False,
    train_transformer=True,
    deqa_reward_model=None,
    qwen_reward_model=None,
    qwen_processor=None,
    net_lpips=None,
):
    total_loss = 0.0
    optimizer.zero_grad()
    approx_kls = []
    clipfracs = []
    # MODIFIED: Dataset now returns control_image (LR image) and hr_image as well
    (
        encoder_hidden_states,
        pooled_prompt_embeds,
        text_ids,
        caption,
        control_image_raw,  # ADDED: Raw LR image tensor for ControlNet
        hr_image_raw,  # ADDED: Raw HR image tensor for LPIPS
    ) = next(loader)
    
    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        pooled_prompt_embeds = repeat_tensor(pooled_prompt_embeds)
        text_ids = repeat_tensor(text_ids)
        control_image_raw = repeat_tensor(control_image_raw)  # ADDED
        hr_image_raw = repeat_tensor(hr_image_raw)  # ADDED: HR image for LPIPS

        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

    # MODIFIED: Pass controlnet, control_image and multi-reward models
    (
        all_deqa_rewards, 
        all_qwen_rewards, 
        all_lpips_rewards,
        all_latents, 
        all_log_probs, 
        sigma_schedule, 
        all_image_ids, 
        all_control_images
    ) = sample_reference_model(
            args,
            device, 
            transformer,
            controlnet,
            vae,
            control_image_raw,
            hr_image_raw,
            encoder_hidden_states, 
            pooled_prompt_embeds, 
            text_ids,
            caption,
            train_vae=train_vae,
            deqa_reward_model=deqa_reward_model,
            qwen_reward_model=qwen_reward_model,
            qwen_processor=qwen_processor,
            net_lpips=net_lpips,
        )
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps =  torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)

    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[
            :, :-1
        ][:, :-1],  # each entry is the latent before timestep t
        "next_latents": all_latents[
            :, 1:
        ][:, :-1],  # each entry is the latent after timestep t
        "log_probs": all_log_probs[:, :-1],
        "deqa_rewards": all_deqa_rewards.to(torch.float32) if all_deqa_rewards is not None else torch.zeros(batch_size, device=device),
        "qwen_rewards": all_qwen_rewards.to(torch.float32) if all_qwen_rewards is not None else torch.zeros(batch_size, device=device),
        "lpips_rewards": all_lpips_rewards.to(torch.float32) if all_lpips_rewards is not None else torch.zeros(batch_size, device=device),
        "image_ids": all_image_ids,
        "text_ids": text_ids,
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "control_images": all_control_images,  # ADDED
    }
    
    # gather rewards across processes
    gathered_deqa_reward = gather_tensor(samples["deqa_rewards"])
    gathered_qwen_reward = gather_tensor(samples["qwen_rewards"])
    gathered_lpips_reward = gather_tensor(samples["lpips_rewards"])
    
    if dist.get_rank()==0:
        print("gathered_deqa_reward", gathered_deqa_reward)
        print("gathered_qwen_reward", gathered_qwen_reward)
        print("gathered_lpips_reward", gathered_lpips_reward)
        with open('./reward_sr_multi.txt', 'a') as f: 
            f.write(f"deqa: {gathered_deqa_reward.mean().item()}, qwen: {gathered_qwen_reward.mean().item()}, lpips: {gathered_lpips_reward.mean().item()}\n")

    if args.use_group:
        n = len(samples["lpips_rewards"]) // (args.num_generations)
        deqa_advantages = torch.zeros_like(samples["deqa_rewards"])
        qwen_advantages = torch.zeros_like(samples["qwen_rewards"])
        lpips_advantages = torch.zeros_like(samples["lpips_rewards"])
        
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            
            if args.use_deqa:
                deqa_group_rewards = samples["deqa_rewards"][start_idx:end_idx]
                deqa_group_mean = deqa_group_rewards.mean()
                deqa_group_std = deqa_group_rewards.std() + 1e-8
                deqa_advantages[start_idx:end_idx] = (deqa_group_rewards - deqa_group_mean) / deqa_group_std
            
            if args.use_qwen:
                qwen_group_rewards = samples["qwen_rewards"][start_idx:end_idx]
                qwen_group_mean = qwen_group_rewards.mean()
                qwen_group_std = qwen_group_rewards.std() + 1e-8
                qwen_advantages[start_idx:end_idx] = (qwen_group_rewards - qwen_group_mean) / qwen_group_std
            
            if args.use_lpips:
                lpips_group_rewards = samples["lpips_rewards"][start_idx:end_idx]
                lpips_group_mean = lpips_group_rewards.mean()
                lpips_group_std = lpips_group_rewards.std() + 1e-8
                lpips_advantages[start_idx:end_idx] = (lpips_group_rewards - lpips_group_mean) / lpips_group_std
        
        final_advantages = (
            args.deqa_weight * deqa_advantages + 
            args.qwen_weight * qwen_advantages + 
            args.lpips_weight * lpips_advantages
        )
        samples["advantages"] = final_advantages
        samples["final_advantages"] = final_advantages
    else:
        deqa_adv = (samples["deqa_rewards"] - gathered_deqa_reward.mean()) / (gathered_deqa_reward.std() + 1e-8) if args.use_deqa else torch.zeros_like(samples["deqa_rewards"])
        qwen_adv = (samples["qwen_rewards"] - gathered_qwen_reward.mean()) / (gathered_qwen_reward.std() + 1e-8) if args.use_qwen else torch.zeros_like(samples["qwen_rewards"])
        lpips_adv = (samples["lpips_rewards"] - gathered_lpips_reward.mean()) / (gathered_lpips_reward.std() + 1e-8) if args.use_lpips else torch.zeros_like(samples["lpips_rewards"])
        
        final_advantages = args.deqa_weight * deqa_adv + args.qwen_weight * qwen_adv + args.lpips_weight * lpips_adv
        samples["advantages"] = final_advantages
        samples["final_advantages"] = final_advantages

    
    perms = torch.stack(
        [
            torch.randperm(len(samples["timesteps"][0]))
            for _ in range(batch_size)
        ]
    ).to(device) 
    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device) [:, None],
            perms,
        ]
    samples_batched = {
        k: v.unsqueeze(1)
        for k, v in samples.items()
    }
    # dict of lists -> list of dicts for easier iteration
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ]
    train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction)
    grad_norm = torch.tensor(0.0)
    for i,sample in list(enumerate(samples_batched_list)):
        for _ in range(train_timesteps):
            clip_range = args.clip_range
            adv_clip_max = args.adv_clip_max
            # MODIFIED: Pass controlnet and control_images
            new_log_probs = grpo_one_step(
                args,
                sample["latents"][:,_],
                sample["next_latents"][:,_],
                sample["encoder_hidden_states"],
                sample["pooled_prompt_embeds"],
                sample["text_ids"],
                sample["image_ids"],
                transformer,
                controlnet,  # ADDED
                sample["control_images"],  # ADDED
                sample["timesteps"][:,_],
                perms[i][_],
                sigma_schedule,
            )

            advantages = torch.clamp(
                sample["advantages"],
                -adv_clip_max,
                adv_clip_max,
            )

            ratio = torch.exp(new_log_probs - sample["log_probs"][:,_])

            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            approx_kls.append(0.5 * torch.mean((new_log_probs - sample["log_probs"][:,_]) ** 2))
            clipfracs.append(torch.mean((torch.abs(ratio - 1.0) > clip_range).float()))
            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * train_timesteps)

            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
        if (i+1)%args.gradient_accumulation_steps==0:
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if dist.get_rank()%8==0:
            print("deqa_reward", sample["deqa_rewards"])
            print("qwen_reward", sample["qwen_rewards"])
            print("lpips_reward", sample["lpips_rewards"])
            print("ratio", ratio)
            print("advantage", sample["advantages"])
            print("final loss", loss.item())
        dist.barrier()
    approx_kl = torch.mean(torch.stack(approx_kls)) if approx_kls else torch.tensor(0.0, device=device)
    clipfrac = torch.mean(torch.stack(clipfracs)) if clipfracs else torch.tensor(0.0, device=device)
    return (
        total_loss,
        grad_norm.item(),
        gathered_deqa_reward.mean().item(),
        gathered_qwen_reward.mean().item(),
        gathered_lpips_reward.mean().item(),
        approx_kl.item(),
        clipfrac.item(),
        gathered_deqa_reward.std().item(),
        gathered_qwen_reward.std().item(),
        gathered_lpips_reward.std().item(),
    )


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # dist.init_process_group("nccl")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32
    
    # Validate training options
    if not args.train_transformer and not args.train_vae:
        raise ValueError("At least one of --train_transformer or --train_vae must be enabled!")
    
    main_print(f"--> Training configuration: transformer={args.train_transformer}, vae={args.train_vae}")
    
    # Load Flux Transformer
    transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype = torch.bfloat16
    )
    
    # ADDED: Load ControlNet for super-resolution
    main_print(f"--> loading ControlNet from {args.controlnet_model_name_or_path}")
    controlnet = FluxControlNetModel.from_transformer(
        transformer,
        attention_head_dim=transformer.config["attention_head_dim"],
        num_attention_heads=transformer.config["num_attention_heads"],
        num_layers=args.num_double_layers,
        num_single_layers=args.num_single_layers,
    )
    # Load ControlNet weights
    controlnet_state_dict = load_file(args.controlnet_model_name_or_path)
    controlnet.load_state_dict(controlnet_state_dict)
    main_print(f"--> ControlNet loaded with {args.num_double_layers} double layers and {args.num_single_layers} single layers")

    from diffusers import FluxPipeline
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from diffusers.utils import convert_unet_state_dict_to_peft
    pipe = FluxPipeline
    
    # ========== Transformer LoRA Setup ==========
    transformer.requires_grad_(False)
    if args.train_transformer:
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer.add_adapter(transformer_lora_config)
        main_print(f"--> Transformer LoRA added with rank={args.lora_rank}, alpha={args.lora_alpha}")

        if args.resume_from_lora_checkpoint:
            lora_state_dict = pipe.lora_state_dict(
                args.resume_from_lora_checkpoint)
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items() if k.startswith("transformer.")
            }
            transformer_state_dict = convert_unet_state_dict_to_peft(
                transformer_state_dict)
            incompatible_keys = set_peft_model_state_dict(transformer,
                                                          transformer_state_dict,
                                                          adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys",
                                          None)
                if unexpected_keys:
                    main_print(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. ")
        
        transformer.config.lora_rank = args.lora_rank
        transformer.config.lora_alpha = args.lora_alpha
        transformer.config.lora_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        True,
        args.use_cpu_offload,
        args.master_weight_type,
    )

    transformer._no_split_modules = [
        no_split_module.__name__ for no_split_module in no_split_modules
    ]
    fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](transformer)
    
    transformer = FSDP(transformer, **fsdp_kwargs,)

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )
    
    # ADDED: Move ControlNet to device (frozen, no FSDP needed)
    controlnet.requires_grad_(False)
    controlnet = controlnet.to(device, dtype=torch.bfloat16)
    
    # ========== VAE Setup (with optional LoRA) ==========
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32 if args.train_vae else torch.bfloat16,  # Use float32 for training
    ).to(device)
    
    if args.train_vae:
        # Add LoRA to VAE
        vae.requires_grad_(False)
        vae_lora_config = LoraConfig(
            r=args.vae_lora_rank,
            lora_alpha=args.vae_lora_alpha,
            init_lora_weights=True,
            # Target modules for VAE encoder and decoder
            target_modules=[
                "to_q", "to_k", "to_v", "to_out.0",  # Attention layers
                "conv1", "conv2",  # ResNet blocks
                "conv_shortcut",  # Skip connections
            ],
        )
        vae.add_adapter(vae_lora_config)
        main_print(f"--> VAE LoRA added with rank={args.vae_lora_rank}, alpha={args.vae_lora_alpha}")
        
        # Resume VAE LoRA if checkpoint exists
        if args.resume_from_vae_lora_checkpoint:
            vae_lora_state_dict = load_file(args.resume_from_vae_lora_checkpoint)
            vae.load_state_dict(vae_lora_state_dict, strict=False)
            main_print(f"--> Resumed VAE LoRA from {args.resume_from_vae_lora_checkpoint}")

    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    # Load the reference model
    main_print(f"--> model loaded")

    deqa_reward_model = None
    if args.use_deqa:
        from src.evaluate.scorer import Scorer_local
        deqa_reward_model = Scorer_local(
            pretrained=args.deqa_model_path,
            device="cpu",
            preprocessor_path=args.deqa_preprocessor_path
        )
        if hasattr(deqa_reward_model, "eval"):
            deqa_reward_model.eval()
        main_print(f"--> DEQA reward model loaded to CPU from {args.deqa_model_path}")

    qwen_reward_model = None
    qwen_processor = None
    if args.use_qwen:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        qwen_reward_model = AutoModelForImageTextToText.from_pretrained(
            args.qwen_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        qwen_reward_model.requires_grad_(False)
        qwen_reward_model.eval()
        qwen_reward_model.to("cpu")
        qwen_processor = AutoProcessor.from_pretrained(args.qwen_model_path)
        main_print(f"--> QWEN reward model loaded to CPU from {args.qwen_model_path}")

    net_lpips = None
    if args.use_lpips:
        net_lpips = lpips.LPIPS(net='vgg').to(device)
        net_lpips.requires_grad_(False)
        main_print(f"--> LPIPS model loaded to GPU")

    # Set models as trainable based on config
    if args.train_transformer:
        transformer.train()
    else:
        transformer.eval()
    
    if args.train_vae:
        vae.train()
    else:
        vae.eval()

    noise_scheduler = None

    # ========== Collect trainable parameters ==========
    params_to_optimize = []
    
    if args.train_transformer:
        transformer_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
        params_to_optimize.extend(transformer_params)
        main_print(f"--> Transformer trainable params: {sum(p.numel() for p in transformer_params) / 1e6:.2f}M")
    
    if args.train_vae:
        vae_params = list(filter(lambda p: p.requires_grad, vae.parameters()))
        params_to_optimize.extend(vae_params)
        main_print(f"--> VAE trainable params: {sum(p.numel() for p in vae_params) / 1e6:.2f}M")
    
    main_print(f"--> Total trainable params: {sum(p.numel() for p in params_to_optimize) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    if args.resume_from_lora_checkpoint and args.train_transformer:
        transformer, optimizer, init_steps = resume_lora_optimizer(
            transformer, args.resume_from_lora_checkpoint, optimizer
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    # MODIFIED: Use SR-specific dataset
    train_dataset = LatentSRDataset(
        json_path=args.data_json_path, 
        num_latent_t=args.num_latent_t, 
        cfg_rate=args.cfg, 
        target_height=args.h, 
        target_width=args.w,
        lr_image_root=args.lr_image_root,
        hr_image_root=args.hr_image_root,
    )
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )
    

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_sr_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    #vae.enable_tiling()

    tb_writer = None
    if rank <= 0:
        project = "flux_controlnet_sr"
        wandb.init(project=project, config=args)
        tb_log_dir = os.path.join(args.output_dir, "tensorboard") if args.output_dir else "./tensorboard"
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        main_print(f"  TensorBoard log dir: {tb_log_dir}")

    # Train!
    total_batch_size = (
        world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Training Transformer: {args.train_transformer}")
    main_print(f"  Training VAE: {args.train_vae}")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    if args.train_transformer:
        main_print(
            f"  Transformer training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
        )
    if args.train_vae:
        main_print(
            f"  VAE training parameters = {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6} M"
        )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)

    # The number of epochs 1 is a random value; you can also set the number of epochs to be two.
    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        for i in range(init_steps+1):
            next(loader)
        for step in range(init_steps+1, args.max_train_steps+1):
            start_time = time.time()
            if step % args.checkpointing_steps == 0:
                # Save Transformer LoRA checkpoint
                if args.train_transformer:
                    save_lora_checkpoint(transformer, optimizer, rank, args.output_dir,
                                            step, pipe, epoch)
                
                # Save VAE LoRA checkpoint
                if args.train_vae and rank == 0:
                    vae_lora_path = os.path.join(args.output_dir, f"vae_lora_step_{step}")
                    os.makedirs(vae_lora_path, exist_ok=True)
                    # Get VAE LoRA state dict
                    vae_lora_state_dict = {
                        k: v for k, v in vae.state_dict().items() 
                        if "lora" in k.lower()
                    }
                    save_file(vae_lora_state_dict, os.path.join(vae_lora_path, "vae_lora.safetensors"))
                    main_print(f"--> Saved VAE LoRA checkpoint to {vae_lora_path}")

                dist.barrier()
            # MODIFIED: Pass controlnet and multi-reward models to train_one_step
            (
                loss,
                grad_norm,
                deqa_reward_mean,
                qwen_reward_mean,
                lpips_reward_mean,
                approx_kl,
                clipfrac,
                deqa_reward_std,
                qwen_reward_std,
                lpips_reward_std,
            ) = train_one_step(
                args,
                device, 
                transformer,
                controlnet,
                vae,
                optimizer,
                lr_scheduler,
                loader,
                noise_scheduler,
                args.max_grad_norm,
                train_vae=args.train_vae,
                train_transformer=args.train_transformer,
                deqa_reward_model=deqa_reward_model,
                qwen_reward_model=qwen_reward_model,
                qwen_processor=qwen_processor,
                net_lpips=net_lpips,
            )
    
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                        "deqa_reward_mean": deqa_reward_mean,
                        "qwen_reward_mean": qwen_reward_mean,
                        "lpips_reward_mean": lpips_reward_mean,
                        "approx_kl": approx_kl,
                        "clipfrac": clipfrac,
                    },
                    step=step,
                )

                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", loss, step)
                    tb_writer.add_scalar("train/grad_norm", grad_norm, step)
                    tb_writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], step)
                    tb_writer.add_scalar("train/approx_kl", approx_kl, step)
                    tb_writer.add_scalar("train/clipfrac", clipfrac, step)

                    tb_writer.add_scalar("reward/deqa_mean", deqa_reward_mean, step)
                    tb_writer.add_scalar("reward/qwen_mean", qwen_reward_mean, step)
                    tb_writer.add_scalar("reward/lpips_mean", lpips_reward_mean, step)

                    tb_writer.add_scalar("reward/deqa_std", deqa_reward_std, step)
                    tb_writer.add_scalar("reward/qwen_std", qwen_reward_std, step)
                    tb_writer.add_scalar("reward/lpips_std", lpips_reward_std, step)

                    tb_writer.add_scalar("time/step_time", step_time, step)
                    tb_writer.add_scalar("time/avg_step_time", avg_step_time, step)

                    if step % 10 == 0:
                        tb_writer.flush()


    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
        main_print("--> TensorBoard writer closed")

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument(
        "--lr_image_root", 
        type=str, 
        default=None,
        help="Root directory for LR images. If None, uses {json_dir}/lr_images"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # ADDED: ControlNet-specific arguments
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained ControlNet model (safetensors file).",
    )
    parser.add_argument(
        "--num_double_layers",
        type=int,
        default=4,
        help="Number of double layers in the ControlNet (default: 4).",
    )
    parser.add_argument(
        "--num_single_layers",
        type=int,
        default=0,
        help="Number of single layers in the ControlNet (default: 0).",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
        help="ControlNet conditioning scale (default: 1.0).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="Guidance scale for classifier-free guidance (default: 2.5, same as test_flux_controlnet.py).",
    )

    # diffusion setting
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #GRPO training
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,   
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=1.0,   
        help="the global loss should be divided by",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether compute advantages for each prompt",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    )
    
    parser.add_argument(
        "--use_deqa",
        action="store_true",
        default=False,
        help="Whether to use DEQA as reward model",
    )
    parser.add_argument(
        "--deqa_model_path",
        type=str,
        default=None,
        help="Path to DEQA model checkpoint",
    )
    parser.add_argument(
        "--deqa_preprocessor_path",
        type=str,
        default=None,
        help="Path to DEQA preprocessor",
    )
    parser.add_argument(
        "--use_qwen",
        action="store_true",
        default=False,
        help="Whether to use QWEN as reward model",
    )
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        default=None,
        help="Path to RefReward (Qwen) model checkpoint",
    )
    parser.add_argument(
        "--use_lpips",
        action="store_true",
        default=False,
        help="Whether to use LPIPS as reward model",
    )
    parser.add_argument(
        "--hr_image_root",
        type=str,
        default=None,
        help="Root directory for HR images (for LPIPS calculation)",
    )
    parser.add_argument(
        "--deqa_weight",
        type=float,
        default=1.0,
        help="Weight for DEQA reward in final advantage",
    )
    parser.add_argument(
        "--qwen_weight",
        type=float,
        default=1.0,
        help="Weight for QWEN reward in final advantage",
    )
    parser.add_argument(
        "--lpips_weight",
        type=float,
        default=1.0,
        help="Weight for LPIPS reward in final advantage",
    )
    
    parser.add_argument(
        "--ignore_last",
        action="store_true",
        default=False,
        help="whether ignore last step of mdp",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=False,
        help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep downsample ratio",
    )
    parser.add_argument(
        "--clip_range",
        type = float,
        default=1e-4,
        help="clip range for grpo",
    )
    parser.add_argument(
        "--adv_clip_max",
        type = float,
        default=5.0,
        help="clipping advantage",
    )
    # ========== LoRA Training Configuration ==========
    parser.add_argument(
        "--train_transformer",
        action="store_true",
        default=True,
        help="Whether to train Transformer with LoRA (default: True).",
    )
    parser.add_argument(
        "--no_train_transformer",
        action="store_false",
        dest="train_transformer",
        help="Disable Transformer LoRA training.",
    )
    parser.add_argument(
        "--train_vae",
        action="store_true",
        default=False,
        help="Whether to train VAE with LoRA (default: False).",
    )
    
    # Transformer LoRA parameters
    parser.add_argument(
        "--lora_alpha", type=int, default=256, help="Alpha parameter for Transformer LoRA."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=128, help="Transformer LoRA rank parameter."
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous Transformer LoRA checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    
    # VAE LoRA parameters
    parser.add_argument(
        "--vae_lora_alpha", type=int, default=64, help="Alpha parameter for VAE LoRA."
    )
    parser.add_argument(
        "--vae_lora_rank", type=int, default=32, help="VAE LoRA rank parameter."
    )
    parser.add_argument(
        "--resume_from_vae_lora_checkpoint",
        type=str,
        default=None,
        help="Path to VAE LoRA checkpoint to resume from.",
    )




    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        # Log exception to a text file for easier debugging in batch runs.
        log_dir = args.output_dir if args.output_dir else "."
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "error_log.txt")
        with open(log_path, "a") as f:
            f.write(f"\n[{datetime.utcnow().isoformat()}Z] Unhandled exception\n")
            f.write("Command: " + " ".join(sys.argv) + "\n")
            f.write("Exception: " + repr(exc) + "\n")
            f.write(traceback.format_exc())
            f.write("\n")
        # Re-raise to keep exit status and stack trace.
        raise
