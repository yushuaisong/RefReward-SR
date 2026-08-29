#!/usr/bin/env python
# coding=utf-8
# Derived from https://github.com/cswry/DP2O-SR/blob/main/test_flux_controlnet.py.
# Modified by Institute of Automation, Chinese Academy of Sciences in 2026.

# ============================================================================
# Test script for Flux ControlNet SR with LoRA weights.
# ============================================================================
# Example:
# python test_flux_controlnet_sr_lora.py \
#   --image_path /path/to/test_LR \
#   --output_dir /path/to/output \
#   --pretrained_model_name_or_path /path/to/FLUX.1-dev \
#   --controlnet_model_name_or_path /path/to/controlnet/model.safetensors \
#   --lora_checkpoint_dir /path/to/lora-checkpoint \
#   --ram_path /path/to/ram_swin_large_14m.pth \
#   --dape_path /path/to/DAPE.pth


import argparse
import math
import os
from pathlib import Path

# Optional: Hugging Face mirror (default is the official endpoint)
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from safetensors.torch import load_file

from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxPipeline
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel

# Optional: RAM for auto-captioning
try:
    import sys
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    RAM_AVAILABLE = True
except ImportError:
    RAM_AVAILABLE = False
    print("RAM not available, will use provided captions or default prompt")

# Optional: Color fix utilities
try:
    from utils_self.wavelet_color_fix import wavelet_color_fix, adain_color_fix
    COLOR_FIX_AVAILABLE = True
except ImportError:
    COLOR_FIX_AVAILABLE = False
    print("Color fix utilities not available")


def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
    return latents


def prepare_control_image(control_image, vae, device, dtype, batch_size, height, width):
    with torch.no_grad():
        control_image = control_image.to(device=device, dtype=torch.float32)
        control_latents = vae.encode(control_image).latent_dist.sample()
        control_latents = (control_latents - 0.1159) * 0.3611
        _, c, h, w = control_latents.shape
        control_latents = pack_latents(control_latents, batch_size, c, h, w)
    return control_latents.to(dtype=dtype)


def flux_step_deterministic(model_output, latents, sigmas, index):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample = latents + dsigma * model_output
    pred_original_sample = latents - sigma * model_output
    return prev_sample, pred_original_sample


def run_inference_step(
    args,
    z,
    sigma_schedule,
    transformer,
    controlnet,
    control_image,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    image_ids,
    device,
):
    sample_steps = len(sigma_schedule) - 1
    progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
    for i in progress_bar:
        sigma = sigma_schedule[i]
        timestep_value = int(sigma * 1000)
        timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=device, dtype=torch.long)
        with torch.no_grad():
            with torch.autocast("cuda", torch.bfloat16):
                controlnet_block_samples, controlnet_single_block_samples = controlnet(
                    hidden_states=z,
                    controlnet_cond=control_image,
                    controlnet_mode=None,
                    conditioning_scale=args.controlnet_conditioning_scale,
                    timestep=timesteps / 1000,
                    guidance=torch.tensor(
                        [args.guidance_scale],
                        device=device,
                        dtype=torch.bfloat16
                    ),
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    txt_ids=text_ids,
                    img_ids=image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )
                pred = transformer(
                    hidden_states=z,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps / 1000,
                    guidance=torch.tensor(
                        [args.guidance_scale],
                        device=device,
                        dtype=torch.bfloat16
                    ),
                    txt_ids=text_ids,
                    pooled_projections=pooled_prompt_embeds,
                    img_ids=image_ids,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    joint_attention_kwargs=None,
                    return_dict=False,
                    controlnet_blocks_repeat=False,
                )[0]
        z, pred_original = flux_step_deterministic(pred, z.to(torch.float32), sigma_schedule, i)
        z = z.to(torch.bfloat16)
    return pred_original


def encode_prompt(prompt, tokenizer_clip, tokenizer_t5, text_encoder_clip, text_encoder_t5, device, max_sequence_length=512):
    text_inputs_clip = tokenizer_clip(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids_clip = text_inputs_clip.input_ids.to(device)
    with torch.no_grad():
        prompt_embeds_clip = text_encoder_clip(text_input_ids_clip, output_hidden_states=False)
        pooled_prompt_embeds = prompt_embeds_clip.pooler_output

    text_inputs_t5 = tokenizer_t5(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids_t5 = text_inputs_t5.input_ids.to(device)
    with torch.no_grad():
        prompt_embeds = text_encoder_t5(text_input_ids_t5)[0]

    text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=prompt_embeds.dtype)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def load_lora_config(lora_checkpoint_dir, fallback_rank, fallback_alpha, fallback_targets):
    config_path = os.path.join(lora_checkpoint_dir, "lora_config.json")
    if not os.path.exists(config_path):
        return fallback_rank, fallback_alpha, fallback_targets
    import json
    with open(config_path, "r") as f:
        cfg = json.load(f)
    params = cfg.get("lora_params", {})
    return (
        int(params.get("lora_rank", fallback_rank)),
        int(params.get("lora_alpha", fallback_alpha)),
        params.get("target_modules", fallback_targets),
    )


def load_transformer_lora(transformer, lora_checkpoint_dir, lora_rank, lora_alpha, target_modules):
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True,
        target_modules=target_modules,
    )
    transformer.add_adapter(lora_config)
    lora_state_dict = FluxPipeline.lora_state_dict(lora_checkpoint_dir)
    transformer_state_dict = {
        f'{k.replace("transformer.", "")}': v
        for k, v in lora_state_dict.items() if k.startswith("transformer.")
    }
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    incompatible_keys = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")
    if incompatible_keys is not None:
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            print(f"Warning: unexpected LoRA keys not found in model: {unexpected_keys}")
    return transformer


def load_models(args, device):
    print(f"Loading models from {args.pretrained_model_name_or_path}...")
    tokenizer_clip = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_t5 = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")

    text_encoder_clip = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    ).to(device, dtype=torch.bfloat16)
    text_encoder_clip.eval()

    text_encoder_t5 = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    ).to(device, dtype=torch.bfloat16)
    text_encoder_t5.eval()

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    ).to(device, dtype=torch.float32)
    vae.eval()

    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer"
    ).to(device, dtype=torch.bfloat16)
    transformer.eval()

    if args.lora_checkpoint_dir:
        lora_rank, lora_alpha, target_modules = load_lora_config(
            args.lora_checkpoint_dir,
            args.lora_rank,
            args.lora_alpha,
            args.lora_target_modules,
        )
        transformer = load_transformer_lora(
            transformer, args.lora_checkpoint_dir, lora_rank, lora_alpha, target_modules
        )
        print(f"Loaded LoRA: rank={lora_rank}, alpha={lora_alpha}, targets={target_modules}")

    print(f"Loading ControlNet from {args.controlnet_model_name_or_path}...")
    controlnet = FluxControlNetModel.from_transformer(
        transformer,
        attention_head_dim=transformer.config["attention_head_dim"],
        num_attention_heads=transformer.config["num_attention_heads"],
        num_layers=args.num_double_layers,
        num_single_layers=args.num_single_layers,
    )
    controlnet_state_dict = load_file(args.controlnet_model_name_or_path)
    controlnet.load_state_dict(controlnet_state_dict)
    controlnet = controlnet.to(device, dtype=torch.bfloat16)
    controlnet.eval()

    print("All models loaded successfully!")
    return {
        "transformer": transformer,
        "controlnet": controlnet,
        "vae": vae,
        "tokenizer_clip": tokenizer_clip,
        "tokenizer_t5": tokenizer_t5,
        "text_encoder_clip": text_encoder_clip,
        "text_encoder_t5": text_encoder_t5,
    }


def process_single_image(
    args,
    image_path,
    models,
    device,
    ram_model=None,
    generator=None,
):
    transformer = models["transformer"]
    controlnet = models["controlnet"]
    vae = models["vae"]
    tokenizer_clip = models["tokenizer_clip"]
    tokenizer_t5 = models["tokenizer_t5"]
    text_encoder_clip = models["text_encoder_clip"]
    text_encoder_t5 = models["text_encoder_t5"]

    validation_image = Image.open(str(image_path)).convert("RGB")
    ori_width, ori_height = validation_image.size

    if ram_model is not None and RAM_AVAILABLE:
        tensor_transforms = transforms.Compose([transforms.ToTensor()])
        ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        lq = tensor_transforms(validation_image).unsqueeze(0).to(device)
        lq = ram_transforms(lq)
        res = inference(lq, ram_model)
        prompt = f"{res[0]}"
    else:
        prompt = args.default_prompt if args.default_prompt else "high quality, detailed, sharp"

    print(f"  Prompt: {prompt}")

    rscale = args.upscale
    if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
        scale = (args.process_size // rscale) / min(ori_width, ori_height)
        validation_image = validation_image.resize((int(scale * ori_width), int(scale * ori_height)))

    validation_image = validation_image.resize((validation_image.size[0] * rscale, validation_image.size[1] * rscale))
    validation_image = validation_image.resize((validation_image.size[0] // 8 * 8, validation_image.size[1] // 8 * 8))
    width, height = validation_image.size

    validation_image_tensor = (torch.from_numpy(np.array(validation_image)).permute(2, 0, 1).float() / 255.0) * 2 - 1
    validation_image_tensor = validation_image_tensor.unsqueeze(0).to(device)

    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
        prompt, tokenizer_clip, tokenizer_t5, text_encoder_clip, text_encoder_t5, device
    )

    spatial_downsample = 8
    in_channels = 16
    latent_h, latent_w = height // spatial_downsample, width // spatial_downsample

    control_image_packed = prepare_control_image(
        validation_image_tensor, vae, device, torch.bfloat16,
        1, latent_h, latent_w
    )

    if generator is not None:
        input_latents = torch.randn(
            (1, in_channels, latent_h, latent_w),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
    else:
        input_latents = torch.randn(
            (1, in_channels, latent_h, latent_w),
            device=device,
            dtype=torch.bfloat16,
        )

    input_latents_packed = pack_latents(input_latents, 1, in_channels, latent_h, latent_w)
    image_ids = prepare_latent_image_ids(1, latent_h // 2, latent_w // 2, device, torch.bfloat16)

    sigma_schedule = torch.linspace(1, 0, args.num_inference_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    pred_original = run_inference_step(
        args,
        input_latents_packed,
        sigma_schedule,
        transformer,
        controlnet,
        control_image_packed,
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
        image_ids,
        device,
    )

    vae.enable_tiling()
    image_processor = VaeImageProcessor(16)

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latents = unpack_latents(pred_original, height, width, 8)
            latents = (latents / 0.3611) + 0.1159
            image = vae.decode(latents.to(torch.float32), return_dict=False)[0]
            decoded_image = image_processor.postprocess(image)

    result_image = decoded_image[0]

    if COLOR_FIX_AVAILABLE and args.align_method != "nofix":
        if args.align_method == "wavelet":
            result_image = wavelet_color_fix(result_image, validation_image)
        elif args.align_method == "adain":
            result_image = adain_color_fix(result_image, validation_image)

    result_image = result_image.resize((ori_width * rscale, ori_height * rscale))
    return result_image


def main(args):
    if args.gpu_id is not None and args.gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    models = load_models(args, device)

    ram_model = None
    if RAM_AVAILABLE and args.ram_path and os.path.exists(args.ram_path):
        print("Loading RAM model for auto-captioning...")
        ram_model = ram(
            pretrained=args.ram_path,
            pretrained_condition=args.dape_path,
            image_size=384,
            vit="swin_l"
        )
        ram_model.eval()
        ram_model.to(device)

    base_path = Path(args.image_path)
    if base_path.is_file():
        image_list = [base_path]
    else:
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp"]
        image_list = []
        for ext in image_extensions:
            image_list.extend(base_path.glob(ext))
            image_list.extend(base_path.glob(ext.upper()))
        image_list = sorted(image_list)

    print(f"Found {len(image_list)} images to process")
    for idx, image_path in enumerate(image_list):
        print(f"\nProcessing [{idx + 1}/{len(image_list)}]: {image_path}")
        for sample_idx in range(args.sample_times):
            result_image = process_single_image(
                args,
                image_path,
                models,
                device,
                ram_model,
                generator,
            )
            basename = os.path.basename(image_path).split(".")[0]
            if args.sample_times > 1:
                out_path = os.path.join(args.output_dir, f"{basename}_sample{sample_idx}.png")
            else:
                out_path = os.path.join(args.output_dir, f"{basename}.png")
            result_image.save(out_path)
            print(f"  Saved: {out_path}")

    print("\nAll images processed!")


def parse_args():
    parser = argparse.ArgumentParser(description="Test Flux ControlNet SR with manual sampling + LoRA")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output_dir", type=str, default="./output_sr_manual", help="Output directory")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Path to pretrained Flux model")
    parser.add_argument("--controlnet_model_name_or_path", type=str, required=True, help="Path to ControlNet model")

    parser.add_argument("--lora_checkpoint_dir", type=str, required=True, help="LoRA checkpoint directory")
    parser.add_argument("--lora_rank", type=int, default=32, help="Fallback LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="Fallback LoRA alpha")
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["to_k", "to_q", "to_v", "to_out.0"],
        help="Fallback LoRA target modules",
    )
    parser.add_argument("--num_double_layers", type=int, default=4, help="Number of double layers in ControlNet")
    parser.add_argument("--num_single_layers", type=int, default=0, help="Number of single layers in ControlNet")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0, help="ControlNet conditioning scale")

    parser.add_argument("--guidance_scale", type=float, default=2.5, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--shift", type=float, default=3.0, help="Time shift for sigma schedule")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU id to use (e.g., 0). If not set, use default CUDA device.",
    )

    parser.add_argument("--process_size", type=int, default=512, help="Process size")
    parser.add_argument("--upscale", type=int, default=4, help="Upscale factor")
    parser.add_argument("--sample_times", type=int, default=1, help="Number of samples per image")
    parser.add_argument("--align_method", choices=["wavelet", "adain", "nofix"], default="adain", help="Color alignment method")

    parser.add_argument("--ram_path", type=str, default="preset/models/ram_swin_large_14m.pth", help="Path to RAM model")
    parser.add_argument("--dape_path", type=str, default="preset/models/DAPE.pth", help="Path to DAPE model")
    parser.add_argument("--default_prompt", type=str, default="", help="Default prompt if RAM not available")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
