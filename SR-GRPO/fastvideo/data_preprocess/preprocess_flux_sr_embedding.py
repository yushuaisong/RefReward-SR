# Copyright (c) [2025] [FastVideo Team]
# SPDX-License-Identifier: [Apache License 2.0]
#
# Modified by Institute of Automation, Chinese Academy of Sciences in 2026.
# Preprocessing script to generate FLUX text embeddings for SR training.
# Input:
#   - LR images: sr_bicubic/0000001.png, 0000002.png, ...
#   - Prompts:   tag/0000001.txt, tag/0000002.txt, ...
# Output:
#   - prompt_embed/
#   - pooled_prompt_embeds/
#   - text_ids/
#   - videos2caption.json

import argparse
import torch
import json
import os
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from diffusers import FluxPipeline
import re
from glob import glob


def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))


class SRPromptDataset(Dataset):
    def __init__(self, data_root, max_samples=None):
        self.data_root = data_root
        self.lr_dir = os.path.join(data_root, "sr_bicubic")
        self.tag_dir = os.path.join(data_root, "tag")
        
        tag_files = sorted(glob(os.path.join(self.tag_dir, "*.txt")))
        
        self.samples = []
        for tag_file in tag_files:
            basename = os.path.basename(tag_file).replace(".txt", "")
            lr_image = os.path.join(self.lr_dir, basename + ".png")
            
            if not os.path.exists(lr_image):
                continue
                
            with open(tag_file, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            
            if contains_chinese(caption):
                continue
                
            self.samples.append({
                "basename": basename,
                "caption": caption,
                "lr_image_path": basename + ".png",
            })
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples from {data_root}")

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def collate_fn(batch):
    return batch


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print(f"world_size: {world_size}, local_rank: {local_rank}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=local_rank
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "pooled_prompt_embeds"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "text_ids"), exist_ok=True)

    train_dataset = SRPromptDataset(args.data_root, max_samples=args.max_samples)
    
    if world_size > 1:
        sampler = DistributedSampler(
            train_dataset, rank=local_rank, num_replicas=world_size, shuffle=False
        )
    else:
        sampler = None
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=(sampler is None),
    )

    print(f"Loading FLUX pipeline from {args.model_path}...")
    pipe = FluxPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # Drop transformer & VAE to save memory; only text encoders are needed.
    pipe.transformer = None
    pipe.vae = None
    torch.cuda.empty_cache()

    json_data = []
    
    for batch in tqdm(train_dataloader, disable=(local_rank != 0), desc="Encoding prompts"):
        try:
            with torch.inference_mode():
                for sample in batch:
                    basename = sample["basename"]
                    caption = sample["caption"]
                    lr_image_path = sample["lr_image_path"]
                    
                    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                        prompt=caption, 
                        prompt_2=caption
                    )
                    
                    prompt_embed_path = os.path.join(
                        args.output_dir, "prompt_embed", basename + ".pt"
                    )
                    pooled_prompt_embeds_path = os.path.join(
                        args.output_dir, "pooled_prompt_embeds", basename + ".pt"
                    )
                    text_ids_path = os.path.join(
                        args.output_dir, "text_ids", basename + ".pt"
                    )
                    
                    torch.save(prompt_embeds[0].cpu(), prompt_embed_path)
                    torch.save(pooled_prompt_embeds[0].cpu(), pooled_prompt_embeds_path)
                    torch.save(text_ids[0].cpu(), text_ids_path)
                    
                    item = {
                        "prompt_embed_path": basename + ".pt",
                        "pooled_prompt_embeds_path": basename + ".pt",
                        "text_ids": basename + ".pt",
                        "caption": caption,
                        "lr_image_path": lr_image_path,
                        "length": 1,
                    }
                    json_data.append(item)
                    
        except Exception as e:
            print(f"Rank {local_rank} Error: {repr(e)}")
            if world_size > 1:
                dist.barrier()
            raise

    if world_size > 1:
        dist.barrier()
        gathered_data = [None] * world_size
        dist.all_gather_object(gathered_data, json_data)
        
        if local_rank == 0:
            all_json_data = [item for sublist in gathered_data for item in sublist]
            json_output_path = os.path.join(args.output_dir, "videos2caption.json")
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(all_json_data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(all_json_data)} samples to {json_output_path}")
    else:
        json_output_path = os.path.join(args.output_dir, "videos2caption.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(json_data)} samples to {json_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SR data for FLUX GRPO training")
    
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="/path/to/training_for_seesr",
        help="Root directory containing sr_bicubic/ and tag/ folders"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./data/sr_embeddings",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/path/to/FLUX.1-dev",
        help="Path to FLUX model"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    
    args = parser.parse_args()
    main(args)
