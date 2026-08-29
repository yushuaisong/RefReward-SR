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
# Adapted from latent_flux_rl_datasets.py for SR training with LR ControlNet input.
import torch
from torch.utils.data import Dataset
import json
import os
import random
from PIL import Image
import numpy as np
from torchvision import transforms


class LatentSRDataset(Dataset):
    """
    Dataset for super-resolution GRPO training.
    
    Args:
        json_path: Path to videos2caption.json
        num_latent_t: Not used, kept for compatibility
        cfg_rate: Probability of using unconditional (empty) prompt
        target_height: Target height for LR image resize
        target_width: Target width for LR image resize
        lr_image_root: Root directory for LR images (if None, uses json_path's parent dir + "lr_images")
    
    Returns:
        - prompt_embed: T5 encoded text features
        - pooled_prompt_embeds: CLIP encoded text features  
        - text_ids: Text IDs
        - caption: Original text caption
        - control_image: Low-resolution image tensor for ControlNet input
    """
    def __init__(
        self, 
        json_path, 
        num_latent_t, 
        cfg_rate, 
        target_height=512, 
        target_width=512,
        lr_image_root=None,
        hr_image_root=None,
    ):
        self.json_path = json_path
        self.cfg_rate = cfg_rate
        self.datase_dir_path = os.path.dirname(json_path)
        
        self.prompt_embed_dir = os.path.join(self.datase_dir_path, "prompt_embed")
        self.pooled_prompt_embeds_dir = os.path.join(self.datase_dir_path, "pooled_prompt_embeds")
        self.text_ids_dir = os.path.join(self.datase_dir_path, "text_ids")
        
        if lr_image_root is not None:
            self.lr_image_dir = lr_image_root
        else:
            self.lr_image_dir = os.path.join(self.datase_dir_path, "lr_images")
        
        if hr_image_root is not None:
            self.hr_image_dir = hr_image_root
        else:
            self.hr_image_dir = None
        
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data_anno = json.load(f)
        
        self.num_latent_t = num_latent_t
        
        self.uncond_prompt_embed = torch.zeros(256, 4096).to(torch.float32)
        self.uncond_prompt_mask = torch.zeros(256).bool()
        
        self.lengths = [
            data_item.get("length", 1) for data_item in self.data_anno
        ]
        
        self.target_height = target_height
        self.target_width = target_width
        
        self.lr_transform = transforms.Compose([
            transforms.Resize(
                (target_height, target_width), 
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        self.hr_transform = transforms.Compose([
            transforms.Resize(
                (target_height, target_width), 
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        print(f"LatentSRDataset initialized:")
        print(f"  - JSON path: {json_path}")
        print(f"  - LR image dir: {self.lr_image_dir}")
        print(f"  - HR image dir: {self.hr_image_dir}")
        print(f"  - Num samples: {len(self.data_anno)}")
        print(f"  - Target size: {target_height}x{target_width}")
        print(f"  - CFG rate: {cfg_rate}")

    def __getitem__(self, idx):
        prompt_embed_file = self.data_anno[idx]["prompt_embed_path"]
        pooled_prompt_embeds_file = self.data_anno[idx]["pooled_prompt_embeds_path"]
        text_ids_file = self.data_anno[idx]["text_ids"]
        lr_image_file = self.data_anno[idx].get("lr_image_path", None)
        
        if random.random() < self.cfg_rate:
            prompt_embed = self.uncond_prompt_embed
            pooled_prompt_embeds = torch.zeros(768).to(torch.float32)
            text_ids = torch.zeros(256, 3).to(torch.float32)
        else:
            prompt_embed = torch.load(
                os.path.join(self.prompt_embed_dir, prompt_embed_file),
                map_location="cpu",
                weights_only=True,
            )
            pooled_prompt_embeds = torch.load(
                os.path.join(self.pooled_prompt_embeds_dir, pooled_prompt_embeds_file),
                map_location="cpu",
                weights_only=True,
            )
            text_ids = torch.load(
                os.path.join(self.text_ids_dir, text_ids_file),
                map_location="cpu",
                weights_only=True,
            )
        
        if lr_image_file is not None:
            lr_image_path = os.path.join(self.lr_image_dir, lr_image_file)
            lr_image = Image.open(lr_image_path).convert("RGB")
            control_image = self.lr_transform(lr_image)
        else:
            control_image = torch.zeros(3, self.target_height, self.target_width)
        
        if self.hr_image_dir is not None and lr_image_file is not None:
            hr_image_path = os.path.join(self.hr_image_dir, lr_image_file)
            if os.path.exists(hr_image_path):
                hr_image = Image.open(hr_image_path).convert("RGB")
                hr_image_tensor = self.hr_transform(hr_image)
            else:
                hr_image_tensor = torch.zeros(3, self.target_height, self.target_width)
        else:
            hr_image_tensor = torch.zeros(3, self.target_height, self.target_width)
        
        caption = self.data_anno[idx].get('caption', '')
        
        return prompt_embed, pooled_prompt_embeds, text_ids, caption, control_image, hr_image_tensor

    def __len__(self):
        return len(self.data_anno)


def latent_sr_collate_function(batch):
    """
    Collate function for SR dataset.
    
    Returns:
        - prompt_embeds: (B, L, D) T5 encoded features
        - pooled_prompt_embeds: (B, D) CLIP encoded features
        - text_ids: (B, L, 3) Text IDs
        - caption: List of captions
        - control_images: (B, 3, H, W) LR images for ControlNet
        - hr_images: (B, 3, H, W) HR images for LPIPS calculation
    """
    prompt_embeds, pooled_prompt_embeds, text_ids, caption, control_images, hr_images = zip(*batch)
    
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    pooled_prompt_embeds = torch.stack(pooled_prompt_embeds, dim=0)
    text_ids = torch.stack(text_ids, dim=0)
    control_images = torch.stack(control_images, dim=0)
    hr_images = torch.stack(hr_images, dim=0)
    
    return prompt_embeds, pooled_prompt_embeds, text_ids, caption, control_images, hr_images


# ============================================================================
# Alternative dataset: directly load from original SeeSR data structure
# ============================================================================
class LatentSRDatasetDirect(Dataset):
    """
    Load directly from SeeSR original data structure without pre-computed embeddings.
    
    Directory layout:
        {data_root}/
        ├── sr_bicubic/  # LR images
        │   ├── 0000001.png
        │   └── ...
        └── tag/         # Prompt files
            ├── 0000001.txt
            └── ...
    
    Note: requires real-time prompt encoding during training, which is less efficient.
    """
    def __init__(
        self, 
        data_root,
        target_height=512, 
        target_width=512,
        max_samples=None,
    ):
        self.data_root = data_root
        self.lr_dir = os.path.join(data_root, "sr_bicubic")
        self.tag_dir = os.path.join(data_root, "tag")
        
        self.target_height = target_height
        self.target_width = target_width
        
        from glob import glob
        tag_files = sorted(glob(os.path.join(self.tag_dir, "*.txt")))
        
        self.samples = []
        for tag_file in tag_files:
            basename = os.path.basename(tag_file).replace(".txt", "")
            lr_image = os.path.join(self.lr_dir, basename + ".png")
            
            if os.path.exists(lr_image):
                self.samples.append({
                    "basename": basename,
                    "lr_image_path": lr_image,
                    "tag_path": tag_file,
                })
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        self.lr_transform = transforms.Compose([
            transforms.Resize(
                (target_height, target_width), 
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        print(f"LatentSRDatasetDirect initialized:")
        print(f"  - Data root: {data_root}")
        print(f"  - Num samples: {len(self.samples)}")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        lr_image = Image.open(sample["lr_image_path"]).convert("RGB")
        control_image = self.lr_transform(lr_image)
        
        with open(sample["tag_path"], "r", encoding="utf-8") as f:
            caption = f.read().strip()
        
        return control_image, caption

    def __len__(self):
        return len(self.samples)


# ============================================================================
# Test code
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing LatentSRDataset (with pre-computed embeddings)")
    print("=" * 60)
    
    json_path = "/path/to/sr_embeddings/videos2caption.json"
    lr_image_root = "/path/to/train_datasets/sr_bicubic"
    
    if os.path.exists(json_path):
        dataset = LatentSRDataset(
            json_path=json_path,
            num_latent_t=1, 
            cfg_rate=0.0,
            target_height=512,
            target_width=512,
            lr_image_root=lr_image_root,
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False, 
            collate_fn=latent_sr_collate_function
        )
        
        for prompt_embed, pooled_prompt_embeds, text_ids, caption, control_image, hr_image in dataloader:
            print(f"prompt_embed: {prompt_embed.shape}")
            print(f"pooled_prompt_embeds: {pooled_prompt_embeds.shape}")
            print(f"text_ids: {text_ids.shape}")
            print(f"caption: {caption}")
            print(f"control_image: {control_image.shape}, range: [{control_image.min():.2f}, {control_image.max():.2f}]")
            print(f"hr_image: {hr_image.shape}, range: [{hr_image.min():.2f}, {hr_image.max():.2f}]")
            break
    else:
        print(f"JSON file not found: {json_path}")
        print("Please run preprocess_flux_sr_embedding.py first.")
    
    print()
    print("=" * 60)
    print("Testing LatentSRDatasetDirect (without pre-computed embeddings)")
    print("=" * 60)
    
    data_root = "/path/to/train_datasets"
    
    if os.path.exists(data_root):
        dataset_direct = LatentSRDatasetDirect(
            data_root=data_root,
            target_height=512,
            target_width=512,
            max_samples=10,
        )
        
        control_image, caption = dataset_direct[0]
        print(f"control_image: {control_image.shape}, range: [{control_image.min():.2f}, {control_image.max():.2f}]")
        print(f"caption: {caption}")
    else:
        print(f"Data root not found: {data_root}")
