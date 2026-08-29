<h1 align="center">(ECCV 2026) RefReward-SR: LR-Conditioned Reward Modeling for Preference-Aligned Super-Resolution</h1>

<div align="center">
  <a href="https://arxiv.org/abs/2603.24198">
    <img src="https://img.shields.io/badge/Paper%20(arXiv)-2603.24198-red?logo=arxiv">
  </a>
</div>

## 🔎 Overview

> **Abstract:** Recent advances in generative super-resolution (SR) have greatly improved visual realism, yet existing evaluation and optimization frameworks remain misaligned with human perception. Full-Reference and No-Reference metrics often fail to reflect perceptual preference, either penalizing semantically plausible details due to pixel misalignment or favoring visually sharp but inconsistent artifacts. Moreover, most SR methods rely on ground-truth (GT)–dependent distribution matching, which does not necessarily correspond to human judgments. In this work, we propose RefReward-SR, a low-resolution (LR) reference-aware reward model for preference-aligned SR. Instead of relying on GT supervision or NR evaluation, RefReward-SR assesses high-resolution (HR) reconstructions conditioned on their LR inputs, treating the LR image as a semantic anchor. Leveraging the visual–linguistic priors of a Multimodal Large Language Model (MLLM), it evaluates semantic consistency and plausibility in a reasoning-aware manner. To support this paradigm, we construct RefSR-18K, the first large-scale LR-conditioned preference dataset for SR, providing pairwise rankings based on LR–HR consistency and HR naturalness. We fine-tune the MLLM with Group Relative Policy Optimization (GRPO) using LR-conditioned ranking rewards, and further integrate GRPO into SR model training with RefReward-SR as the core reward signal for preference-aligned generation. Extensive experiments show that our framework achieves substantially better alignment with human judgments, producing reconstructions that preserve semantic consistency while enhancing perceptual plausibility and visual naturalness.

<p align="center">
  <img src="figs/img1.png" alt="Overview of RefReward-SR">
</p>

<p align="center">
  <img src="figs/img2.png" alt="RefReward-SR framework">
</p>

## Repository Structure

The project consists of two components:

1. **RefReward-SR** (`RefReward-SR/`) — Training and evaluating the LR-conditioned reward model.
2. **SR-GRPO** (`SR-GRPO/`) — Integrating RefReward-SR into GRPO-based fine-tuning of a FLUX ControlNet SR model.

## Released Weights and Data

| Item | Description | Hugging Face | Size |
|------|-------------|--------------|------|
| RefReward-SR | Qwen3-VL-8B LR-conditioned reward model trained via GRPO | [casyssong/RefReward-SR](https://huggingface.co/casyssong/RefReward-SR) | 17G |
| SR-GRPO-LoRA | LoRA adapter for FLUX ControlNet SR (RefReward-SR + DeQA-Score + LPIPS) | [casyssong/SR-GRPO-LoRA](https://huggingface.co/casyssong/SR-GRPO-LoRA) | 72M |
| RefSR-18K | LR-conditioned preference dataset | [casyssong/RefSR-18K](https://huggingface.co/datasets/casyssong/RefSR-18K) | ~22.2G |

## Environment Setup

**Tested on:** Python 3.10, CUDA 12.8, 8×H20 GPUs.

```bash
# 1. Create a new conda environment
conda create -n refreward python=3.10 -y
conda activate refreward

# 2. Install PyTorch (match your CUDA version, see https://pytorch.org)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Install flash-attn (must be compiled separately)
pip install flash-attn==2.7.3 --no-build-isolation

# 4. Install remaining dependencies (from the repository root)
pip install -r requirements.txt
```

### RAM + DAPE

RAM and DAPE are used for auto-captioning during SR-GRPO training and inference (`--ram_path` / `--dape_path`). A vendored copy is included under `RefReward-SR/recognize-anything/`:

```bash
pip install -e RefReward-SR/recognize-anything/
```

Download the pretrained checkpoints (used for auto-captioning during SR-GRPO training and inference):
- [RAM](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth)
- [DAPE](https://drive.google.com/drive/folders/12HXrRGEXUAnmHRaf0bIn-S8XSK4Ku0JO?usp=drive_link) (degradation-aware prompt extractor)

### Optional: GroundingDINO (for Local Crop Evaluation)

Only needed if you want to use `--auto-detect-boxes` for local crop scoring during RefReward evaluation. A vendored copy is included under `RefReward-SR/GroundingDINO/`:

```bash
pip install fairscale
pip install --no-build-isolation -e RefReward-SR/GroundingDINO
```

Download the pretrained checkpoint:
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)

---

## Part 1: RefReward-SR — LR-Conditioned Reward Model

### 1.1 Overview

RefReward-SR trains a Qwen3-VL model to score super-resolution image quality via GRPO. Given a low-resolution (LR) reference image and a super-resolved (HR) image, the model outputs a quality score. It supports two inference modes:
- **fast** — direct scoring without chain-of-thought
- **think** — reasoning + scoring with chain-of-thought

### 1.2 Required Downloads

| Resource | Source | Size | Usage |
|----------|--------|------|-------|
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Hugging Face | - | Base model for GRPO training |
| RefSR-18K `train_images/` | [casyssong/RefSR-18K](https://huggingface.co/datasets/casyssong/RefSR-18K) | 12G | 4499 groups (0001-4499), used for model training |
| RefSR-18K `test_in_domain_images/` | [casyssong/RefSR-18K](https://huggingface.co/datasets/casyssong/RefSR-18K) | 527M | 200 in-domain benchmark groups with LR/HR image pairs from multiple SR methods |
| RefSR-18K `test_out_domain_images/` | [casyssong/RefSR-18K](https://huggingface.co/datasets/casyssong/RefSR-18K) | 532M | 200 out-domain benchmark groups with LR/HR image pairs from multiple SR methods |
| RefSR-18K `train_extra_images/` (supplementary) | [casyssong/RefSR-18K](https://huggingface.co/datasets/casyssong/RefSR-18K) | 9.1G | 3562 groups (4500-8061), additional LSDIR-based groups annotated by us; **not used in model training**, provided for future research |

Each image group in RefSR-18K contains `LR.png` (low-resolution input) and 4 SR outputs from different methods (`HR_A.png`, `HR_B.png`, `HR_C.png`, `HR_D.png`), with human-annotated pairwise rankings based on LR–HR consistency and HR naturalness.

After downloading the `casyssong/RefSR-18K` dataset repository from Hugging Face, place its contents under a local dataset root. The example paths in the training and evaluation scripts use:

```
../RefReward-SR-dataset/
├── train_images/
├── train_extra_images/
├── test_in_domain_images/
└── test_out_domain_images/
```

The following annotation files are already included in this repository under `RefReward-SR/dataset/` (they are also mirrored in the Hugging Face dataset):
- `train.csv` — annotations for `train_images/` (4499 groups)
- `train_extra.csv` — annotations for `train_extra_images/` (3562 groups)
- `train_sr.jsonl` — generated by `make_data_sr.py` from `train.csv` + `train_images/`
- `test_in_domain.csv` / `test_out_domain.csv` — test set ground-truth rankings
- `raw-in-domain/` / `raw-out-domain/` — per-annotator rankings for agreement evaluation

### 1.3 Data Preparation

If you need to regenerate `train_sr.jsonl` (e.g., after updating training images or annotations):

```bash
# Run from the repo root (not from RefReward-SR/)
# Edit TRAIN_IMAGES_ROOT in make_data_sr.py to point to your downloaded training images
python RefReward-SR/dataset/make_data_sr.py
```

### 1.4 Training

```bash
cd RefReward-SR
# Edit paths in the script: MODEL_PATH, IMAGE_FOLDERS, DATA_FILE
bash scripts/train_refreward_qwen3vl.sh
```

Key training parameters:
- 8 GPUs, DeepSpeed ZeRO-3
- `num_generations=6`, `max_steps=1200`
- `flash_attention_2`, `bf16`
- Checkpoints saved every 50 steps

### 1.5 Evaluation

Download the released [`casyssong/RefReward-SR`](https://huggingface.co/casyssong/RefReward-SR) repository before evaluation. In the commands below, `/path/to/checkpoint` should point to the downloaded `checkpoint-1200` model directory.

> **Note on Global vs. Global+Local scoring:**
> The global+local scoring mode provides finer-grained evaluation by additionally scoring object-level sub-regions detected via RAM + GroundingDINO. However, **global-only scoring is sufficient in most cases**. If ground-truth HR images are not available for box detection, the SR output itself can be used as the input for detecting local regions.

**Single image pair:**

```bash
cd RefReward-SR
CUDA_VISIBLE_DEVICES=0 python test_single.py /path/to/checkpoint \
    --lr /path/to/LR.png --hr /path/to/HR.png --mode think
```

**Full benchmark:**

`test_benchmark.py` supports the following options:

| Option | Values | Description |
|--------|--------|-------------|
| `--mode` | `fast` / `think` | `fast`: direct scoring; `think`: chain-of-thought reasoning + scoring |
| `--dataset` | `in` / `out` | `in`: in-domain test set (200 groups); `out`: out-domain test set (200 groups) |
| `--local-crop` | flag | Enable local crop scoring (score sub-regions in addition to full image) |
| `--auto-detect-boxes` | flag | Auto-detect crop regions via RAM + GroundingDINO (requires `--grounding-checkpoint` and `--ram-checkpoint`) |

```bash
cd RefReward-SR

# Global scoring, fast mode, in-domain
python test_benchmark.py /path/to/checkpoint --mode fast --dataset in

# Global scoring, think mode, out-domain
python test_benchmark.py /path/to/checkpoint --mode think --dataset out

# Global + local crop scoring, fast mode, in-domain (requires GroundingDINO + RAM)
python test_benchmark.py /path/to/checkpoint --mode fast --dataset in \
    --local-crop --auto-detect-boxes \
    --grounding-checkpoint /path/to/groundingdino_swint_ogc.pth \
    --ram-checkpoint /path/to/ram_swin_large_14m.pth

# Global + local crop scoring, think mode, out-domain
python test_benchmark.py /path/to/checkpoint --mode think --dataset out \
    --local-crop --auto-detect-boxes \
    --grounding-checkpoint /path/to/groundingdino_swint_ogc.pth \
    --ram-checkpoint /path/to/ram_swin_large_14m.pth
```

---

## Part 2: SR-GRPO — Preference-Aligned SR Fine-tuning

### 2.1 Overview

SR-GRPO integrates RefReward-SR into the GRPO training loop to fine-tune a Flux ControlNet SR model with LoRA for preference-aligned generation. Three complementary reward signals guide the optimization:

| Reward | Weight | Description |
|--------|--------|-------------|
| **RefReward-SR** | 1.0 | LR-conditioned semantic reward — evaluates semantic consistency and perceptual plausibility (from Part 1) |
| **DeQA-Score** | 1.0 | Low-level image quality assessment |
| **LPIPS** | 1.0 | Perceptual similarity to ground-truth |

### 2.2 Required Downloads

| Resource | Source | Usage |
|----------|--------|-------|
| [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Hugging Face | Base diffusion model |
| [ControlNet SR checkpoint (C-FLUX)](https://drive.google.com/drive/folders/1LVSwWbjJdn5Wxy79AwV5LdPX2S2Hotx1?usp=sharing) | Google Drive (from [DP2O-SR](https://github.com/cswry/DP2O-SR)) | ControlNet weights for SR |
| [DeQA-Score](https://github.com/zhiyuanyou/DeQA-Score) | GitHub | DeQA reward model + preprocessor |
| RefReward checkpoint | [casyssong/RefReward-SR](https://huggingface.co/casyssong/RefReward-SR) (`checkpoint-1200`), or from Part 1 training | Qwen3-VL reward model |
| [SR-GRPO-LoRA](https://huggingface.co/casyssong/SR-GRPO-LoRA) | Hugging Face | Released LoRA adapter for inference |
| RAM + DAPE | See [Environment Setup](#ram--dape) | Auto-captioning for SR training and inference |

**Training data:** Download [LSDIR](https://github.com/ofsoundof/LSDIR) and [FFHQ](https://github.com/NVlabs/ffhq-dataset) (first 10k images), then follow [SeeSR](https://github.com/cswry/SeeSR)'s data preparation pipeline to generate degraded LR/HR pairs (`gt/`, `sr_bicubic/`) and tag prompts (`tag/`).

**Pre-computed text embeddings** (CLIP + T5 for FLUX) are generated from the `tag/` prompts using:

```bash
cd SR-GRPO
python fastvideo/data_preprocess/preprocess_flux_sr_embedding.py \
    --data_root /path/to/training_for_seesr \
    --output_dir /path/to/sr_embeddings \
    --model_path /path/to/FLUX.1-dev
```

This produces `videos2caption.json` and the `.pt` embedding files (`prompt_embed/`, `pooled_prompt_embeds/`, `text_ids/`) required by the training script.

### 2.3 Training

```bash
cd SR-GRPO
# Edit all /path/to/... placeholders in the script
bash scripts/train_dancegrpo_sr.sh
```

### 2.4 Inference

Run SR inference with a trained LoRA checkpoint. To use the released weights, download the separate [`casyssong/SR-GRPO-LoRA`](https://huggingface.co/casyssong/SR-GRPO-LoRA) repository and pass its local directory as `--lora_checkpoint_dir`:

```bash
cd SR-GRPO
python test_flux_controlnet_sr_lora.py \
    --pretrained_model_name_or_path /path/to/FLUX.1-dev \
    --controlnet_model_name_or_path /path/to/controlnet/model.safetensors \
    --lora_checkpoint_dir /path/to/lora-checkpoint \
    --image_path /path/to/input_LR_images \
    --output_dir /path/to/output \
    --ram_path /path/to/ram_swin_large_14m.pth \
    --dape_path /path/to/DAPE.pth \
    --seed 42
```

### 2.5 Image Quality Evaluation

`test_metrics_full.py` computes a comprehensive set of image quality metrics between SR outputs and ground-truth images, including both full-reference (FR) and no-reference (NR) metrics, as well as FID.

**Additional dependencies:**

```bash
pip install pyiqa==0.1.15.post2 basicsr==1.4.2
```

> **Note:** If you encounter `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'` with newer versions of torchvision, patch the basicsr source:
> ```bash
> # In /path/to/site-packages/basicsr/data/degradations.py, change:
> #   from torchvision.transforms.functional_tensor import rgb_to_grayscale
> # to:
> #   from torchvision.transforms.functional import rgb_to_grayscale
> ```

**Supported metrics:**

| Category | Metrics |
|----------|---------|
| Full-Reference (FR) | PSNR, SSIM, LPIPS, TOPIQ-FR, AFINE-FR, FID |
| No-Reference (NR) | MANIQA, MUSIQ, CLIPIQA, CLIPIQA+, TOPIQ-NR, AFINE-NR, QALIGN, NIMA, TOPIQ-IAA |

**Usage:**

```bash
cd SR-GRPO

python test_metrics_full.py \
    --inp_imgs /path/to/SR_output_images \
    --gt_imgs /path/to/GT_images \
    --log /path/to/save_logs
```

Results (per-image and average metrics) are printed to the console and saved to timestamped log files under the `--log` directory.

---

## License

Except where otherwise noted, original source code in this repository is licensed under the [Apache License 2.0](LICENSE). Third-party components remain subject to their respective licenses.

**Released weights and dataset:** For license and usage information regarding [RefReward-SR](https://huggingface.co/casyssong/RefReward-SR), [SR-GRPO-LoRA](https://huggingface.co/casyssong/SR-GRPO-LoRA), and [RefSR-18K](https://huggingface.co/datasets/casyssong/RefSR-18K), please refer to their respective Hugging Face model and dataset cards.

**FFHQ:** Used in SR-GRPO training data. The dataset is under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); individual images may carry CC BY-NC or other Flickr licenses — **non-commercial** use only. See [FFHQ LICENSE](https://github.com/NVlabs/ffhq-dataset/blob/master/LICENSE.txt).

---

## Acknowledgements

- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (Black Forest Labs)
- [LSDIR](https://github.com/ofsoundof/LSDIR)
- [DP2O-SR](https://github.com/cswry/DP2O-SR)
- [SeeSR](https://github.com/cswry/SeeSR)
- [DeQA-Score](https://github.com/zhiyuanyou/DeQA-Score)
- [VisualQuality-R1](https://github.com/TianheWu/VisualQuality-R1)
- [Open-R1](https://github.com/huggingface/open-r1)
- [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

## Citation

```bibtex
@article{song2026refreward,
  title={RefReward-SR: LR-Conditioned Reward Modeling for Preference-Aligned Super-Resolution},
  author={Song, Yushuai and Quan, Weize and Wang, Weining and Sun, Jiahui and Liu, Jing and Li, Meng and Yu, Pengbin and Chen, Zhentao and Shen, Wei and Yuan, Lunxi and Yan, Dong-ming},
  journal={arXiv preprint arXiv:2603.24198},
  year={2026}
}
```
