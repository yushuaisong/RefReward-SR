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

import os
import sys
import argparse
import json
import re
import time

from PIL import Image
import numpy as np
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MAX_NEW_TOKENS = 512
MAX_RETRY = 5

LOCAL_MAX_BOXES = 4
LOCAL_SELECT_K = 4

PROMPT_TEXT = (
    "You are a highly discerning expert in image quality assessment."
    "Your task is to evaluate a super-resolution (SR) output by comparing it with the original low-resolution reference."
    "You will be given two images: the first (LR) shows the original scene, and the second (HR) is the super-resolution result."
    "Assume the HR image may contain problematic regions."
    "Carefully inspect both images and identify any areas in the HR image that look unnatural or distorted "
    "(e.g., distorted regions or warped lines, unrealistic textures, unreasonable objects) "
    "or semantically inconsistent with the LR image (e.g., missing, added or altered objects, inconsistent texture)."
    "For each problematic region, state its approximate location in the HR image (e.g., 'top left', 'center lower') "
    "and briefly explain why it appears incorrect."
    "After inspection, assign an overall quality score from 1.00 to 5.00, with two decimal places (e.g., 1.31, 2.77, 4.53)."
    "First output your reasoning in <thinking>…</thinking> tags (a brief summary of observed defects and semantic issues)."
    "Then output only one numeric score in <answer>…</answer> tags (no extra text)."
)


# =========================
# Model loading
# =========================
def load_model(model_path):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor


# =========================
# Score parsing
# =========================
_answer_re = re.compile(r"<answer>\s*([0-9]+(?:\.[0-9]+)?)\s*</answer>", re.IGNORECASE)


class AnswerTagParseError(ValueError):
    pass


def parse_score(text: str) -> float:
    m = _answer_re.search(text)
    if not m:
        raise AnswerTagParseError(f"Cannot parse <answer> score from output: {text}")
    return float(m.group(1))


# =========================
# Inference (supports think / fast mode)
# =========================
def _score_with_inputs(lr_input, hr_input, model, processor, mode: str) -> tuple:
    """Run inference and return (score, response_text)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": lr_input},
                {"type": "image", "image": hr_input},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    if mode == "fast":
        text += "<thinking>…</thinking><answer>"

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, image_patch_size=16, return_video_metadata=True,
    )
    if video_kwargs is None:
        video_kwargs = {}
    else:
        video_kwargs = dict(video_kwargs)
    video_metadatas = None
    if video_inputs is not None and len(video_inputs) > 0 and isinstance(video_inputs[0], tuple):
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        **video_kwargs,
        do_resize=False,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    last_error = None
    for attempt in range(1, MAX_RETRY + 1):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(attempt > 1),
            temperature=0.7 if attempt > 1 else 1.0,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        out_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

        try:
            if mode == "fast":
                score = parse_score("<answer>" + out_text)
            else:
                score = parse_score(out_text)
            if attempt > 1:
                print(f"[RETRY] Attempt {attempt}/{MAX_RETRY} succeeded.")
            return score, out_text
        except AnswerTagParseError as e:
            last_error = e
            print(f"[RETRY] Attempt {attempt}/{MAX_RETRY} failed to parse score: {str(e)[:120]}")

    raise last_error


# =========================
# Local crop utilities
# =========================
def _bbox_iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1 + 1) * (y2 - y1 + 1)
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _select_low_overlap_boxes(boxes: list[dict], k: int = LOCAL_SELECT_K) -> list[dict]:
    if len(boxes) <= k:
        return boxes
    bboxes = [b["bbox"] for b in boxes]
    n = len(bboxes)
    mean_iou = []
    for i in range(n):
        ious = [_bbox_iou(bboxes[i], bboxes[j]) for j in range(n) if j != i]
        mean_iou.append(sum(ious) / len(ious) if ious else 0.0)
    idxs = np.argsort(mean_iou)[:k]
    return [boxes[i] for i in idxs]


def _scale_bbox(bbox, scale_x: float, scale_y: float, max_w: int, max_h: int):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1 * scale_x)), max_w - 1))
    y1 = max(0, min(int(round(y1 * scale_y)), max_h - 1))
    x2 = max(0, min(int(round(x2 * scale_x)), max_w - 1))
    y2 = max(0, min(int(round(y2 * scale_y)), max_h - 1))
    return [x1, y1, x2, y2]


def _crop_with_bbox(img: Image.Image, bbox) -> Image.Image:
    x1, y1, x2, y2 = bbox
    return img.crop((x1, y1, x2 + 1, y2 + 1))


def _load_boxes_json(json_path: str) -> dict | None:
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    if "orig_size" not in record:
        image_path = record.get("image")
        if image_path and os.path.exists(image_path):
            with Image.open(image_path) as im:
                w, h = im.size
            record["orig_size"] = [w, h]
    return record


def _compute_local_scores(
    lr_path: str, hr_path: str, box_record: dict | None, model, processor, mode: str,
):
    """Compute area-weighted local score from selected boxes.
    Returns (weighted_score, box_count, total_local_area, global_area).
    """
    if box_record is None:
        return None, 0, 0.0, 0.0
    boxes = box_record.get("boxes", [])
    if not boxes:
        return None, 0, 0.0, 0.0

    if len(boxes) > LOCAL_MAX_BOXES:
        boxes = _select_low_overlap_boxes(boxes, k=LOCAL_SELECT_K)

    hr_img = Image.open(hr_path).convert("RGB")
    lr_img = Image.open(lr_path).convert("RGB")
    hr_w, hr_h = hr_img.size
    lr_w, lr_h = lr_img.size

    ref_w, ref_h = box_record.get("orig_size", [hr_w, hr_h])
    scale_x_hr = hr_w / float(ref_w)
    scale_y_hr = hr_h / float(ref_h)
    scale_x_lr = lr_w / float(ref_w)
    scale_y_lr = lr_h / float(ref_h)

    local_scores = []
    local_areas = []
    for b in boxes:
        bbox_ref = b["bbox"]
        bbox_hr = _scale_bbox(bbox_ref, scale_x_hr, scale_y_hr, hr_w, hr_h)
        bbox_lr = _scale_bbox(bbox_ref, scale_x_lr, scale_y_lr, lr_w, lr_h)
        hr_crop = _crop_with_bbox(hr_img, bbox_hr)
        lr_crop = _crop_with_bbox(lr_img, bbox_lr)

        score, _ = _score_with_inputs(lr_crop, hr_crop, model, processor, mode)
        area = float((bbox_hr[2] - bbox_hr[0] + 1) * (bbox_hr[3] - bbox_hr[1] + 1))
        local_scores.append(score)
        local_areas.append(area)
        print(f"    [LOCAL] bbox={bbox_ref} -> score={score:.2f}, area={area:.0f}")

    if not local_scores:
        return None, len(boxes), 0.0, float(hr_w * hr_h)
    total_area = float(sum(local_areas))
    weighted = sum(s * a for s, a in zip(local_scores, local_areas)) / max(total_area, 1.0)
    return weighted, len(boxes), total_area, float(hr_w * hr_h)


# =========================
# Main
# =========================
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="RefReward-SR Single Image Pair Test (think/fast, with optional local crop)")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--lr", type=str, required=True, help="Path to LR (low-resolution) image")
    parser.add_argument("--hr", type=str, required=True, help="Path to HR (super-resolution) image")
    parser.add_argument("--mode", type=str, choices=["think", "fast"], required=True,
                        help="Inference mode: 'think' (full reasoning) or 'fast' (skip thinking)")
    parser.add_argument("--local-crop", action="store_true",
                        help="Enable local crop scoring (requires --boxes-json or --auto-detect-boxes)")
    parser.add_argument("--boxes-json", type=str, default=None,
                        help="Path to a *_boxes_filtered.json file for local crop scoring")
    parser.add_argument("--auto-detect-boxes", action="store_true",
                        help="Automatically detect boxes using RAM + GroundingDINO (no pre-computed JSON needed)")
    parser.add_argument("--grounding-checkpoint", type=str, default=None,
                        help="Path to GroundingDINO checkpoint (groundingdino_swint_ogc.pth)")
    parser.add_argument("--ram-checkpoint", type=str, default=None,
                        help="Path to RAM checkpoint (ram_swin_large_14m.pth)")
    parser.add_argument("--grounding-config", type=str, default=None,
                        help="Path to GroundingDINO config (auto-detected if not provided)")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint
    lr_path = args.lr
    hr_path = args.hr
    mode = args.mode
    use_local_crop = args.local_crop
    boxes_json = args.boxes_json
    auto_detect = args.auto_detect_boxes

    if not os.path.isdir(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    if not os.path.isfile(lr_path):
        print(f"Error: LR image not found: {lr_path}")
        sys.exit(1)
    if not os.path.isfile(hr_path):
        print(f"Error: HR image not found: {hr_path}")
        sys.exit(1)
    if use_local_crop and not boxes_json and not auto_detect:
        print("Error: --local-crop requires --boxes-json or --auto-detect-boxes")
        sys.exit(1)
    if boxes_json and not os.path.isfile(boxes_json):
        print(f"Error: Boxes JSON not found: {boxes_json}")
        sys.exit(1)
    if auto_detect and (not args.grounding_checkpoint or not args.ram_checkpoint):
        print("Error: --auto-detect-boxes requires --grounding-checkpoint and --ram-checkpoint")
        sys.exit(1)

    # --- Auto-detect boxes (before loading the scoring model to save GPU memory) ---
    box_record = None
    if use_local_crop and auto_detect:
        from auto_box_detector import AutoBoxDetector
        print(f"\n[AUTO-DETECT] Loading RAM + GroundingDINO for box detection...")
        detector = AutoBoxDetector(
            grounding_checkpoint=args.grounding_checkpoint,
            ram_checkpoint=args.ram_checkpoint,
            grounding_config=args.grounding_config,
        )
        print(f"[AUTO-DETECT] Detecting boxes on HR image: {hr_path}")
        box_record = detector.detect(hr_path)
        box_count_detected = len(box_record.get("boxes", []))
        print(f"[AUTO-DETECT] Detected {box_count_detected} filtered boxes")
        detector.cleanup()
        del detector
        torch.cuda.empty_cache()
        print(f"[AUTO-DETECT] Detector released, GPU memory freed\n")
    elif use_local_crop and boxes_json:
        box_record = _load_boxes_json(boxes_json)

    print(f"{'=' * 60}")
    print(f"  Checkpoint  : {checkpoint_dir}")
    print(f"  LR image    : {lr_path}")
    print(f"  HR image    : {hr_path}")
    print(f"  Mode        : {mode}")
    print(f"  Local crop  : {use_local_crop}")
    if use_local_crop:
        if auto_detect:
            print(f"  Box source  : auto-detect (RAM + GroundingDINO)")
        else:
            print(f"  Boxes JSON  : {boxes_json}")
    print(f"{'=' * 60}")

    lr_img = Image.open(lr_path)
    hr_img = Image.open(hr_path)
    print(f"  LR size     : {lr_img.size[0]} x {lr_img.size[1]}")
    print(f"  HR size     : {hr_img.size[0]} x {hr_img.size[1]}")
    lr_img.close()
    hr_img.close()

    print(f"\n[LOADING] Loading model...")
    model, processor = load_model(checkpoint_dir)
    print("[LOADING] Model loaded successfully\n")

    t0 = time.time()
    try:
        # Global score
        global_score, response_text = _score_with_inputs(lr_path, hr_path, model, processor, mode)
        print(f"[GLOBAL] Score: {global_score:.2f}")

        # Local crop score
        final_score = global_score
        local_weighted = None
        box_count = 0
        if use_local_crop:
            if box_record is None:
                box_record = _load_boxes_json(boxes_json)
            box_count = len(box_record.get("boxes", [])) if box_record else 0
            print(f"\n[LOCAL] Using {box_count} boxes")

            local_weighted, used_count, local_area, global_area = _compute_local_scores(
                lr_path, hr_path, box_record, model, processor, mode,
            )
            if local_weighted is not None:
                final_score = (
                    global_score * global_area + local_weighted * local_area
                ) / max(global_area + local_area, 1.0)
                print(f"[LOCAL] Weighted local score: {local_weighted:.2f} (from {used_count} boxes)")
                print(f"[LOCAL] local_area={local_area:.0f}, global_area={global_area:.0f}")
            else:
                print("[LOCAL] No valid boxes, using global score only")

        elapsed = time.time() - t0

        print(f"\n{'=' * 60}")
        print(f"  RESULT")
        print(f"{'=' * 60}")
        print(f"  Global score : {global_score:.2f}")
        if use_local_crop:
            print(f"  Local score  : {local_weighted:.2f}" if local_weighted is not None else "  Local score  : N/A")
            print(f"  Box count    : {box_count}")
        print(f"  Final score  : {final_score:.2f}")
        print(f"  Time         : {elapsed:.1f}s")
        print(f"{'=' * 60}")

        if mode == "think":
            print(f"\n[Full Response (thinking process)]:")
            print(f"{'-' * 60}")
            print(response_text)
            print(f"{'-' * 60}")
        else:
            print(f"\n[Response]: {response_text.strip()}")

    except AnswerTagParseError as e:
        elapsed = time.time() - t0
        print(f"\n[ERROR] Failed to parse score after {MAX_RETRY} attempts ({elapsed:.1f}s): {repr(e)}")
        sys.exit(1)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n[ERROR] Inference failed ({elapsed:.1f}s): {repr(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        del model, processor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
