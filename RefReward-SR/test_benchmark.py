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
import time
import re
import itertools
from contextlib import redirect_stdout
from io import StringIO
from typing import TextIO, cast

from PIL import Image
import pandas as pd
import numpy as np
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def _progress(msg: str):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# =========================
# Dataset paths
# =========================
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
CSV_IN_DOMAIN = os.path.join(DATASET_DIR, "test_in_domain.csv")
CSV_OUT_DOMAIN = os.path.join(DATASET_DIR, "test_out_domain.csv")
IMAGE_ROOT_IN = "/path/to/RefReward-SR-dataset/test_in_domain_images"
IMAGE_ROOT_OUT = "/path/to/RefReward-SR-dataset/test_out_domain_images"
RAW_IN_DOMAIN_DIR = os.path.join(DATASET_DIR, "raw-in-domain")
RAW_OUT_DOMAIN_DIR = os.path.join(DATASET_DIR, "raw-out-domain")

MAX_NEW_TOKENS = 512
MAX_RETRY = 5

# Local crop defaults
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


class TeeStdout:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def parse_score(text: str) -> float:
    m = _answer_re.search(text)
    if not m:
        raise AnswerTagParseError(f"Cannot parse <answer> score from output: {text}")
    return float(m.group(1))


# =========================
# Inference (supports think / fast mode)
# =========================
def _score_with_inputs(lr_input, hr_input, model, processor, mode: str) -> tuple:
    """Run inference and return (score, response_text).
    mode='fast':  skip thinking, directly output answer.
    mode='think': let model generate full thinking + answer.
    """
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
                _progress(f"[RETRY]  Attempt {attempt}/{MAX_RETRY} succeeded.")
            return score, out_text
        except AnswerTagParseError as e:
            last_error = e
            _progress(f"[RETRY]  Attempt {attempt}/{MAX_RETRY} failed to parse score: {str(e)[:120]}")

    raise last_error


@torch.no_grad()
def eval_hr_score(lr_path: str, hr_path: str, model, processor, mode: str) -> tuple:
    return _score_with_inputs(lr_path, hr_path, model, processor, mode)


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


def _load_boxes(subdir: str, boxes_dir: str) -> dict | None:
    path = os.path.join(boxes_dir, f"{subdir}_boxes_filtered.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        record = json.load(f)
    if "orig_size" not in record:
        image_path = record.get("image")
        if image_path and os.path.exists(image_path):
            with Image.open(image_path) as im:
                w, h = im.size
            record["orig_size"] = [w, h]
    return record


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

    if not local_scores:
        return None, len(boxes), 0.0, float(hr_w * hr_h)
    total_area = float(sum(local_areas))
    weighted = sum(s * a for s, a in zip(local_scores, local_areas)) / max(total_area, 1.0)
    return weighted, len(boxes), total_area, float(hr_w * hr_h)


# =========================
# Ranking utilities
# =========================
def scores_to_midranks(scores: list[float], tie_eps: float = 0.0) -> np.ndarray:
    scores_arr = np.array(scores, dtype=float)
    order = np.argsort(-scores_arr)
    ranks = np.zeros_like(scores_arr, dtype=float)
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j < n:
            if abs(scores_arr[order[j]] - scores_arr[order[i]]) <= tie_eps:
                j += 1
            else:
                break
        mid = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = mid
        i = j
    return ranks


# =========================
# Agreement computation
# =========================
def get_pairwise_orders(row: pd.Series, prefix: str = 'Total'):
    imgs = []
    for i in [1, 2, 3, 4]:
        img = row.get(f'Image{i}')
        score = row.get(f'{prefix}{i}')
        if bool(pd.notna(img)) and bool(pd.notna(score)):
            imgs.append((img, score))
    orders = {}
    for (a, sa), (b, sb) in itertools.combinations(imgs, 2):
        if sa > sb:
            orders[(a, b)] = 1
            orders[(b, a)] = -1
        elif sa < sb:
            orders[(a, b)] = -1
            orders[(b, a)] = 1
        else:
            orders[(a, b)] = 0
            orders[(b, a)] = 0
    return orders


def compute_agreement_anyN(dfs, prefix='Total'):
    n = len(dfs)
    if n < 2:
        raise ValueError("dfs must contain at least 2 DataFrames")
    pairs_ij = list(itertools.combinations(range(n), 2))
    total = 0
    agree_allN = 0
    pairwise = {ij: 0 for ij in pairs_ij}
    pairwise_total = {ij: 0 for ij in pairs_ij}
    common_index = dfs[0].index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)
    for idx in common_index:
        orders = [get_pairwise_orders(df.loc[idx], prefix) for df in dfs]
        all_pairs = set().union(*[o.keys() for o in orders])
        for pair in all_pairs:
            votes = [o.get(pair, None) for o in orders]
            if any(v is None for v in votes):
                continue
            total += 1
            if len(set(votes)) == 1:
                agree_allN += 1
            for ij in pairs_ij:
                i, j = ij
                pairwise_total[ij] += 1
                if votes[i] == votes[j]:
                    pairwise[ij] += 1
    res = {
        "total_pairs": total,
        "allN_rate": (agree_allN / total) if total else None,
    }
    for (i, j) in pairs_ij:
        res[f"agree_{i}_{j}"] = (
            pairwise[(i, j)] / pairwise_total[(i, j)]
            if pairwise_total[(i, j)] else None
        )
    return res


def compute_agreement_per_id(df_a: pd.DataFrame, df_b: pd.DataFrame, idx, prefix="Total"):
    ra = df_a.loc[idx]
    rb = df_b.loc[idx]
    oa = get_pairwise_orders(ra, prefix)
    ob = get_pairwise_orders(rb, prefix)
    common_pairs = set(oa.keys()) & set(ob.keys())
    total = 0
    agree = 0
    for pair in common_pairs:
        total += 1
        if oa[pair] == ob[pair]:
            agree += 1
    return (agree / total if total else None), total, agree


# =========================
# Load GT from CSV
# =========================
def load_gt_data(csv_path: str) -> pd.DataFrame:
    gt = pd.read_csv(csv_path)
    gt["Subdirectory"] = gt["Subdirectory"].apply(lambda x: str(int(x)).zfill(4))
    gt = gt.set_index("Subdirectory")
    gt_rank = pd.DataFrame(index=gt.index)
    for i in range(1, 5):
        gt_rank[f"Image{i}"] = gt[f"Image{i}"]
        gt_rank[f"Total{i}"] = gt[f"FinalRank{i}"].astype(float)
    return gt_rank


def load_raw_annotator_rankings(raw_dir: str) -> list[pd.DataFrame] | None:
    """Load rankings1/2/3.csv from raw annotator directory.
    Returns list of DataFrames indexed by Subdirectory (new_id str), or None if unavailable.
    """
    if not os.path.isdir(raw_dir):
        return None
    dfs = []
    for fname in ["rankings1.csv", "rankings2.csv", "rankings3.csv"]:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        df["Subdirectory"] = df["Subdirectory"].apply(lambda x: str(int(x)).zfill(4))
        df = df.set_index("Subdirectory")
        dfs.append(df)
    return dfs


# =========================
# Evaluate one dataset split
# =========================
def evaluate_split(
    model, processor, mode: str, csv_path: str, image_root: str,
    split_name: str, output_dir: str,
    use_local_crop: bool = False, boxes_dir: str | None = None,
    raw_rankings_dir: str | None = None,
):
    output_buffer = StringIO()
    tee = TeeStdout(sys.stdout, output_buffer)

    with redirect_stdout(cast(TextIO, tee)):
        print(f"\n{'=' * 80}")
        print(f"Evaluating split: {split_name}  |  mode: {mode}  |  local_crop: {use_local_crop}")
        print(f"CSV: {csv_path}")
        print(f"Image root: {image_root}")
        if use_local_crop:
            print(f"Boxes dir: {boxes_dir}")
        print(f"{'=' * 80}")

        gt_rank = load_gt_data(csv_path)

        pred_rows = []
        global_pred_rows = []
        stats_rows = []
        response_records = []
        skipped_parse_ids = []
        box_gt0_subdirs = []

        total_ids = len(gt_rank.index)
        done = 0
        skipped = 0
        failed = 0
        t0 = time.time()
        _progress(f"[INFO] {split_name}: Total samples: {total_ids}")

        for idx_i, subdir in enumerate(gt_rank.index, start=1):
            t_id0 = time.time()
            print(f"\n[START] ({idx_i}/{total_ids}) Subdirectory={subdir}")

            folder = os.path.join(image_root, subdir)
            lr_path = os.path.join(folder, "LR.png")

            if not os.path.exists(lr_path):
                skipped += 1
                print(f"[SKIP]  ({idx_i}/{total_ids}) LR not found: {lr_path}")
                continue

            images = [gt_rank.loc[subdir, f"Image{i}"] for i in range(1, 5)]
            hr_paths = [os.path.join(folder, img) for img in images]

            miss = [p for p in hr_paths if not os.path.exists(p)]
            if miss:
                skipped += 1
                print(f"[SKIP]  ({idx_i}/{total_ids}) HR missing for {subdir}: {miss}")
                continue

            try:
                box_record = None
                box_count = 0
                if use_local_crop and boxes_dir:
                    box_record = _load_boxes(subdir, boxes_dir)
                    box_count = len(box_record.get("boxes", [])) if box_record else 0

                scores = []
                global_scores = []
                local_scores = []
                responses = []
                for p in hr_paths:
                    global_score, response_text = eval_hr_score(lr_path, p, model, processor, mode)
                    global_scores.append(global_score)
                    responses.append(response_text)

                    if use_local_crop:
                        local_score, used_count, local_area, global_area = _compute_local_scores(
                            lr_path, p, box_record, model, processor, mode,
                        )
                        local_scores.append(local_score)
                        if local_score is not None:
                            final_score = (
                                global_score * global_area + local_score * local_area
                            ) / max(global_area + local_area, 1.0)
                        else:
                            final_score = global_score
                        scores.append(final_score)
                    else:
                        local_scores.append(None)
                        scores.append(global_score)

                for img_i, (img_name, resp) in enumerate(zip(images, responses)):
                    response_records.append({
                        "subdir": subdir, "image": img_name,
                        "score": scores[img_i], "response": resp,
                    })

                print("[SCORES] " + " | ".join([f"{images[i]}: {scores[i]:.2f}" for i in range(4)]))

                pred_ranks = scores_to_midranks(scores, tie_eps=0.0).tolist()
                global_ranks = scores_to_midranks(global_scores, tie_eps=0.0).tolist()
                print("[PRED_RANK] " + " | ".join([f"{images[i]}: {pred_ranks[i]:.2f}" for i in range(4)]))
                print("[GT_RANK]   " + " | ".join(
                    [f"{images[i]}: {float(gt_rank.loc[subdir, f'Total{i + 1}']):.2f}" for i in range(4)]
                ))

                pred_row = {"Subdirectory": subdir}
                for i in range(1, 5):
                    pred_row[f"Image{i}"] = images[i - 1]
                    pred_row[f"Total{i}"] = pred_ranks[i - 1]
                pred_rows.append(pred_row)

                global_row = {"Subdirectory": subdir}
                for i in range(1, 5):
                    global_row[f"Image{i}"] = images[i - 1]
                    global_row[f"Total{i}"] = global_ranks[i - 1]
                global_pred_rows.append(global_row)

                if box_count > 0:
                    box_gt0_subdirs.append(subdir)

                pred_row_df = pd.DataFrame([pred_row]).set_index("Subdirectory")
                agree_id, total_pairs_id, agree_pairs_id = compute_agreement_per_id(
                    pred_row_df, gt_rank, subdir, prefix="Total"
                )
                print(f"[AGREE_ID] Subdirectory={subdir} agree={agree_id} ({agree_pairs_id}/{total_pairs_id})")

                stats_row = {
                    "Subdirectory": subdir,
                    **{f"Image{i+1}": images[i] for i in range(4)},
                    **{f"score{i+1}": scores[i] for i in range(4)},
                    **{f"global_score{i+1}": global_scores[i] for i in range(4)},
                    **{f"pred_rank{i+1}": pred_ranks[i] for i in range(4)},
                    **{f"gt_rank{i+1}": float(gt_rank.loc[subdir, f"Total{i+1}"]) for i in range(4)},
                    "box_count": box_count,
                    **{f"local_score{i+1}": local_scores[i] for i in range(4)},
                    "agreement_id": agree_id,
                    "agreement_id_pairs": total_pairs_id,
                    "agreement_id_agree_pairs": agree_pairs_id,
                }
                stats_rows.append(stats_row)

                done += 1
                dt = time.time() - t_id0
                elapsed = time.time() - t0
                avg_per_sample = elapsed / done if done else 0
                eta = avg_per_sample * (total_ids - idx_i)
                print(
                    f"[DONE]   ({idx_i}/{total_ids}) Subdirectory={subdir} "
                    f"time={dt:.1f}s | done={done} skip={skipped} fail={failed} | "
                    f"elapsed={elapsed / 60:.1f}min | ETA={eta / 60:.1f}min"
                )
                _progress(
                    f"[PROGRESS] {split_name} {idx_i}/{total_ids} ({100 * idx_i / total_ids:.1f}%) | "
                    f"done={done} skip={skipped} fail={failed} | "
                    f"this={dt:.1f}s avg={avg_per_sample:.1f}s | "
                    f"elapsed={elapsed / 60:.1f}min ETA={eta / 60:.1f}min"
                )

            except AnswerTagParseError as e:
                skipped += 1
                skipped_parse_ids.append(subdir)
                dt = time.time() - t_id0
                print(f"[SKIP_PARSE] ({idx_i}/{total_ids}) Subdirectory={subdir} time={dt:.1f}s | err={repr(e)}")
                _progress(f"[SKIP_PARSE] {split_name} {idx_i}/{total_ids} Subdirectory={subdir}")
                continue

            except Exception as e:
                failed += 1
                dt = time.time() - t_id0
                print(f"[FAIL]   ({idx_i}/{total_ids}) Subdirectory={subdir} time={dt:.1f}s | err={repr(e)}")
                _progress(f"[FAIL] {split_name} {idx_i}/{total_ids} Subdirectory={subdir} err={repr(e)}")
                continue

        # Build prediction DataFrame
        pred_df = pd.DataFrame(pred_rows).set_index("Subdirectory")
        common = pred_df.index.intersection(gt_rank.index)
        pred_df = pred_df.loc[common]
        gt_rank_aligned = gt_rank.loc[common]
        global_pred_df = pd.DataFrame(global_pred_rows).set_index("Subdirectory")
        global_pred_df = global_pred_df.loc[common]

        print(f"\nSkipped due to <answer> parse failure: {len(skipped_parse_ids)}")
        if skipped_parse_ids:
            print("First 50 skipped ids:", skipped_parse_ids[:50])

        # Agreement (combined = global+local when local_crop enabled, else global-only)
        res = compute_agreement_anyN([pred_df, gt_rank_aligned], prefix="Total")
        agreement = res.get('agree_0_1')
        print(f"\nAgreement (pred vs GT): {res}")
        print(f"Final agreement (agree_0_1): {agreement}")

        # Recall@1 and Filter@1
        n_methods = 4
        recall_count = 0
        filter_count = 0
        total_count = 0

        for subdir in common:
            gt_ranks_i = [float(gt_rank_aligned.loc[subdir, f"Total{k + 1}"]) for k in range(n_methods)]
            pred_ranks_i = [float(pred_df.loc[subdir, f"Total{k + 1}"]) for k in range(n_methods)]

            gt_best_rank = min(gt_ranks_i)
            gt_worst_rank = max(gt_ranks_i)
            gt_best_idx = {k for k, r in enumerate(gt_ranks_i) if r == gt_best_rank}
            gt_worst_idx = {k for k, r in enumerate(gt_ranks_i) if r == gt_worst_rank}

            pred_best_rank = min(pred_ranks_i)
            pred_worst_rank = max(pred_ranks_i)
            pred_best_idx = {k for k, r in enumerate(pred_ranks_i) if r == pred_best_rank}
            pred_worst_idx = {k for k, r in enumerate(pred_ranks_i) if r == pred_worst_rank}

            total_count += 1
            if gt_best_idx & pred_best_idx:
                recall_count += 1
            if gt_worst_idx & pred_worst_idx:
                filter_count += 1

        recall_at_1 = recall_count / total_count if total_count > 0 else None
        filter_at_1 = filter_count / total_count if total_count > 0 else None

        print(f"\n{'=' * 60}")
        print(f"  [{split_name}] Mode: {mode}  |  Local crop: {use_local_crop}")
        print(f"  Agreement : {agreement:.4f}" if agreement else "  Agreement : N/A")
        print(f"  Recall@1  : {recall_at_1:.4f}  ({recall_count}/{total_count})" if recall_at_1 is not None else "  Recall@1  : N/A")
        print(f"  Filter@1  : {filter_at_1:.4f}  ({filter_count}/{total_count})" if filter_at_1 is not None else "  Filter@1  : N/A")
        print(f"{'=' * 60}")

        # Global-only agreement (when local crop is enabled, show separate global-only stats)
        if use_local_crop:
            res_global = compute_agreement_anyN([global_pred_df, gt_rank_aligned], prefix="Total")
            agreement_global = res_global.get('agree_0_1')
            print(f"\n{'=' * 60}")
            print(f"  Agreement (global-only pred vs GT): {agreement_global:.4f}" if agreement_global else "  Agreement (global-only pred vs GT): N/A")

            box_gt0_set = set(box_gt0_subdirs) & set(common)
            if box_gt0_set:
                box_gt0_idx = pd.Index(sorted(box_gt0_set))
                print(f"\n  --- Subdirectories with box_count > 0: {len(box_gt0_idx)} ---")
                res_box_global = compute_agreement_anyN(
                    [global_pred_df.loc[box_gt0_idx], gt_rank_aligned.loc[box_gt0_idx]], prefix="Total"
                )
                print(f"  Agreement (global-only vs GT, box>0): {res_box_global.get('agree_0_1'):.4f}")
                res_box_combined = compute_agreement_anyN(
                    [pred_df.loc[box_gt0_idx], gt_rank_aligned.loc[box_gt0_idx]], prefix="Total"
                )
                print(f"  Agreement (global+local vs GT, box>0): {res_box_combined.get('agree_0_1'):.4f}")
            print(f"{'=' * 60}")

        # Annotator-level agreement (model vs individual raw annotators)
        avg_model_ann = None
        avg_inter_ann = None
        if raw_rankings_dir:
            raw_ann_dfs = load_raw_annotator_rankings(raw_rankings_dir)
            if raw_ann_dfs:
                print(f"\n{'=' * 60}")
                print(f"  Annotator Agreement ({split_name})")
                print(f"{'=' * 60}")

                model_ann_agreements = []
                for i, ann_df in enumerate(raw_ann_dfs):
                    res_ann = compute_agreement_anyN([pred_df, ann_df], prefix="Total")
                    val = res_ann.get('agree_0_1')
                    model_ann_agreements.append(val)
                    print(f"  Model vs Annotator {i+1}: {val:.4f}" if val else f"  Model vs Annotator {i+1}: N/A")

                inter_ann_agreements = []
                for (i, j) in itertools.combinations(range(len(raw_ann_dfs)), 2):
                    res_ij = compute_agreement_anyN([raw_ann_dfs[i], raw_ann_dfs[j]], prefix="Total")
                    val = res_ij.get('agree_0_1')
                    inter_ann_agreements.append(val)
                    print(f"  Annotator {i+1} vs Annotator {j+1}: {val:.4f}" if val else f"  Annotator {i+1} vs Annotator {j+1}: N/A")

                valid_ma = [v for v in model_ann_agreements if v is not None]
                valid_ia = [v for v in inter_ann_agreements if v is not None]
                avg_model_ann = sum(valid_ma) / len(valid_ma) if valid_ma else None
                avg_inter_ann = sum(valid_ia) / len(valid_ia) if valid_ia else None

                print(f"  ---")
                print(f"  Model vs Annotators (avg): {avg_model_ann:.4f}" if avg_model_ann else "  Model vs Annotators (avg): N/A")
                print(f"  Inter-annotator (avg):     {avg_inter_ann:.4f}" if avg_inter_ann else "  Inter-annotator (avg):     N/A")
                print(f"{'=' * 60}")

        total_elapsed = time.time() - t0
        _progress(
            f"\n[RESULT] {split_name}: agree_0_1={agreement}"
            f" | done={done} skip={skipped} fail={failed}"
            f" | total_time={total_elapsed / 60:.1f}min"
        )

        # Save per-sample stats CSV
        suffix = f"{mode}_crop" if use_local_crop else mode
        out_stats_csv = os.path.join(output_dir, f"{split_name}_{suffix}_stats.csv")
        pd.DataFrame(stats_rows).to_csv(out_stats_csv, index=False, encoding="utf-8")
        print(f"\nPer-sample stats saved to: {out_stats_csv}")

        # Save responses (thinking process)
        if mode == "think":
            responses_txt = os.path.join(output_dir, f"{split_name}_{suffix}_responses.txt")
            with open(responses_txt, 'w', encoding='utf-8') as f_resp:
                for rec in response_records:
                    f_resp.write(f"{'=' * 80}\n")
                    f_resp.write(f"Subdirectory: {rec['subdir']}\n")
                    f_resp.write(f"Image: {rec['image']}\n")
                    f_resp.write(f"Score: {rec['score']:.2f}\n")
                    f_resp.write(f"{'-' * 80}\n")
                    f_resp.write(f"{rec['response']}\n")
                    f_resp.write(f"{'=' * 80}\n\n")
            print(f"Full responses saved to: {responses_txt}")

    return output_buffer.getvalue(), agreement, recall_at_1, filter_at_1, avg_model_ann, avg_inter_ann


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RefReward-SR Benchmark Test (think/fast, with optional local crop)")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, choices=["think", "fast"], required=True,
                        help="Inference mode: 'think' (full reasoning) or 'fast' (skip thinking)")
    parser.add_argument("--dataset", type=str, choices=["in", "out", "both"], default="both",
                        help="Which dataset split to evaluate: 'in' (in-domain), 'out' (out-domain), 'both'")
    parser.add_argument("--local-crop", action="store_true",
                        help="Enable local crop scoring (requires --boxes-dir or --auto-detect-boxes)")
    parser.add_argument("--boxes-dir", type=str, default=None,
                        help="Directory containing {subdir}_boxes_filtered.json files")
    parser.add_argument("--auto-detect-boxes", action="store_true",
                        help="Automatically detect boxes using RAM + GroundingDINO (no pre-computed boxes needed)")
    parser.add_argument("--grounding-checkpoint", type=str, default=None,
                        help="Path to GroundingDINO checkpoint (groundingdino_swint_ogc.pth)")
    parser.add_argument("--ram-checkpoint", type=str, default=None,
                        help="Path to RAM checkpoint (ram_swin_large_14m.pth)")
    parser.add_argument("--grounding-config", type=str, default=None,
                        help="Path to GroundingDINO config (auto-detected if not provided)")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint
    checkpoint_name = os.path.basename(checkpoint_dir)
    mode = args.mode
    use_local_crop = args.local_crop
    boxes_dir = args.boxes_dir
    auto_detect = args.auto_detect_boxes

    if use_local_crop and not boxes_dir and not auto_detect:
        print("Error: --local-crop requires --boxes-dir or --auto-detect-boxes")
        sys.exit(1)
    if boxes_dir and not os.path.isdir(boxes_dir):
        print(f"Error: Boxes directory not found: {boxes_dir}")
        sys.exit(1)
    if auto_detect and (not args.grounding_checkpoint or not args.ram_checkpoint):
        print("Error: --auto-detect-boxes requires --grounding-checkpoint and --ram-checkpoint")
        sys.exit(1)

    if not os.path.isdir(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    suffix = f"{mode}_crop" if use_local_crop else mode
    output_dir = os.path.join(os.path.dirname(checkpoint_dir), f"result_refreward_{suffix}")
    os.makedirs(output_dir, exist_ok=True)

    splits = []
    if args.dataset in ("in", "both"):
        splits.append(("test_in_domain", CSV_IN_DOMAIN, IMAGE_ROOT_IN, RAW_IN_DOMAIN_DIR))
    if args.dataset in ("out", "both"):
        splits.append(("test_out_domain", CSV_OUT_DOMAIN, IMAGE_ROOT_OUT, RAW_OUT_DOMAIN_DIR))

    # --- Auto-detect boxes before loading the scoring model (saves GPU memory) ---
    if use_local_crop and auto_detect:
        from auto_box_detector import AutoBoxDetector

        boxes_dir = os.path.join(output_dir, "auto_boxes")
        os.makedirs(boxes_dir, exist_ok=True)

        _progress("[AUTO-DETECT] Loading RAM + GroundingDINO for box detection...")
        print("[AUTO-DETECT] Loading RAM + GroundingDINO for box detection...")
        detector = AutoBoxDetector(
            grounding_checkpoint=args.grounding_checkpoint,
            ram_checkpoint=args.ram_checkpoint,
            grounding_config=args.grounding_config,
        )
        print("[AUTO-DETECT] Models loaded successfully")

        for split_name, csv_path, image_root, _raw_dir in splits:
            gt_rank = load_gt_data(csv_path)
            total = len(gt_rank.index)
            _progress(f"[AUTO-DETECT] Pre-detecting boxes for {split_name} ({total} subdirectories)...")
            for idx_i, subdir in enumerate(gt_rank.index, start=1):
                out_path = os.path.join(boxes_dir, f"{subdir}_boxes_filtered.json")
                if os.path.exists(out_path):
                    continue
                hr_name = "GT.png"
                hr_path = os.path.join(image_root, subdir, hr_name)
                if not os.path.exists(hr_path):
                    continue
                try:
                    detector.detect_and_save(hr_path, out_path)
                except Exception as e:
                    _progress(f"[AUTO-DETECT] Failed for {subdir}: {repr(e)}")
                    continue
                if idx_i % 10 == 0 or idx_i == total:
                    _progress(f"[AUTO-DETECT] {split_name}: {idx_i}/{total}")

        detector.cleanup()
        del detector
        torch.cuda.empty_cache()
        _progress("[AUTO-DETECT] Detection complete, GPU memory freed")
        print("[AUTO-DETECT] Detection complete, GPU memory freed\n")

    _progress(f"[INFO] Loading model from {checkpoint_dir} ...")
    print(f"[LOADING] Loading model from {checkpoint_dir}")
    model, processor = load_model(checkpoint_dir)
    print("[LOADING] Model loaded successfully")
    _progress("[LOADING] Model loaded successfully")

    summary = {}
    for split_name, csv_path, image_root, raw_dir in splits:
        try:
            output_text, agreement, recall, filt, model_vs_ann, inter_ann = evaluate_split(
                model, processor, mode, csv_path, image_root, split_name, output_dir,
                use_local_crop=use_local_crop, boxes_dir=boxes_dir,
                raw_rankings_dir=raw_dir,
            )
            out_txt = os.path.join(output_dir, f"{checkpoint_name}_{split_name}_{suffix}_output.txt")
            with open(out_txt, 'w', encoding='utf-8') as f:
                f.write(output_text)
            print(f"\n[SUCCESS] {split_name} evaluation completed. Output: {out_txt}")
            summary[split_name] = {
                "agreement": agreement, "recall@1": recall, "filter@1": filt,
                "model_vs_annotators_avg": model_vs_ann,
                "inter_annotator_avg": inter_ann,
            }
        except Exception as e:
            print(f"\n[ERROR] Failed on {split_name}: {repr(e)}")
            import traceback
            traceback.print_exc()
            summary[split_name] = {"error": repr(e)}

    # Print final summary
    print(f"\n{'=' * 80}")
    print(f"FINAL SUMMARY  |  Checkpoint: {checkpoint_name}  |  Mode: {mode}  |  Local crop: {use_local_crop}")
    print(f"{'=' * 80}")
    for split_name, metrics in summary.items():
        if "error" in metrics:
            print(f"  {split_name}: ERROR - {metrics['error']}")
        else:
            agr = metrics['agreement']
            rec = metrics['recall@1']
            fil = metrics['filter@1']
            mva = metrics.get('model_vs_annotators_avg')
            iaa = metrics.get('inter_annotator_avg')
            print(f"  {split_name}:")
            print(f"    Agreement (vs GT)     : {agr:.4f}" if agr is not None else f"    Agreement (vs GT)     : N/A")
            print(f"    Recall@1              : {rec:.4f}" if rec is not None else f"    Recall@1              : N/A")
            print(f"    Filter@1              : {fil:.4f}" if fil is not None else f"    Filter@1              : N/A")
            print(f"    Model vs Annotators   : {mva:.4f}" if mva is not None else f"    Model vs Annotators   : N/A")
            print(f"    Inter-annotator       : {iaa:.4f}" if iaa is not None else f"    Inter-annotator       : N/A")
    print(f"{'=' * 80}")

    # Save summary JSON
    summary_json = os.path.join(output_dir, f"{checkpoint_name}_{suffix}_summary.json")
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump({
            "checkpoint": checkpoint_name, "mode": mode,
            "local_crop": use_local_crop,
            "auto_detect_boxes": auto_detect,
            "boxes_dir": boxes_dir,
            "results": summary,
        }, f, indent=2)
    print(f"Summary saved to: {summary_json}")

    del model, processor
    torch.cuda.empty_cache()
