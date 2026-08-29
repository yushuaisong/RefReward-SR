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
import json
import pandas as pd

BASE_ROOT = "RefReward-SR/dataset"

# TRAIN_IMAGES_ROOT = os.path.join(BASE_ROOT, "train_images")
TRAIN_IMAGES_ROOT = "/path/to/RefReward-SR-dataset/train_images"
CSV_PATH = os.path.join(BASE_ROOT, "train.csv")
OUT_JSONL = os.path.join(BASE_ROOT, "train_sr.jsonl")
OUT_SKIPPED_CSV = os.path.join(BASE_ROOT, "skipped_details.csv")

PROMPT = (
    "You are a highly discerning expert in image quality assessment."
    "Your task is to evaluate a super-resolution (SR) output by comparing it with the original low-resolution reference."
    "You will be given two images: the first (LR) shows the original scene, and the second (HR) is the super-resolution result."
    "Assume the HR image may contain problematic regions."
    "Carefully inspect both images and identify any areas in the HR image that look unnatural or distorted (e.g., distorted regions or warped lines, unrealistic textures, unreasonable objects) or semantically inconsistent with the LR image (e.g., missing, added or altered objects, inconsistent texture)."
    "For each problematic region, state its approximate location in the HR image (e.g., 'top left', 'center lower') and briefly explain why it appears incorrect."
    "After inspection, assign an overall quality score from 1.00 to 5.00, with two decimal places (e.g., 1.31, 2.77, 4.53)."
)

def to_subdir4(x) -> str:
    return str(int(str(x).strip())).zfill(4)

df = pd.read_csv(CSV_PATH, encoding="gbk")

rank_map = {}   # subdir7 -> {image_name -> final_rank}
order_map = {}  # subdir7 -> [img1, img2, img3, img4]

for _, row in df.iterrows():
    subdir7 = to_subdir4(row["Subdirectory"])

    img_list = []
    img_to_rank = {}
    for i in range(1, 5):
        img = str(row[f"Image{i}"]).strip()
        rnk = float(row[f"FinalRank{i}"])
        img_list.append(img)
        img_to_rank[img] = rnk

    rank_map[subdir7] = img_to_rank
    order_map[subdir7] = img_list

all_subdirs = []
for name in os.listdir(TRAIN_IMAGES_ROOT):
    p = os.path.join(TRAIN_IMAGES_ROOT, name)
    if os.path.isdir(p) and name.isdigit():
        all_subdirs.append(name)

all_subdirs.sort(key=lambda s: int(s))

os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

written = 0
skipped_no_csv = 0
skipped_missing_files = 0
skipped_details = []

with open(OUT_JSONL, "w", encoding="utf-8") as f:
    cur_id = 1

    for folder_name in all_subdirs:
        folder_path = os.path.join(TRAIN_IMAGES_ROOT, folder_name)

        if folder_name not in rank_map:
            skipped_no_csv += 1
            skipped_details.append({
                "folder_name": folder_name,
                "reason": "no_csv_row",
                "detail": "Subdirectory exists in train_images but not found in CSV"
            })
            continue

        lr_name = "LR.png"
        lr_path = os.path.join(folder_path, lr_name)
        if not os.path.exists(lr_path):
            skipped_missing_files += 1
            skipped_details.append({
                "folder_name": folder_name,
                "reason": "missing_lr",
                "detail": lr_path
            })
            continue

        hr_list = order_map[folder_name]
        missing_hr = [hr for hr in hr_list if not os.path.exists(os.path.join(folder_path, hr))]
        if missing_hr:
            skipped_missing_files += 1
            skipped_details.append({
                "folder_name": folder_name,
                "reason": "missing_hr",
                "detail": ";".join(missing_hr)
            })
            continue

        img_to_rank = rank_map[folder_name]
        for hr_name in hr_list:
            obj = {
                "id": cur_id,
                "folder_name": folder_name,
                "HR_image": [hr_name],
                "LR_image": [lr_name],
                "conversations": [
                    {"from": "human", "value": PROMPT},
                    {"from": "gpt", "value": float(img_to_rank[hr_name])},
                ],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
            cur_id += 1

print("===== Done =====")
print(f"Output JSONL: {OUT_JSONL}")
print(f"Written lines: {written}")
print(f"Skipped (no CSV row): {skipped_no_csv}")
print(f"Skipped (missing files): {skipped_missing_files}")

if skipped_details:
    skipped_df = pd.DataFrame(skipped_details)
    skipped_df.to_csv(OUT_SKIPPED_CSV, index=False, encoding="utf-8")
    print(f"\nSkipped details saved to: {OUT_SKIPPED_CSV}")

    no_csv = [x["folder_name"] for x in skipped_details if x["reason"] == "no_csv_row"]
    if no_csv:
        print("\n[Skipped - no_csv_row] folder_name list:")
        for x in no_csv:
            print(" ", x)
else:
    print("\nNo skipped details.")
