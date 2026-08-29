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

"""
Automatic bounding box detection using RAM (Recognize Anything Model) + GroundingDINO.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from PIL import Image

# ---------------------------------------------------------------------------
# Low-texture label sets – boxes on these regions are filtered out because
# they lack meaningful high-frequency detail for SR quality assessment.
# ---------------------------------------------------------------------------
LOW_TEXTURE_PHRASES = {
    "sky", "blue sky", "night sky", "evening sky",
    "sunrise", "sunset", "twilight", "dusk", "dawn", "horizon",
    "cloud", "cloudy", "storm", "storm cloud", "stormy",
    "thunderstorm", "fog", "foggy", "smoke", "rain", "rainy",
    "snow", "snowy", "snowstorm", "snowfield",
    "ice", "glacier", "iceberg", "ice shelf", "ice floe", "ice cave", "tundra",
    "water", "water surface", "underwater",
    "sea", "sea view", "lake", "salt lake", "river", "river valley",
    "coast", "shore", "shoreline", "beach", "sand", "sand bar",
    "desert", "river bank", "riverbed",
}

LOW_TEXTURE_TOKENS = {
    "sky", "cloud", "fog", "smoke", "horizon", "storm", "rain", "snow",
    "water", "surface", "river", "lake", "sea", "underwater",
    "ice", "glacier", "iceberg",
    "shore", "shoreline", "coast", "beach", "sand",
    "field",
}


def _normalize_label(label: str) -> str:
    base = label.rsplit("(", 1)[0].strip()
    return base.lower()


def _is_low_texture_label(label: str) -> bool:
    base = _normalize_label(label)
    if base in LOW_TEXTURE_PHRASES:
        return True
    tokens = [t for t in re.split(r"[^a-z]+", base) if t]
    if not tokens:
        return False
    return all(token in LOW_TEXTURE_TOKENS for token in tokens)


def _aspect_ratio_too_large(box: torch.Tensor, max_ratio: float) -> bool:
    x0, y0, x1, y1 = box.tolist()
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    if w == 0 or h == 0:
        return True
    return max(w / h, h / w) > max_ratio


def _area_ratio(box_norm: torch.Tensor) -> float:
    """box_norm is (cx, cy, w, h) in normalised coordinates."""
    return float(max(0.0, box_norm[2].item()) * max(0.0, box_norm[3].item()))


def _should_remove_by_overlap(
    box_a: torch.Tensor, box_b: torch.Tensor, overlap_thresh: float,
) -> bool:
    ax0, ay0, ax1, ay1 = box_a.tolist()
    bx0, by0, bx1, by1 = box_b.tolist()
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return False
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    if area_a <= 0 or area_b <= 0:
        return False
    return inter / min(area_a, area_b) >= overlap_thresh


def _remove_overlaps(
    boxes: List[torch.Tensor],
    labels: List[str],
    scores: List[float],
    mode: str,
    overlap_thresh: float,
) -> Tuple[List[torch.Tensor], List[str], List[float]]:
    keep = list(range(len(boxes)))
    changed = True
    while changed:
        changed = False
        for i in range(len(keep)):
            if changed:
                break
            for j in range(i + 1, len(keep)):
                idx_a, idx_b = keep[i], keep[j]
                if not _should_remove_by_overlap(boxes[idx_a], boxes[idx_b], overlap_thresh):
                    continue
                area_a = (boxes[idx_a][2] - boxes[idx_a][0]).item() * (
                    boxes[idx_a][3] - boxes[idx_a][1]
                ).item()
                area_b = (boxes[idx_b][2] - boxes[idx_b][0]).item() * (
                    boxes[idx_b][3] - boxes[idx_b][1]
                ).item()
                remove_idx = idx_b if mode == "keep_large" else idx_a
                if mode == "keep_large" and area_a < area_b:
                    remove_idx = idx_a
                if mode == "keep_small" and area_a < area_b:
                    remove_idx = idx_b
                if area_a == area_b:
                    remove_idx = idx_a if scores[idx_a] < scores[idx_b] else idx_b
                keep.remove(remove_idx)
                changed = True
                break
    return (
        [boxes[i] for i in keep],
        [labels[i] for i in keep],
        [scores[i] for i in keep],
    )


# ---------------------------------------------------------------------------
# Lazy import helpers – GroundingDINO and RAM are only required when
# AutoBoxDetector is actually instantiated.
# ---------------------------------------------------------------------------
def _import_groundingdino():
    """Import GroundingDINO, supporting both pip-installed and local-clone paths."""
    try:
        import groundingdino.datasets.transforms as T
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
        import groundingdino as _pkg
        return T, build_model, SLConfig, clean_state_dict, get_phrases_from_posmap, _pkg
    except ImportError:
        pass
    try:
        import GroundingDINO.groundingdino.datasets.transforms as T
        from GroundingDINO.groundingdino.models import build_model
        from GroundingDINO.groundingdino.util.slconfig import SLConfig
        from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
        import GroundingDINO.groundingdino as _pkg
        return T, build_model, SLConfig, clean_state_dict, get_phrases_from_posmap, _pkg
    except ImportError:
        raise ImportError(
            "GroundingDINO is required for --auto-detect-boxes.\n"
            "Install with: pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
        )


def _import_ram():
    """Import RAM (Recognize Anything Model)."""
    try:
        from ram.models import ram
        from ram import inference_ram
        return ram, inference_ram
    except ImportError:
        raise ImportError(
            "recognize-anything (RAM) is required for --auto-detect-boxes.\n"
            "Install with: pip install git+https://github.com/xinyu1205/recognize-anything.git\n"
            "You may also need: pip install fairscale"
        )


def _find_grounding_config(pkg) -> Optional[str]:
    """Try to locate the bundled GroundingDINO config file."""
    cfg = os.path.join(os.path.dirname(pkg.__file__), "config", "GroundingDINO_SwinT_OGC.py")
    if os.path.exists(cfg):
        return cfg
    return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class AutoBoxDetector:
    """Detect and filter bounding boxes using RAM + GroundingDINO.

    Usage::

        detector = AutoBoxDetector(
            grounding_checkpoint="groundingdino_swint_ogc.pth",
            ram_checkpoint="ram_swin_large_14m.pth",
        )
        box_record = detector.detect("image.png")
        detector.cleanup()

    ``box_record`` is a dict compatible with the ``_load_boxes()`` format::

        {
            "image": "<path>",
            "tags": "cat, dog, ...",
            "boxes": [{"label": "cat(0.87)", "score": 0.87, "bbox": [x0, y0, x1, y1]}, ...],
            "orig_size": [width, height],
        }
    """

    def __init__(
        self,
        grounding_checkpoint: str,
        ram_checkpoint: str,
        grounding_config: Optional[str] = None,
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        iou_threshold: float = 0.5,
        min_area_ratio: float = 0.1,
        max_area_ratio: float = 0.7,
        max_aspect_ratio: float = 4.5,
        overlap_ratio: float = 0.7,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.iou_threshold = iou_threshold
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.overlap_ratio = overlap_ratio

        (
            self._T, self._build_model, self._SLConfig,
            self._clean_state_dict, self._get_phrases,
            self._gdino_pkg,
        ) = _import_groundingdino()
        self._ram_cls, self._inference_ram = _import_ram()
        import torchvision.transforms as TS
        self._TS = TS

        if grounding_config is None:
            grounding_config = _find_grounding_config(self._gdino_pkg)
        if grounding_config is None or not os.path.exists(grounding_config):
            raise FileNotFoundError(
                f"GroundingDINO config not found: {grounding_config}\n"
                "Provide --grounding-config explicitly."
            )

        args = self._SLConfig.fromfile(grounding_config)
        args.device = device
        self.dino_model = self._build_model(args)
        ckpt = torch.load(grounding_checkpoint, map_location="cpu")
        self.dino_model.load_state_dict(
            self._clean_state_dict(ckpt["model"]), strict=False,
        )
        self.dino_model.eval()

        self.ram_model = self._ram_cls(
            pretrained=ram_checkpoint, image_size=384, vit="swin_l",
        )
        self.ram_model.eval()
        self.ram_model = self.ram_model.to(device)

    # ---- internal helpers ------------------------------------------------

    def _load_image(self, image_path: str) -> Tuple[Image.Image, torch.Tensor]:
        image_pil = Image.open(image_path).convert("RGB")
        transform = self._T.Compose([
            self._T.RandomResize([800], max_size=1333),
            self._T.ToTensor(),
            self._T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)
        return image_pil, image_tensor

    def _get_ram_tags(self, image_pil: Image.Image) -> str:
        normalize = self._TS.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )
        transform = self._TS.Compose([
            self._TS.Resize((384, 384)), self._TS.ToTensor(), normalize,
        ])
        raw_image = image_pil.resize((384, 384))
        raw_image = transform(raw_image).unsqueeze(0).to(self.device)
        tags, _ = self._inference_ram(raw_image, self.ram_model)
        return tags.replace(" |", ",")

    def _get_grounding_output(self, image_tensor: torch.Tensor, caption: str):
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."
        model = self.dino_model.to(self.device)
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            outputs = model(image_tensor[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        filt_mask = logits.max(dim=1)[0] > self.box_threshold
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]

        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases = []
        scores = []
        for logit, _ in zip(logits_filt, boxes_filt):
            phrase = self._get_phrases(
                logit > self.text_threshold, tokenized, tokenizer,
            )
            pred_phrases.append(phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases

    # ---- public API ------------------------------------------------------

    @torch.no_grad()
    def detect(self, image_path: str) -> dict:
        """Detect and filter boxes for a single image.

        Returns a dict with keys ``image``, ``tags``, ``boxes``, ``orig_size``
        that is compatible with the pre-computed ``*_boxes_filtered.json``
        format used by the test scripts.
        """
        image_pil, image_tensor = self._load_image(image_path)
        tags = self._get_ram_tags(image_pil)

        boxes_filt, scores, pred_phrases = self._get_grounding_output(
            image_tensor, tags,
        )

        w, h = image_pil.size
        boxes_norm = boxes_filt.clone()
        if boxes_filt.numel():
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([w, h, w, h])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            nms_idx = torchvision.ops.nms(
                boxes_filt, scores, self.iou_threshold,
            ).numpy().tolist()
            boxes_filt = boxes_filt[nms_idx]
            boxes_norm = boxes_norm[nms_idx].cpu()
            pred_phrases = [pred_phrases[i] for i in nms_idx]
            scores = scores[nms_idx]
        else:
            boxes_filt = boxes_filt.cpu()
            boxes_norm = boxes_norm.cpu()

        # ---- filtering (low-texture, aspect ratio, area, overlap) --------
        filtered_boxes: List[torch.Tensor] = []
        filtered_labels: List[str] = []
        filtered_scores: List[float] = []

        for box, box_n, label, score in zip(
            boxes_filt, boxes_norm, pred_phrases, scores,
        ):
            if _is_low_texture_label(label):
                continue
            if _aspect_ratio_too_large(box, self.max_aspect_ratio):
                continue
            ratio = _area_ratio(box_n)
            if ratio < self.min_area_ratio or ratio > self.max_area_ratio:
                continue
            filtered_boxes.append(box)
            filtered_labels.append(label)
            filtered_scores.append(float(score.item()))

        filtered_boxes, filtered_labels, filtered_scores = _remove_overlaps(
            filtered_boxes, filtered_labels, filtered_scores,
            "keep_large", self.overlap_ratio,
        )

        boxes_out = []
        for box, label, score in zip(
            filtered_boxes, filtered_labels, filtered_scores,
        ):
            x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
            boxes_out.append({
                "label": label,
                "score": float(score),
                "bbox": [x0, y0, x1, y1],
            })

        return {
            "image": image_path,
            "tags": tags,
            "boxes": boxes_out,
            "orig_size": [w, h],
        }

    def detect_and_save(self, image_path: str, output_path: str) -> dict:
        """Detect boxes and save to JSON file."""
        record = self.detect(image_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return record

    def cleanup(self):
        """Release GPU memory held by RAM and GroundingDINO models."""
        if hasattr(self, "dino_model"):
            del self.dino_model
        if hasattr(self, "ram_model"):
            del self.ram_model
        torch.cuda.empty_cache()
