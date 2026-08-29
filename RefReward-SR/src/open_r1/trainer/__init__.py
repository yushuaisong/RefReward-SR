# Derived from https://github.com/TianheWu/VisualQuality-R1/blob/master/src/open-r1-multimodal/src/open_r1/trainer/__init__.py.
# Modified by Institute of Automation, Chinese Academy of Sciences in 2026.

from .grpo_config import GRPOConfig
from .grpo_trainer_sr_qwen3vl import VLMGRPOTrainer_sr_qwen3vl

__all__ = ["GRPOConfig", "VLMGRPOTrainer_sr_qwen3vl"]
