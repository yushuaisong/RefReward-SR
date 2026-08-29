# Copyright 2025 The HuggingFace Team. All rights reserved.
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
#
# Modified by Institute of Automation, Chinese Academy of Sciences in 2026.

import os
import re
import json
import pathlib
import random

from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.utils import logging
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from open_r1.trainer import VLMGRPOTrainer_sr_qwen3vl, GRPOConfig
from open_r1.vlm_modules import *

logger = logging.get_logger(__name__)


tokenizer = None

def initialize_tokenizer(model_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    question_template: Optional[str] = field(
        default="scoring",
        metadata={
            "help": "Choose scoring or comparing question"
        },
    )


def extract_first_number(model_answer):
    match = re.search(r'-?\d+(\.\d+)?', model_answer)
    if match:
        return float(match.group())
    else:
        return random.randint(1, 5)


def fidelity_reward(pred1, pred2, var1, var2, gt, device):
    esp = 1e-6
    try:
        normal_dist = torch.distributions.Normal(0, 1)
        _cur = (pred1 - pred2) / torch.sqrt(var1 + var2 + esp)
        p = normal_dist.cdf(_cur)
    except:
        print("Meet Error ...")
        p = torch.tensor(0.5, dtype=torch.float32, device=device)
    
    reward = torch.sqrt(p * gt + esp) + torch.sqrt((1 - p) * (1 - gt) + esp)
    return reward

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    device = kwargs.get("device")
    n_gen = kwargs.get("num_generations")
    image_pair_size = kwargs.get("per_device_image_pair_size")

    reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]
    for i in range(len(reshaped_solution)):
        for j in range(len(reshaped_solution[i])):
            _cur = reshaped_solution[i][j]
            sol_match = re.search(r'<answer>(.*?)</answer>', _cur)
            ground_truth = sol_match.group(1).strip() if sol_match else sol_match.strip()
            reshaped_solution[i][j] = float(ground_truth)

    contents = [completion[0]["content"] for completion in completions]
    reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

    batch_mean, batch_var, batch_pred = [], [], []
    for i in range(len(reshaped_content)):
        cur_pred_list = []
        for j in range(len(reshaped_content[i])):
            try:
                content_matches = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                student_answer = content_matches[-1].strip() if content_matches else reshaped_content[i][j].strip()
                pred = extract_first_number(student_answer)
            except:
                print("Meet Error ...")
                pred = random.uniform(1, 5)
            cur_pred_list.append(pred)
        
        batch_pred.append(cur_pred_list)
        p = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
        p_mean = torch.mean(p)
        p_var = torch.var(p)
        batch_mean.append([p_mean])
        batch_var.append([p_var])
    
    rewards = []

    for i in range(len(batch_pred)):
        for j in range(len(batch_pred[i])):
            _reward_sum, _count_idx = 0, 0

            group_id = i // image_pair_size
            start = group_id * image_pair_size
            end = start + image_pair_size

            for z in range(start, end):
                if z != i:
                    input_pred1 = batch_pred[i][j]
                    input_pred2 = batch_mean[z][0]
                    input_var1 = batch_var[i][0]
                    input_var2 = batch_var[z][0]

                    if reshaped_solution[i][j] < reshaped_solution[z][0]:
                        input_gt = torch.tensor(1.0, dtype=torch.float32, device=device)
                    elif reshaped_solution[i][j] > reshaped_solution[z][0]:
                        input_gt = torch.tensor(0.0, dtype=torch.float32, device=device)
                    else:
                        input_gt = torch.tensor(0.5, dtype=torch.float32, device=device)

                    _reward = fidelity_reward(
                        pred1=input_pred1,
                        pred2=input_pred2,
                        var1=input_var1,
                        var2=input_var2,
                        gt=input_gt,
                        device=device
                    )

                    _reward_sum += _reward
                    _count_idx += 1

            _cur_reward = _reward_sum / _count_idx
            rewards.append(_cur_reward)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("HR_image_path") if "HR_image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                image_path = [image_path[i:i + n_gen] for i in range(0, len(image_path), n_gen)]

                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {_cur_reward} -------------\n")
                    f.write(f"accu_reward_method: {_cur_reward}\n")
                    f.write(f"image_path: {image_path[i][j]}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {reshaped_content[i][j]}\n")
                    f.write(f"Solution: {reshaped_solution[i][j]}\n") 

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format.""" 
    pattern = r"<thinking>.*?</thinking>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")

    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward
}

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)
    question_prompt = vlm_module_cls.get_question_template_sr(task_type=script_args.question_template)

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")

    training_args.max_completion_length = 256
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    if script_args.reward_method is None:
        accu_reward_methods = ["default"] * len(data_files)
    else:
        accu_reward_methods = script_args.reward_method.split(":")
        assert len(accu_reward_methods) == len(data_files), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"
    
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    all_data = []
    for data_file, image_folder, accu_reward_method in zip(data_files, image_folders, accu_reward_methods):
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
              
                item['HR_image_path'] = [os.path.join(image_folder, item['folder_name'], image) for image in item['HR_image']]
                del item['HR_image']
                item['LR_image_path'] = [os.path.join(image_folder, item['folder_name'], image) for image in item['LR_image']]
                del item['LR_image']

                item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
                
                # Handle solution that could be a float or string
                solution_value = item['conversations'][1]['value']
                if isinstance(solution_value, str):
                    item['solution'] = solution_value.replace('<answer>', '').replace('</answer>', '').strip()
                else:
                    item['solution'] = str(solution_value)
                
                del item['conversations']
                item['accu_reward_method'] = item.get('accu_reward_method', accu_reward_method)
                all_data.append(item)

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        return {
            'HR_image_path': example['HR_image_path'],
            'LR_image_path': example['LR_image_path'],
            "folder_name": example['folder_name'],
            'problem': example['problem'],
            'solution': f"<answer> {example['solution']} </answer>",
            'accu_reward_method': example['accu_reward_method'],
            'messages': [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example['LR_image_path'][0]},
                        {"type": "image", "image": example['HR_image_path'][0]},
                        {"type": "text", "text": question_prompt.format(Question=example['problem'])},
                    ],
                }
            ]
        }

    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)

    splits = {'train': dataset}

    trainer_cls = VLMGRPOTrainer_sr_qwen3vl
    print("using trainer:", trainer_cls.__name__)
    initialize_tokenizer(model_args.model_name_or_path)

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
