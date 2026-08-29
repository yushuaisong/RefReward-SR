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

# python test_metrics_full.py   --inp_imgs "${ROOT_DIR}/DanceGRPO_SR/preset/datasets/flux-grpo-180"   --gt_imgs "${ROOT_DIR}/DanceGRPO_SR/preset/datasets/benchmark_realsr/test_HR"   --log "${ROOT_DIR}/OSEDiff/preset/datasets/benchmark_realsr/flux-grpo-180-all"

import os
import sys
import glob
import argparse
import logging
from datetime import datetime
import time

import cv2
import torch
import pyiqa
from basicsr.utils import img2tensor

def get_timestamp():
    """Returns the current timestamp in a specific format."""
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    """Sets up a logger with specified configurations."""
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )
    logger.setLevel(level)

    if tofile:
        log_file = os.path.join(root, f"{phase}_{get_timestamp()}.log")
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

def dict2str(opt, indent=1):
    """Converts a dictionary to a formatted string for logging."""
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent * 2) + f"{k}:[\n"
            msg += dict2str(v, indent + 1)
            msg += ' ' * (indent * 2) + "]\n"
        else:
            msg += ' ' * (indent * 2) + f"{k}: {v}\n"
    return msg

def main():
    parser = argparse.ArgumentParser(description="Image Quality Assessment Script (Full Table)")

    parser.add_argument(
        "--inp_imgs",
        nargs="+",
        required=True,
        help="Path(s) to the input (SR) images directories."
    )

    parser.add_argument(
        "--gt_imgs",
        nargs="+",
        required=True,
        help="Path(s) to the ground truth (GT) images directories."
    )

    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Directory path to save the log files."
    )

    parser.add_argument(
        "--log_name",
        type=str,
        default='METRICS',
        help="Base name for the log files."
    )
    
    parser.add_argument(
        "--skip_heavy",
        action='store_true',
        help="If set, skips heavy models like QAlign to save time/memory."
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create log directory if it doesn't exist
    os.makedirs(args.log, exist_ok=True)

    # Initialize logger
    try:
        # Try to infer a meaningful name from the first path
        args.log_name = os.path.basename(os.path.normpath(args.inp_imgs[0]))
    except:
        args.log_name = 'METRICS'
        
    setup_logger('base', args.log, f'test_{args.log_name}', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info("===== Configuration =====")
    logger.info(dict2str(vars(args)))
    logger.info("==========================\n")

    logger.info("Initializing IQA metrics based on the table...")
    
    nr_metric_names = [
        'MANIQA', 'MUSIQ', 'CLIPIQA', 'CLIPIQA+', 'TOPIQ-NR', 'AFINE-NR', 
        'QALIGN', 'NIMA', 'TOPIQ-IAA'
    ]

    metrics_dict = {
        # --- Pink Region (Fidelity) ---
        'PSNR': 'psnr',
        'SSIM': 'ssim',
        
        # --- Blue Region (FR Perception) ---
        'LPIPS': 'lpips',
        'TOPIQ-FR': 'topiq_fr',
        'AFINE-FR': 'afine_fr',

        # --- Green Region (NR Quality) ---
        'MANIQA': 'maniqa-pipal',
        'MUSIQ': 'musiq',
        'CLIPIQA': 'clipiqa',
        'CLIPIQA+': 'clipiqa+',
        'TOPIQ-NR': 'topiq_nr',
        'AFINE-NR': 'afine_nr',
        
        # --- Yellow Region (Aesthetics) ---
        'NIMA': 'nima',
        'TOPIQ-IAA': 'topiq_iaa'
    }

    if not args.skip_heavy:
        metrics_dict['QALIGN'] = 'qalign'
    else:
        logger.warning("Skipping QALIGN as per --skip_heavy flag.")

    iqa_metrics = {}
    for name, model_name in metrics_dict.items():
        try:
            logger.info(f"Loading {name} ({model_name})...")
            # Y channel handling for PSNR/SSIM
            if name in ['PSNR', 'SSIM']:
                iqa_metrics[name] = pyiqa.create_metric(model_name, test_y_channel=True, color_space='ycbcr').to(device)
            else:
                iqa_metrics[name] = pyiqa.create_metric(model_name, device=device)
        except Exception as e:
            logger.error(f"Failed to load metric {name}: {e}")

    # Initialize FID separately (Directory based)
    try:
        fid_metric = pyiqa.create_metric('fid', device=device)
    except Exception as e:
        logger.error(f"Failed to load FID: {e}")
        fid_metric = None

    logger.info(f"Successfully initialized {len(iqa_metrics)} metrics.\n")

    # Validate input and GT directories
    if len(args.inp_imgs) != len(args.gt_imgs):
        logger.error("The number of input image directories and GT image directories must be the same.")
        sys.exit(1)

    init_imgs_names = []
    for dir_idx, init_dir in enumerate(args.inp_imgs):
        gt_dir = args.gt_imgs[dir_idx]
        img_gt_list = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
        img_sr_list = sorted(glob.glob(os.path.join(init_dir, '*.png')))
        
        # Fallback to jpg if png not found
        if not img_sr_list:
            img_sr_list = sorted(glob.glob(os.path.join(init_dir, '*.jpg')))
        if not img_gt_list:
             img_gt_list = sorted(glob.glob(os.path.join(gt_dir, '*.jpg')))

        dir_name = os.path.basename(os.path.normpath(init_dir))
        init_imgs_names.append(dir_name)

        logger.info(f"Directory [{dir_name}]: {len(img_gt_list)} GT images vs {len(img_sr_list)} SR images.")
        if len(img_gt_list) != len(img_sr_list):
            logger.warning(f"Mismatch in number of images! Evaluating only overlapping files based on sorting.")
            min_len = min(len(img_gt_list), len(img_sr_list))
            img_gt_list = img_gt_list[:min_len]
            img_sr_list = img_sr_list[:min_len]

    logger.info("\n===== Starting Evaluation =====\n")

    # Iterate over each directory
    for dir_idx, init_dir in enumerate(args.inp_imgs):
        gt_dir = args.gt_imgs[dir_idx]
        img_gt_list = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
        img_sr_list = sorted(glob.glob(os.path.join(init_dir, '*.png')))
        
        # Re-verify lists inside loop
        if not img_sr_list: img_sr_list = sorted(glob.glob(os.path.join(init_dir, '*.jpg')))
        if not img_gt_list: img_gt_list = sorted(glob.glob(os.path.join(gt_dir, '*.jpg')))
        min_len = min(len(img_gt_list), len(img_sr_list))
        img_gt_list = img_gt_list[:min_len]
        img_sr_list = img_sr_list[:min_len]

        dir_name = init_imgs_names[dir_idx]

        # Initialize accumulators for average metrics
        metrics_accum = {metric: 0.0 for metric in iqa_metrics.keys()}

        logger.info(f"Testing Directory: [{dir_name}]")

        # Iterate over each image pair
        for img_idx, sr_path in enumerate(img_sr_list):
            gt_path = img_gt_list[img_idx]
            img_name = os.path.basename(sr_path)

            start_time = time.time()

            # Read and preprocess images
            sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)

            if sr_img is None or gt_img is None:
                logger.warning(f"Image read failed for {img_name}. Skipping.")
                continue

            # Compute metrics
            with torch.no_grad():
                metrics = {}
                for name, metric in iqa_metrics.items():
                    try:
                        sr_tensor = img2tensor(sr_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0
                        gt_tensor = img2tensor(gt_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0
                        
                        if name in nr_metric_names:
                            metrics[name] = metric(sr_tensor).item()
                        else:
                            metrics[name] = metric(sr_tensor, gt_tensor).item()
                    except Exception as e:
                        logger.error(f"Error computing {name} for {img_name}: {e}")
                        metrics[name] = float('nan')

            # Accumulate metrics (skip NaN values)
            for name in metrics_accum:
                if name in metrics and not (isinstance(metrics[name], float) and metrics[name] != metrics[name]):
                    metrics_accum[name] += metrics[name]

            # Calculate runtime
            end_time = time.time()
            runtime = end_time - start_time

            # Log per-image metrics and runtime
            metrics_str = "; ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            logger.info(f"{dir_name}/{img_name} | {metrics_str} | Time: {runtime:.2f}s")

        # Compute average metrics
        num_images = len(img_sr_list)
        if num_images > 0:
            avg_metrics = {k: round(v / num_images, 4) for k, v in metrics_accum.items()}
        else:
            avg_metrics = {k: 0.0 for k in metrics_accum.keys()}

        # Compute FID for the directory
        fid_value = 0.0
        fid_runtime = 0.0
        if fid_metric is not None:
            logger.info("Calculating FID... (This might take a while)")
            fid_start_time = time.time()
            try:
                fid_value = fid_metric(init_dir, gt_dir).item()
            except Exception as e:
                logger.error(f"Error calculating FID: {e}")
                fid_value = -1.0
            fid_end_time = time.time()
            fid_runtime = fid_end_time - fid_start_time

        # Log average metrics for the directory
        avg_metrics_str = "; ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        logger.info(f"\n===== Average Metrics for [{dir_name}] =====\n{avg_metrics_str} | FID: {fid_value:.6f} | FID Time: {fid_runtime:.2f}s\n")

    logger.info("===== Evaluation Completed =====")

if __name__ == "__main__":
    main()
