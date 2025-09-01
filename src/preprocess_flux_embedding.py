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

import yaml
import ast
import numpy as np
import argparse
import torch
import pandas as pd
from accelerate.logging import get_logger
import cv2  
import json
import os
from PIL import Image
import torch.distributed as dist
from pathlib import Path  
from flux.pipeline_tools import encode_images, encode_cond_images, prepare_text_input
logger = get_logger(__name__)
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
from utils.data_utils import read_odps_table, ossProcessor, download_image
from diffusers import FluxPipeline

model_weights = {
    "flux": {
        "model_cache_dir": "/mnt/workspace/user/zengdawei.zdw/image_aigc_service_icanvas/models/huggingface/hub",
        "model_path": "/mnt/workspace/user/zengdawei.zdw/image_aigc_service_icanvas/models/huggingface/hub/models--black-forest-labs--FLUX.1-dev",
        "bfl_repo": "black-forest-labs/FLUX.1-dev",
        "access_token": ""
    },
    "ecomini": {
        "model_cache_dir": "/mnt/workspace/user/zengdawei.zdw/tmp/OminiControl/runs/train_mulcond_1024_fuse_20250411-021218",
        "ckpt_path": "ckpt/480000",
        "lora_name": "pytorch_lora_weights.safetensors"
    },
    "depth-estimation": {
        "model_repo": "LiheYoung/depth-anything-small-hf"
    },
    "lama_inpainting": {
        "model_cache_dir": "/mnt/workspace/user/zengdawei.zdw/image_aigc_service_icanvas/models/damo/cv_fft_inpainting_lama"
    }
}

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def make_prompt(prompt_struct):
    try:
        prompt = None
        l = len(prompt_struct)

        if len(prompt_struct) <= 10:
            print("Invalid structured prompt:\n")
            print(prompt_struct)
        scene_dsgn = ast.literal_eval(prompt_struct)
        fore_scene = scene_dsgn.get("Foreground", "")
        prompt = fore_scene + " " + \
            scene_dsgn.get("Background", "") + " " + \
            scene_dsgn.get("Scene theme", "") + ". " + \
            scene_dsgn.get("Tone", "") # + style_prompt
    except Exception as e:
        print("err: ", e)
        print("Invalid structured prompt: \n")
        print(prompt_struct)

    return prompt


class MultiCondDataset(Dataset):
    def __init__(
        self, txt_path, vae_debug,
    ):
        self.txt_path = txt_path
        self.vae_debug = vae_debug

        self.train_dataset = pd.read_csv(self.txt_path)

        # with open(self.txt_path, "r", encoding="utf-8") as f:
            # self.train_dataset = [ line for line in f.read().splitlines() if not contains_chinese(line)][:50000]

    def __getitem__(self, idx):
        #import pdb;pdb.set_trace()
        item_id, item_title, pcate_lv1_id, pcate_lv1_name, pcate_lv2_id, pcate_lv2_name, \
            pcate_lv3_id, pcate_lv3_name, ori_image_url, seg_mask_url, prompt, prompt_id, negative_prompt, \
                item_image_content_id, gen_type = self.train_dataset.iloc[idx]
        
        prompt = make_prompt(prompt)
        
        if self.vae_debug:
            latents = torch.load(
                os.path.join(
                    args.output_dir, "latent", self.train_dataset.iloc[idx]["latent_path"]
                ),
                map_location="cpu",
            )
        else:
            latents = []

        return dict(item_id=item_id, item_title=item_title, pcate_lv3_name=pcate_lv3_name, prompt=prompt, \
            ori_image_url=ori_image_url, seg_mask_url=seg_mask_url, latents=latents)

    def __len__(self):
        return self.train_dataset.shape[0]

def get_config(config_path):
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_pipeline(device):
    model_cache_dir = model_weights["flux"].get("model_cache_dir")
    access_token = model_weights["flux"].get("access_token", "")

    root_path = model_weights["ecomini"].get("model_cache_dir")
    lora_path = os.path.join(root_path, model_weights["ecomini"].get("ckpt_path"))
    lora_name = model_weights["ecomini"].get("lora_name", "")
    config_path = os.path.join(root_path, "config.yaml")
    config = get_config(config_path)
    model_config = config.get("model", {})

    pipe = FluxPipeline.from_pretrained(
        pretrained_model_name_or_path=model_weights["flux"].get("bfl_repo"),
        cache_dir=model_cache_dir,
        torch_dtype=torch.bfloat16,
        token=access_token
    ).to(device)

    pipe.load_lora_weights(
        lora_path,
        weight_name=lora_name,
    )
    
    # merge lora weight to base layers
    pipe.fuse_lora(lora_scale=1.0)
    pipe.unload_lora_weights()
    return pipe

def depth_pipe():
    from transformers import pipeline
    model_cache_dir = "/mnt/workspace/user/zengdawei.zdw/image_aigc_service_icanvas/models/huggingface/hub"
    _depth_pipe = pipeline(
        task="depth-estimation",
        model="LiheYoung/depth-anything-small-hf",
        cache_dir=model_cache_dir,
        device="cpu",
    )
    return _depth_pipe

def get_canny_edge(img):
    condition_size = 512
    resize_ratio = condition_size / max(img.size)
    img = img.resize(
        (int(img.size[0] * resize_ratio), int(img.size[1] * resize_ratio))
    )
    img_np = np.array(img)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 50, 200) # TODO low_thres 50 or 100
    return Image.fromarray(edges).convert("RGB")

def masked_image_proc(image):
    if image.mode != 'RGBA':
        raise ValueError("输入图像必须是RGBA模式")
    white_bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(white_bg, image)

    return composite.convert('RGB')

def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=local_rank
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "text_ids"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "pooled_prompt_embeds"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "depth_img"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "canny_img"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "ori_img"), exist_ok=True)

    latents_txt_path = args.prompt_csv
    train_dataset = MultiCondDataset(latents_txt_path, args.vae_debug)
    sampler = DistributedSampler(
        train_dataset, rank=local_rank, num_replicas=world_size, shuffle=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # load model
    flux_pipe = load_pipeline(device)
    ossproc = ossProcessor(bucket_name="ae-p4p-sg-dev")

    json_data = []
    cidx = 0
    for _, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        try:
            with torch.inference_mode():
                if args.vae_debug:
                    latents = data["latents"]
                for idx, item_id in enumerate(data["item_id"]):
                    title = data["item_title"]
                    pcate_lv3_name = data["pcate_lv3_name"]
                    prompts = data["prompt"]
                    ori_image_url = data["ori_image_url"]
                    seg_mask_url = data["seg_mask_url"]
                    latents = data["latents"]

                    item_id_str = str(item_id.item())
                    
                    prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                        flux_pipe, prompts=prompts
                    )
                    
                    prompt_embed_path = os.path.join(
                        args.output_dir, "prompt_embed", item_id_str + ".pt"
                    )
                    pooled_prompt_embeds_path = os.path.join(
                        args.output_dir, "pooled_prompt_embeds", item_id_str + ".pt"
                    )

                    text_ids_path = os.path.join(
                        args.output_dir, "text_ids", item_id_str + ".pt"
                    )

                    # cond img
                    # image = download_image(ori_image_url[idx])
                    # image = image.resize((1024, 1024)).convert("RGB")

                    masked_image = ossproc.oss_download(seg_mask_url[idx])
                    masked_image = masked_image_proc(masked_image)
                    masked_image = masked_image.resize((512, 512))
                    
                    canny_img = get_canny_edge(masked_image)
                    depth_img = depth_pipe()(masked_image)["depth"].convert("RGB")
                    subject_img = masked_image

                    depth_img_path = os.path.join(args.output_dir, f"depth_img/{item_id_str}.png")
                    canny_img_path = os.path.join(args.output_dir, f"canny_img/{item_id_str}.png")
                    ori_img_path = os.path.join(args.output_dir, f"ori_img/{item_id_str}.png")

                    subject_img.save(ori_img_path)
                    depth_img.save(depth_img_path)
                    canny_img.save(canny_img_path)

                    # save latent
                    torch.save(prompt_embeds[idx], prompt_embed_path)
                    torch.save(pooled_prompt_embeds[idx], pooled_prompt_embeds_path)
                    torch.save(text_ids[idx], text_ids_path)
                    item = {}
                    item["title"] = title
                    item["item_id"] = item_id_str
                    item["pcate_lv3_name"] = pcate_lv3_name
                    item["prompt_embed_path"] = item_id_str + ".pt"
                    item["text_ids"] = item_id_str + ".pt"
                    item["pooled_prompt_embeds_path"] = item_id_str + ".pt"   
                    item["prompt"] = prompts[idx]
                    item["depth_img"] = item_id_str + ".png"
                    item["canny_img"] = item_id_str + ".png"
                    item["org_img"] = item_id_str + ".png"     
                    json_data.append(item)
        except Exception as e:
            print(f"Rank {local_rank} Error: {repr(e)}")
            dist.barrier()
            raise 

        cidx += 1
        print('idx: ', cidx) 

    dist.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    if local_rank == 0:
        # os.remove(latents_json_path)
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "txt2img.json"), "w") as f:
            json.dump(all_json_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/DiffusionRL/data/rl_embeddings',
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--vae_debug", action="store_true")
    parser.add_argument("--prompt_csv", type=str, default="/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/Data/diffusion_rl/csv/train.csv")
    args = parser.parse_args()
    main(args)