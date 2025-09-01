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

import os 
import datetime
import logging
import copy
import sys
sys.path.append('.')
import argparse
import math
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torchviz import make_dot
from utils.load import load_lora_weights
from pathlib import Path
from utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from utils.data_utils import get_img_base64, logger
from utils.env_utils import get_rank
from utils.communications_flux import sp_parallel_dataloader_wrapper
# from utils.validation import log_validation
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, fully_shard
from torch.distributed.fsdp import fully_shard, FSDPModule

from torch.utils.data.distributed import DistributedSampler
from utils.dataset_utils import LengthGroupedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
# from utils.load import load_transformer
from diffusers.optimization import get_scheduler
# from utils import check_min_version
from latent_flux_rl_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from utils.checkpoint import (
    save_checkpoint,
    save_lora_checkpoint,
)
from utils.logging_ import main_print
from rewards import load_reward_models, load_ctr_model, load_hpsv2_model, load_ecp_model
from vlm_api.tongyi_request import qwenvl_creative_advertising_understand
import cv2
from diffusers.image_processor import VaeImageProcessor
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.31.0")
import time

from collections import deque
import numpy as np
from einops import rearrange
import torch.distributed as dist
from torch.nn import functional as F
from typing import List
from PIL import Image
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL
from flux.transformer_ import tranformer_forward
from flux.block import block_forward, single_block_forward, attn_forward
from flux.condition import Condition
from flux.pipeline_tools import encode_cond_images, prepare_text_input

model_weights = {
    "flux": {
        "model_cache_dir": "/root/autodl-tmp/.cache/hub",
        "model_path": "/root/autodl-tmp/.cache/hub/models--black-forest-labs--FLUX.1-dev",
        "bfl_repo": "black-forest-labs/FLUX.1-dev",
    },
    # "ecomini": {
    #     "model_cache_dir": "/mnt/workspace/user/zengdawei.zdw/tmp/OminiControl/runs/train_mulcond_1024_fuse_20250411-021218",
    #     "ckpt_path": "ckpt/120000",
    #     "lora_name": "pytorch_lora_weights.safetensors"
    # },
    "ecomini": {
        "model_cache_dir": "/root/autodl-tmp/DiffusionGRPO/models",
        "ckpt_path": "lora",
        "lora_name": "pytorch_lora_weights.safetensors"
    },
    "depth-estimation": {
        "model_repo": "LiheYoung/depth-anything-small-hf"
    },
    "lama_inpainting": {
        "model_cache_dir": "/mnt/workspace/user/zengdawei.zdw/image_aigc_service_icanvas/models/damo/cv_fft_inpainting_lama"
    }
}

def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)

    
def load_flux_transformer_(device):
    flux_pipe = FluxPipeline.from_pretrained(
            pretrained_model_name_or_path=model_weights["flux"].get("bfl_repo"),
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16
        ).to(device)

    root_path = model_weights["ecomini"].get("model_cache_dir")
    lora_path = os.path.join(root_path, model_weights["ecomini"].get("ckpt_path"))
    lora_name = model_weights["ecomini"].get("lora_name", "")

    flux_pipe.load_lora_weights(
        lora_path,
        weight_name=lora_name,
    )

    # flux_pipe.fuse_lora(lora_scale=1.0)
    # flux_pipe.unload_lora_weights()
    
    # merge lora weight to base layers
    # flux_pipe.fuse_lora(lora_scale=1.0)
    # flux_pipe.unload_lora_weights()

    return flux_pipe

def load_flux_transformer():

    transformer = FluxTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path=model_weights["flux"].get("bfl_repo"),
            cache_dir=args.cache_dir,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
    )
    
    root_path = model_weights["ecomini"].get("model_cache_dir")
    lora_path = os.path.join(root_path, model_weights["ecomini"].get("ckpt_path"))
    lora_name = model_weights["ecomini"].get("lora_name", "")

    load_lora_weights(transformer, lora_path, logger=logger, weight_name=lora_name)

    return transformer

# def set_params_trainable_(flux_pipe, transformer):
    # try:
    #     with open('/root/autodl-tmp/DiffusionGRPO/debug/params.txt', 'w') as f:
    #         for name, param in transformer.named_parameters():
    #             flag = "True" if param.requires_grad else "False"
    #             f.write(name + ' ' + flag + '\n')
    #             # param.requires_grad = False
    #             # param.data = param.data.to(torch.float32)

    #     with open('/root/autodl-tmp/DiffusionGRPO/debug/lora_params.txt', 'w') as f:
    #         for name, param in transformer.named_parameters():
    #             if 'lora' in name:
    #                 f.write(name + '\n')

    #     flux_pipe.fuse_lora(lora_scale=1.0)
    #     flux_pipe.unload_lora_weights()
    #     print("success save fuse lora")

    #     flux_pipe_copied = copy.deepcopy(flux_pipe)

    #     transformer_copied = flux_pipe_copied.transformer

    #     del flux_pipe_copied
    #     torch.cuda.empty_cache()

    #     with open('/root/autodl-tmp/DiffusionGRPO/debug/fused_lora_params.txt', 'w') as f:
    #         for name, param in transformer.named_parameters():
    #             flag = "True" if param.requires_grad else "False"
    #             f.write(name + ' ' + flag + '\n')
    #             # param.requires_grad = False
    #             # param.data = param.data.to(torch.float32)

    #     with open('/root/autodl-tmp/DiffusionGRPO/debug/unload_lora_params.txt', 'w') as f:
    #         for name, param in transformer.named_parameters():
    #             if 'lora' in name:
    #                 f.write(name + '\n')
    #     print("success save fuse lora")

    # except Exception as e:
        
    #     print(f"error: {e}")

    # finally:
    #     sys.exit("Program terminated after processing parameters.")

    # for name, param in transformer.named_parameters():
    #     param.requires_grad = True

def set_params_trainable__(flux_pipe):
    t_pre = copy.deepcopy(flux_pipe.transformer)

    flux_pipe.fuse_lora(lora_scale=1.0)
    flux_pipe.unload_lora_weights()
    t_post = flux_pipe.transformer
    
    # set layer
    t_post.x_embedder_hs = t_pre.x_embedder.base_layer
    for index_block, block in enumerate(t_post.transformer_blocks):
        block.norm1_hs = t_pre.transformer_blocks[index_block].norm1
        block.norm1_hs.linear = t_pre.transformer_blocks[index_block].norm1.linear.base_layer

        block.attn.to_q_hs = t_pre.transformer_blocks[index_block].attn.to_q.base_layer
        block.attn.to_k_hs = t_pre.transformer_blocks[index_block].attn.to_k.base_layer
        block.attn.to_v_hs = t_pre.transformer_blocks[index_block].attn.to_v.base_layer

        block.attn.to_out_hs = t_pre.transformer_blocks[index_block].attn.to_out
        block.attn.to_out_hs[0] =  t_pre.transformer_blocks[index_block].attn.to_out[0].base_layer

        block.ff_hs = t_pre.transformer_blocks[index_block].ff
        block.ff_hs.net[2] = t_pre.transformer_blocks[index_block].ff.net[2].base_layer

    for index_block, block in enumerate(t_post.single_transformer_blocks):
        block.norm_hs = t_pre.single_transformer_blocks[index_block].norm
        block.norm_hs.linear = t_pre.single_transformer_blocks[index_block].norm.linear.base_layer
        
        block.attn.to_q_hs = t_pre.single_transformer_blocks[index_block].attn.to_q.base_layer
        block.attn.to_k_hs = t_pre.single_transformer_blocks[index_block].attn.to_k.base_layer
        block.attn.to_v_hs = t_pre.single_transformer_blocks[index_block].attn.to_v.base_layer

        block.proj_mlp_hs = t_pre.single_transformer_blocks[index_block].proj_mlp.base_layer

        block.proj_out_hs = t_pre.single_transformer_blocks[index_block].proj_out.base_layer


    # set transformer forward
    for name, param in t_post.named_parameters():
        param.requires_grad = False
        # if "_hs" in name:
        #     param.requires_grad = True

    t_post.forward = tranformer_forward.__get__(t_post) 

    for index_block, block in enumerate(t_post.transformer_blocks):
        block.attn.forward = attn_forward.__get__(block.attn)
        block.forward = block_forward.__get__(block) 

    for index_block, block in enumerate(t_post.single_transformer_blocks):
        block.attn.forward = attn_forward.__get__(block.attn)
        block.forward = single_block_forward.__get__(block)

    del flux_pipe, t_pre
    torch.cuda.empty_cache()

    

    return t_post

def set_params_trainable(transformer):
    # set transformer forward
    for name, param in transformer.named_parameters():
        param.requires_grad = True
    #     if "lora_" in name:
    #         param.requires_grad = True

    transformer.forward = tranformer_forward.__get__(transformer) 

    for index_block, block in enumerate(transformer.transformer_blocks):
        block.attn.forward = attn_forward.__get__(block.attn)
        block.forward = block_forward.__get__(block) 

    for index_block, block in enumerate(transformer.single_transformer_blocks):
        block.attn.forward = attn_forward.__get__(block.attn)
        block.forward = single_block_forward.__get__(block)


def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
        

    if grpo:
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = (
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
        )
        - math.log(std_dev_t)- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean,pred_original_sample



def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

def run_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        vae,
        transformer,
        encoder_hidden_states, 
        pooled_prompt_embeds, 
        ori_img,
        canny_img,
        depth_img,
        text_ids,
        image_ids,
        position_delta, 
        grpo_sample,
    ):
    if grpo_sample:
        device = transformer.device
        h, w = args.h, args.w
        model_config = {'union_cond_attn': True, 'add_cond_attn': False, 'latent_lora': False}
        # prepare input
        condition_types = ['canny', 'depth', 'subject']
        conditions = [canny_img, depth_img, ori_img]
        
        position_delta = position_delta[0]
        
        # Prepare inputs
        with torch.no_grad():
            condition_latents, condition_ids = list(), list()
            
            for cond_ in conditions:
                # Prepare conditions
                cond_latents, cond_ids = encode_cond_images(
                    vae, 
                    cond_,
                    height=h,
                    width=w,
                    device=device,
                    dtype=transformer.dtype
                    ) # (1, 1024, 64) (1024, 3)

                # Add position delta
                cond_ids[:, 1] += position_delta[0]
                cond_ids[:, 2] += position_delta[1]
                
                condition_latents.append(cond_latents)
                condition_ids.append(cond_ids)

            condition_latents = torch.cat(condition_latents, axis=1) # (1, 3072, 64)
            condition_ids = torch.cat(condition_ids, axis=0) # (3072, 3)

            # Prepare condition type
            condition_type_ids = torch.tensor(
                [
                    Condition.get_type_id(condition_type)
                    for condition_type in condition_types
                ]
            ).to(device) 

            condition_type_ids = (
                torch.ones_like(condition_ids[:, 0]).reshape(len(condition_types), -1) * condition_type_ids.unsqueeze(1)
            ).reshape(-1,1) # (3072, 1)

        all_latents = [z] # (1, 4096, 64)
        all_log_probs = []

        for i in progress_bar:  # Add progress bar
            B = encoder_hidden_states.shape[0]
            sigma = sigma_schedule[i] # [0, 1]
            timestep_value = int(sigma * 1000) # [0, 1000]
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long) # torch.Size[1]
            transformer.eval()
            with torch.autocast("cuda", torch.bfloat16):
                pred = transformer(
                    # Model config
                    model_config=model_config,
                    # Inputs of the condition (new feature)
                    condition_latents=condition_latents, # (1, 3072, 64)
                    condition_ids=condition_ids, # (3072, 3)
                    condition_type_ids=condition_type_ids, # (3072, 1)
                    # Inputs to the original transformer
                    hidden_states=z, # (1, 4096, 64)
                    timestep=timesteps/1000, # torch.Size(1)
                    guidance=torch.tensor(
                        [3.5],
                        device=z.device,
                        dtype=torch.bfloat16
                    ),
                    pooled_projections=pooled_prompt_embeds, # (1, 768)
                    encoder_hidden_states=encoder_hidden_states, # (1, 512, 4096)
                    txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1), # (512, 3)
                    img_ids=image_ids, # (4096, 3)
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                # pred = transformer(
                #     hidden_states=z,
                #     encoder_hidden_states=encoder_hidden_states,
                #     timestep=timesteps/1000,
                #     guidance=torch.tensor(
                #         [3.5],
                #         device=z.device,
                #         dtype=torch.bfloat16
                #     ),
                #     txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1), # B, L
                #     pooled_projections=pooled_prompt_embeds,
                #     img_ids=image_ids,
                #     joint_attention_kwargs=None,
                #     return_dict=False,
                # )[0]
            z, pred_original, log_prob = flux_step(pred, z.to(torch.bfloat16), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True) # log_prob = log[p(x_{t-1})]
            z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)

        latents = pred_original
        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64) -> (batch_size, num_steps + 1, 4096, 64)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1) -> (batch_size, num_steps)

        torch.cuda.empty_cache()

        return z, latents, all_latents, all_log_probs, condition_latents, condition_ids, condition_type_ids

        
def grpo_one_step(
            args,
            latents,
            pre_latents,
            encoder_hidden_states, 
            pooled_prompt_embeds, 
            text_ids,
            image_ids,
            condition_latents,
            condition_ids,
            condition_type_ids,
            transformer,
            timesteps,
            i,
            sigma_schedule,
):
    B = encoder_hidden_states.shape[0]
    model_config = {'union_cond_attn': True, 'add_cond_attn': False, 'latent_lora': False}
    transformer.train()
    with torch.autocast("cuda", torch.bfloat16):
        pred = transformer(
            # Model config
            model_config=model_config,
            # Inputs of the condition (new feature)
            condition_latents=condition_latents,
            condition_ids=condition_ids,
            condition_type_ids=condition_type_ids,
            # Inputs to the original transformer
            hidden_states=latents,
            timestep=timesteps/1000, 
            guidance=torch.tensor(
                [3.5],
                device=latents.device,
                dtype=torch.bfloat16
            ),
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1),
            img_ids=image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]   
        # pred = transformer(
        #     hidden_states=latents,
        #     encoder_hidden_states=encoder_hidden_states,
        #     timestep=timesteps/1000,
        #     guidance=torch.tensor(
        #         [3.5],
        #         device=latents.device,
        #         dtype=torch.bfloat16
        #     ),
        #     txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1), # B, L
        #     pooled_projections=pooled_prompt_embeds,
        #     img_ids=image_ids.squeeze(0),
        #     joint_attention_kwargs=None,
        #     return_dict=False,
        # )[0]
    z, pred_original, log_prob = flux_step(pred, latents.to(torch.bfloat16), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.bfloat16), grpo=True, sde_solver=True)
    return log_prob

def sample_reference_model(
    args,
    device, 
    vae,
    transformer,
    encoder_hidden_states, 
    pooled_prompt_embeds, 
    text_ids,
    caption,
    title,
    pcate_lv3_name,
    ori_img, 
    canny_img, 
    depth_img, 
    position_delta,
    ctr_model, 
    hpsv2_model, 
    hpsv2_preprocess_val, 
    hpsv2_processor, 
    ecp_model
):
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_rewards = {"ctr_reward":[], "ecp_reward":[], "t2i_reward":[], "hps_reward":[]}  
    all_image_ids = []
    all_condition_latents = []
    all_condition_ids = [] 
    all_condition_type_ids = []
    if args.init_same_noise:
        input_latents = torch.randn(
                (1, IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
                device=device,
                dtype=torch.bfloat16,
            )

    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_pooled_prompt_embeds = pooled_prompt_embeds[batch_idx]
        batch_text_ids = text_ids[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]
        batch_ori_img = ori_img[batch_idx]
        batch_depth_img = depth_img[batch_idx]
        batch_canny_img = canny_img[batch_idx]
        batch_position_delta = position_delta[batch_idx]
        if not args.init_same_noise:
            input_latents = torch.randn(
                    (len(batch_idx), IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
                    device=device,
                    dtype=torch.bfloat16,
                )
        input_latents_new = pack_latents(input_latents, len(batch_idx), IN_CHANNELS, latent_h, latent_w)
        image_ids = prepare_latent_image_ids(len(batch_idx), latent_h // 2, latent_w // 2, device, torch.bfloat16)
        grpo_sample=True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        with torch.no_grad():
            z, latents, batch_latents, batch_log_probs, batch_condition_latents, batch_condition_ids, batch_condition_type_ids = run_sample_step(
                args,
                input_latents_new,
                progress_bar,
                sigma_schedule,
                vae,
                transformer,
                batch_encoder_hidden_states,
                batch_pooled_prompt_embeds,
                batch_ori_img,
                batch_canny_img,
                batch_depth_img,
                batch_text_ids,
                image_ids,
                batch_position_delta,
                grpo_sample,
            )
        
        all_image_ids.append(image_ids)
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        all_condition_latents.append(batch_condition_latents)
        all_condition_ids.append(batch_condition_ids)
        all_condition_type_ids.append(batch_condition_type_ids)
        vae.enable_tiling()
        
        image_processor = VaeImageProcessor(16)
        rank = dist.get_rank()

        
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents = unpack_latents(latents, h, w, 8)
                latents = (latents / 0.3611) + 0.1159
                image = vae.decode(latents, return_dict=False)[0]
                decoded_image = image_processor.postprocess(
                image)

        main_print(f"--> decode image and save to log dir...", logger)
        
        if dist.get_rank() <= 0:
            file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
            sample_save_path = os.path.join(os.path.dirname(file_handler.baseFilename), "samples")
            if not os.path.exists(sample_save_path):
                os.makedirs(sample_save_path)
            decoded_image[0].save(os.path.join(sample_save_path, f"image_{index}.png"))

        if args.use_aesthetics_reward:
            with torch.no_grad():
                image_path = decoded_image[0]
                image = hpsv2_preprocess_val(image_path).unsqueeze(0).to(device=device, non_blocking=True)
                # Process the prompt
                text = hpsv2_processor([batch_caption[0]]).to(device=device, non_blocking=True)
                # Calculate the HPS
                with torch.amp.autocast('cuda'):
                    outputs = hpsv2_model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image)
                all_rewards["hps_reward"].append(hps_score)

        if args.use_ecp_reward:
            with torch.no_grad():
                image = decoded_image[0]
                prompt = batch_caption[0].to(device=device, non_blocking=True)
                ecp_score = ecp_model.score(prompt, image)
                all_rewards["ecp_reward"].append(ecp_score)

        if args.use_ctr_reward:
            with torch.no_grad():
                image = decoded_image[0]
                prompt = batch_caption[0].to(device=device, non_blocking=True)
                ctr_score = ctr_model.core_model(title, pcate_lv3_name, image, prompt)
                all_rewards["ctr_reward"].append(ctr_score)

        if args.use_t2i_align_reward:
            import bert_score
            image = decoded_image[0]
            prompt = batch_caption[0].to(device=device, non_blocking=True)
            candidate = qwenvl_creative_advertising_understand(image, prompt)
            P, R, F1 = bert_score.score(candidate, caption, lang="en", verbose=True)
            all_rewards["t2i_reward"].append(F1)
        

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_image_ids = torch.stack(all_image_ids, dim=0)
    all_condition_latents = torch.stack(all_condition_latents, dim=0)
    all_condition_ids = torch.stack(all_condition_ids, dim=0)
    all_condition_type_ids = torch.stack(all_condition_type_ids, dim=0)
    for reward_type in all_rewards:
        if all_rewards[reward_type]:
            all_rewards[reward_type] = torch.cat(all_rewards[reward_type], dim=0) # torch.Size(12)
    
    return all_rewards, all_latents, all_log_probs, sigma_schedule, all_image_ids, all_condition_latents, all_condition_ids, all_condition_type_ids


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def train_one_step(
    args,
    device,
    vae,
    transformer,
    ctr_model, 
    hpsv2_model, 
    hpsv2_preprocess_val, 
    hpsv2_processor, 
    ecp_model,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    max_grad_norm
):
    total_loss = 0.0
    optimizer.zero_grad()
    (
        encoder_hidden_states, 
        pooled_prompt_embeds, 
        text_ids,
        caption,
        title, 
        pcate_lv3_name,
        ori_img, 
        canny_img, 
        depth_img, 
        position_delta
    ) = next(loader)
    #device = latents.device
    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        pooled_prompt_embeds = repeat_tensor(pooled_prompt_embeds)
        text_ids = repeat_tensor(text_ids)
        ori_img = repeat_tensor(ori_img)
        depth_img = repeat_tensor(depth_img)
        canny_img = repeat_tensor(canny_img)
        position_delta = repeat_tensor(position_delta)

        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

        if isinstance(title, str):
            title = [title] * args.num_generations
        elif isinstance(title, list):
            title = [item for item in title for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported title type: {type(title)}")

        if isinstance(pcate_lv3_name, str):
            pcate_lv3_name = [pcate_lv3_name] * args.num_generations
        elif isinstance(pcate_lv3_name, list):
            pcate_lv3_name = [item for item in pcate_lv3_name for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported pcate_lv3_name type: {type(pcate_lv3_name)}")

    rewards, all_latents, all_log_probs, sigma_schedule, all_image_ids, all_condition_latents, all_condition_ids, all_condition_type_ids = sample_reference_model(
            args,
            device, 
            vae,
            transformer,
            encoder_hidden_states, 
            pooled_prompt_embeds, 
            text_ids,
            caption,
            title,
            pcate_lv3_name,
            ori_img, 
            canny_img, 
            depth_img, 
            position_delta,
            ctr_model, 
            hpsv2_model, 
            hpsv2_preprocess_val, 
            hpsv2_processor, 
            ecp_model,
        )
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)] # (12, 16) list2d
    device = all_latents.device
    timesteps = torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long) # (12, 16)

    group_rewards = {}
    for reward_type in rewards:
        group_rewards[reward_type] = rewards[reward_type]
        if isinstance(rewards[reward_type], torch.Tensor):
            group_rewards[reward_type] = rewards[reward_type].to(torch.float32)

    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[
            :, :-1
        ][:, :-1],  # each entry is the latent before timestep t
        "next_latents": all_latents[
            :, 1:
        ][:, :-1],  # each entry is the latent after timestep t
        "condition_latents": all_condition_latents,
        "condition_ids": all_condition_ids,
        "condition_type_ids": all_condition_type_ids,
        "log_probs": all_log_probs[:, :-1],
        "rewards": group_rewards,
        "image_ids": all_image_ids,
        "text_ids": text_ids,
        "encoder_hidden_states": encoder_hidden_states, # (12, 512, 4096)
        "pooled_prompt_embeds": pooled_prompt_embeds, # (12, 768)
    }

    gathered_reward = {}

    for reward_type in samples["rewards"]:
        gathered_reward[reward_type] = samples["rewards"][reward_type]
        if isinstance(samples["rewards"][reward_type], torch.Tensor):
            gathered_reward[reward_type] = gather_tensor(samples["rewards"][reward_type])

    if get_rank()==0:
        print("gathered_reward", gathered_reward)
        file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
        with open(os.path.join(os.path.dirname(file_handler.baseFilename), 'reward.txt'), 'a') as f: 
            reward_str = ''
            for reward_type in gathered_reward:
                if isinstance(gathered_reward[reward_type], torch.Tensor):
                    reward_str += reward_type + ': ' + f'{gathered_reward[reward_type].mean().item():.4f}' + '    '
                else:
                    reward_str += reward_type + ': null    '
            f.write(reward_str + '\n')

    # 计算 advantage
    samples["advantages"] = {}
    if args.use_group:
        for reward_type in samples["rewards"]:
            if not isinstance(samples["rewards"][reward_type], torch.Tensor):
                continue
            n = len(samples["rewards"][reward_type]) // (args.num_generations)
            advantages = torch.zeros_like(samples["rewards"][reward_type])
            
            for i in range(n):
                start_idx = i * args.num_generations
                end_idx = (i + 1) * args.num_generations
                group_rewards = samples["rewards"][reward_type][start_idx:end_idx]
                group_mean = group_rewards.mean()
                group_std = group_rewards.std() + 1e-8
                advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
            
            samples["advantages"][reward_type] = advantages
    else:
        for reward_type in samples["rewards"]:
            if not isinstance(samples["rewards"][reward_type], torch.Tensor):
                continue

            advantages = (samples["rewards"][reward_type] - gathered_reward[reward_type].mean())/(gathered_reward[reward_type].std()+1e-8)
            samples["advantages"][reward_type] = advantages

    samples["final_advantages"] = samples["advantages"]["hps_reward"]

    perms = torch.stack(
        [
            torch.randperm(len(samples["timesteps"][0])) # batch_size=num_generations (batch_size, num_steps)
            for _ in range(batch_size)
        ]
    ).to(device) 
    for key in ["timesteps", "latents", "next_latents", "log_probs"]: # 每个样本的列重排
        samples[key] = samples[key][
            torch.arange(batch_size).to(device) [:, None],
            perms,
        ]

    samples_batched = {}
    for k, v in samples.items():
        if k in ('rewards', 'advantages'):
            continue
        elif k in ('image_ids', 'condition_latents', 'condition_ids', 'condition_type_ids'):
            samples_batched.update({k: v})
        else:
            samples_batched.update({k: v.unsqueeze(1)})

    # dict of lists -> list of dicts for easier iteration
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ] # len=batch_size
    
    train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction)
    for i, sample in list(enumerate(samples_batched_list)):
        for _ in range(train_timesteps):
            clip_range = args.clip_range
            adv_clip_max = args.adv_clip_max
            new_log_probs = grpo_one_step(
                args,
                sample["latents"][:,_],
                sample["next_latents"][:,_],
                sample["encoder_hidden_states"],
                sample["pooled_prompt_embeds"],
                sample["text_ids"],
                sample["image_ids"],
                sample["condition_latents"],
                sample["condition_ids"],
                sample["condition_type_ids"],
                transformer,
                sample["timesteps"][:,_], # all_batch, _ step
                perms[i][_],
                sigma_schedule,
            )

            advantages = torch.clamp(
                sample["final_advantages"],
                -adv_clip_max,
                adv_clip_max,
            )

            ratio = torch.exp(new_log_probs - sample["log_probs"][:,_])

            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * train_timesteps)

            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()

        print("idx i: ", i)
        if (i+1)%args.gradient_accumulation_steps==0:
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if dist.get_rank()%8==0:
            # print("sample['rewards'] ", type(sample['rewards']))
            # main_print(f"reward: {sample['rewards'].item():4f}", logger)
            main_print(f"ratio: {ratio.item():.2f}", logger)
            main_print(f"advantage {sample['final_advantages'].item():.4f}", logger)
            main_print(f"final loss: {loss.item():.4f}", logger)
        dist.barrier()
    return total_loss, grad_norm.item()


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    world_size = int(os.environ["WORLD_SIZE"])
    timeout = datetime.timedelta(minutes=600)
    dist.init_process_group("nccl", timeout=timeout)
    local_rank = dist.get_rank()
    rank = dist.get_rank()
    print("local-rank ", local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    device = torch.device(f'cuda:{device}')
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # load reward models
    ctr_model, hpsv2_model, hpsv2_preprocess_val, hpsv2_processor, ecp_model = None, None, None, None, None
    if args.use_ctr_reward:
        ctr_model = load_ctr_model(args, device, logger)

    if args.use_aesthetics_reward:
        hpsv2_model, hpsv2_preprocess_val, hpsv2_processor = load_hpsv2_model(args, device, logger)
    
    if args.use_ecp_reward:
        ecp_model = load_ecp_model(args, device, logger)

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}", logger)
    # keep the master weight to float32
    # Load FLUX Transformer
    transformer = load_flux_transformer()
    # transformer = FluxTransformer2DModel.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         subfolder="transformer",
    #         torch_dtype = torch.bfloat16
    # )

    set_params_trainable(transformer)

    # with open('/root/autodl-tmp/DiffusionGRPO/debug/lora_trans.txt', 'w') as file:
    #     for name, param in transformer.named_parameters():
    #         flag = 'True' if param.requires_grad else 'False'
    #         file.write(name + ' ' + flag +'\n')
    
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    
    transformer = FSDP(transformer, **fsdp_kwargs,) # 60032 M
    # for name, module in transformer.named_modules():
    #     if('lora_A' in name or 'lora_B' in name) and ('default_0' in name):
    #         fully_shard(module)

    # fully_shard(transformer)
    # transformer.to_empty(device='cuda')


    # for layer in transformer.layers:
    #     fully_shard(layer, **fsdp_kwargs)
    # fully_shard(transformer, **fsdp_kwargs)
    # x_embedder_w = transformer.x_embedder.base_layer.weight.shape
    # x_embedder_a = transformer.x_embedder.lora_A.default_0.weight.shape
    # x_embedder_b = transformer.x_embedder.lora_B.default_0.weight.shape
    # print("x_embedder_w: ", x_embedder_w, " x_embedder_a: ", x_embedder_a, " x_embedder_b: ", x_embedder_b)

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
    ).to(device)

    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}", logger
    )
    # Load the reference model
    main_print(f"--> model loaded", logger)

    # Set model as trainable.
    transformer.train()

    noise_scheduler = None

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    # with open('/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/DiffusionRL/debug/trainable_params.txt', 'w') as f:
    #     for name, param in transformer.named_parameters():
    #         if param.requires_grad:
    #             f.write(name + '\n')

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}", logger)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg, device)
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )   
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    allocated_memory = torch.cuda.memory_allocated()  # 已分配显存
    reserved_memory = torch.cuda.memory_reserved()  # 预留的显存
    print(f"After Load Data Allocated Memory: {allocated_memory / 1024**2:.2f} MB")
    print(f"Reserved Memory: {reserved_memory / 1024**2:.2f} MB")


    #vae.enable_tiling()

    if rank <= 0:
        project = "flux"
        wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * world_size
        * args.gradient_accumulation_steps # 每个 gpu 都执行 steps 步骤
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****", logger)
    main_print(f"  Num examples = {len(train_dataset)}", logger)
    main_print(f"  Dataloader size = {len(train_dataloader)}", logger)
    main_print(f"  Resume training from step {init_steps}", logger)
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}", logger)
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}", logger
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}", logger)
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}", logger)
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B", logger
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}", logger)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)

    # The number of epochs 1 is a random value; you can also set the number of epochs to be two.
    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch

        for step in range(init_steps+1, args.max_train_steps+1):
            start_time = time.time()
            if step % args.checkpointing_steps == 0:
                save_checkpoint(transformer, rank, args.output_dir,
                                step, epoch)

                dist.barrier()
            loss, grad_norm = train_one_step(
                args,
                device, 
                vae,
                transformer,
                ctr_model, 
                hpsv2_model, 
                hpsv2_preprocess_val, 
                hpsv2_processor, 
                ecp_model,
                optimizer,
                lr_scheduler,
                loader,
                noise_scheduler,
                args.max_grad_norm,
            )
    
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                    },
                    step=step,
                )

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #GRPO training
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="image height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="image width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,   
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=1.0,   
        help="the global loss should be divided by",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether compute advantages for each prompt",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--use_hpsv2",
        action="store_true",
        default=False,
        help="whether use hpsv2 as reward model",
    )
    parser.add_argument(
        "--use_pickscore",
        action="store_true",
        default=False,
        help="whether use pickscore as reward model",
    )
    parser.add_argument(
        "--ignore_last",
        action="store_true",
        default=False,
        help="whether ignore last step of mdp",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=False,
        help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep downsample ratio",
    )
    parser.add_argument(
        "--clip_range",
        type = float,
        default=1e-4,
        help="clip range for grpo",
    )
    parser.add_argument(
        "--adv_clip_max",
        type = float,
        default=5.0,
        help="clipping advantage",
    )
    parser.add_argument(
        "--log_dir",
        type = str,
        default='runs',
        help="log save dir"
    )
    parser.add_argument(
        "--use_ctr_reward",
        action="store_true",
        default=False,
        help="whether use the ctr score as reward"
    )
    parser.add_argument(
        "--use_ecp_reward",
        action="store_true",
        help="whether use the ecp score as e-commerce preference reward"
    )
    parser.add_argument(
        "--use_aesthetics_reward",
        action="store_true",
        help="whether use the hpsv2 score as aesthetics reward"
    )
    parser.add_argument(
        "--use_t2i_align_reward",
        action="store_true",
        help="whether use the vlm score as t2i alignment reward"
    )
    args = parser.parse_args()
    main(args)