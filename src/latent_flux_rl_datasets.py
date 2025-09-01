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

import torch
from torch.utils.data import Dataset
import json
import os
import random
from torchvision import transforms
from PIL import Image 
import numpy as np

class LatentDataset(Dataset):
    def __init__(
        self, json_path, num_latent_t, cfg_rate, device
    ):
        # data_merge_path: video_dir, latent_dir, prompt_embed_dir, json_path
        self.json_path = json_path
        self.cfg_rate = cfg_rate # sample probability
        self.device = device
        self.datase_dir_path = os.path.dirname(json_path)
        #self.video_dir = os.path.join(self.datase_dir_path, "video")
        #self.latent_dir = os.path.join(self.datase_dir_path, "latent")
        self.prompt_embed_dir = os.path.join(self.datase_dir_path, "prompt_embed")
        self.pooled_prompt_embeds_dir = os.path.join(
            self.datase_dir_path, "pooled_prompt_embeds"
        )
        self.text_ids_dir = os.path.join(
            self.datase_dir_path, "text_ids"
        )
        self.ori_img_dir = os.path.join(
            self.datase_dir_path, "ori_img"
        )
        self.canny_img_dir = os.path.join(
            self.datase_dir_path, "canny_img"
        )
        self.depth_img_dir = os.path.join(
            self.datase_dir_path, "depth_img"
        )
        with open(self.json_path, "r") as f:
            self.data_anno = json.load(f)
        # json.load(f) already keeps the order
        # self.data_anno = sorted(self.data_anno, key=lambda x: x['latent_path'])
        self.num_latent_t = num_latent_t
        # just zero embeddings [256, 4096]
        self.uncond_prompt_embed = torch.zeros(256, 4096).to(torch.float32)
        # 256 zeros
        self.uncond_prompt_mask = torch.zeros(256).bool()
        self.lengths = [
            data_item["length"] if "length" in data_item else 1
            for data_item in self.data_anno
        ]

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        # latent_file = self.data_anno[idx]["latent_path"]
        prompt_embed_file = self.data_anno[idx]["prompt_embed_path"]
        pooled_prompt_embeds_file = self.data_anno[idx]["pooled_prompt_embeds_path"]
        text_ids_file = self.data_anno[idx]["text_ids"]
        pcate_lv3_name = self.data_anno[idx]["pcate_lv3_name"]
        title = self.data_anno[idx]["title"]
        ori_img_name = self.data_anno[idx]["org_img"]
        canny_img_name = self.data_anno[idx]["canny_img"]
        depth_img_name = self.data_anno[idx]["depth_img"]
        if random.random() < self.cfg_rate:
            prompt_embed = self.uncond_prompt_embed
        else:
            prompt_embed = torch.load(
                os.path.join(self.prompt_embed_dir, prompt_embed_file),
                map_location="cpu",
                weights_only=True,
            )
            pooled_prompt_embeds = torch.load(
                os.path.join(
                    self.pooled_prompt_embeds_dir, pooled_prompt_embeds_file
                ),
                map_location="cpu",
                weights_only=True,
            )
            text_ids = torch.load(
                os.path.join(
                    self.text_ids_dir, text_ids_file
                ),
                map_location="cpu",
                weights_only=True,
            )
            ori_img_path = os.path.join(self.ori_img_dir, ori_img_name) # (3, 512, 512)
            ori_img = self.to_tensor(Image.open(ori_img_path))

            canny_img_path = os.path.join(self.ori_img_dir, ori_img_name)
            canny_img = self.to_tensor(Image.open(canny_img_path)) # (3, 512, 512)

            depth_img_path = os.path.join(self.depth_img_dir, depth_img_name)
            depth_img = self.to_tensor(Image.open(depth_img_path)) # (3, 512, 512)

            position_delta = np.array([0, 0])

        return prompt_embed, pooled_prompt_embeds, text_ids, self.data_anno[idx]['prompt'], title, pcate_lv3_name, ori_img, canny_img, depth_img, position_delta

    def __len__(self):
        return len(self.data_anno)


def latent_collate_function(batch):
    # return latent, prompt, latent_attn_mask, text_attn_mask
    # latent_attn_mask: # b t h w
    # text_attn_mask: b 1 l
    # needs to check if the latent/prompt' size and apply padding & attn mask
    prompt_embeds, pooled_prompt_embeds, text_ids, caption, title, pcate_lv3_name, ori_img, canny_img, depth_img, position_delta = zip(*batch)
    # attn mask
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    pooled_prompt_embeds = torch.stack(pooled_prompt_embeds, dim=0)
    text_ids = torch.stack(text_ids, dim=0)
    ori_img = torch.stack(ori_img, dim=0)
    canny_img = torch.stack(canny_img, dim=0)
    depth_img = torch.stack(depth_img, dim=0)
    position_delta = tuple(torch.from_numpy(pd) for pd in position_delta)
    position_delta = torch.stack(position_delta, dim=0)

    #latents = torch.stack(latents, dim=0)
    return prompt_embeds, pooled_prompt_embeds, text_ids, caption, title, pcate_lv3_name, ori_img, canny_img, depth_img, position_delta


if __name__ == "__main__":
    dataset = LatentDataset("data/rl_embeddings/videos2caption.json", num_latent_t=28, cfg_rate=0.0)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=latent_collate_function
    )
    for prompt_embed, prompt_attention_mask, caption in dataloader:
        print(
            prompt_embed.shape,
            prompt_attention_mask.shape,
            caption
        )
        import pdb

        pdb.set_trace()