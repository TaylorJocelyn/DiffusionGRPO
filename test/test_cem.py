import os
import numpy
import sys
sys.path.append('.')
# from src.rewards import SingleRewardModel
import torch
import torch.nn as nn

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/9 20:29
# @Author  : 晨皋
# @File    : cem_model.py.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from transformers import BertModel, BertTokenizer, SwinModel, SwinConfig
class SingleRewardModel(nn.Module):
    def __init__(self,
                 text_model_name='google-bert/bert-base-uncased',
                 image_model_name='microsoft/swin-base-patch4-window7-224',
                 hidden_size=512,
                 fusion_dim=512,
                 lambda_r=0.1,
                 freeze_encoders=True,
                 cache_dir="/mnt/workspace/user/huixiang.chx/model"):
        """
        单图推理优化的Reward Model
        Args:
            text_model_name: 文本模型名称
            image_model_name: 图像模型名称
            hidden_size: 隐藏层大小
            fusion_dim: 融合特征维度
            lambda_r: 点式损失权重
        """
        super(SingleRewardModel, self).__init__()
        self.lambda_r = lambda_r

        # 1. 文本编码器 (用于商品标题和图像描述)
        self.text_encoder = BertModel.from_pretrained(text_model_name, cache_dir=cache_dir)
        self.text_tokenizer = BertTokenizer.from_pretrained(text_model_name, cache_dir=cache_dir)
        text_dim = self.text_encoder.config.hidden_size

        # 2. 图像编码器 (使用Swin Transformer)
        swin_config = SwinConfig.from_pretrained(image_model_name)
        self.image_encoder = SwinModel(swin_config)
        image_dim = self.image_encoder.config.hidden_size

        # 3. 冻结预训练模型的权重（可选，但推荐作为起点）
        if freeze_encoders:
            print("Freezing encoder weights.")
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # text_dim = 1024
        # image_dim = 1024
        # cate_dim = 1024
        # image_cap_dim = 1024

        # 特征降维层
        self.title_fc = nn.Linear(text_dim, hidden_size)
        self.cate_fc = nn.Linear(text_dim, hidden_size)
        self.caption_fc = nn.Linear(text_dim, hidden_size)
        self.image_fc = nn.Linear(image_dim, hidden_size)

        # 跨模态交互Transformer
        self.cross_modal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8),
            num_layers=3
        )

        # CTR预测头
        self.ctr_predictor = nn.Sequential(
            nn.Linear(hidden_size*4, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def encode_image(self, image):
        """编码单张图像"""
        outputs = self.image_encoder(pixel_values=image)
        # 提取池化后的图像特征
        image_features = outputs.pooler_output
        return image_features

    def encode_text(self, text):
        """编码文本"""
        # 如果文本是列表，则处理整个批次
        if isinstance(text, list):
            inputs = self.text_tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128
            ).to(next(self.parameters()).device)
        else:  # 处理单个字符串
            inputs = self.text_tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128
            ).to(next(self.parameters()).device)

        outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # 取CLS token

    def forward(self, item_title, cate_name, image, image_caption, return_emb=False):
        """
        前向传播（处理单个创意）
        Args:
            item_title: 商品标题 (batch_size,) 或 单个字符串
            image: 图像张量 (batch_size, channels, height, width) 或 单张图像
            caption: 图像描述 (batch_size,) 或 单个字符串
        Returns:
            score: CTR预测分数 (batch_size, 1) 或 单个分数
        """
        # 1. 标题特征提取
        # 标题
        title_feat = self.encode_text(item_title)
        # 2. 图片描述特征提取
        cap_feat = self.encode_text(image_caption)
        # 3. 图像特征提取
        if image.dim() == 3:  # 单张图像 (channels, height, width)
            image = image.unsqueeze(0)  # 添加批次维度
        img_feat = self.encode_image(image)
        # 4. 类目特征提取
        cate_feat = self.encode_text(cate_name)
        # title_feat = item_title
        # cate_feat = cate_name
        # img_feat = image
        # cap_feat = image_caption


        # 4. 特征降维
        title_reduced = self.title_fc(title_feat)
        cate_reduced = self.cate_fc(cate_feat)
        image_reduced = self.image_fc(img_feat)
        caption_reduced = self.caption_fc(cap_feat)

        # 5. 跨模态交互
        # 拼接特征: [标题, 描述, 图像]
        combined = torch.stack([title_reduced, cate_reduced, caption_reduced, image_reduced], dim=1)

        # 6. Transformer处理 (跨模态交互)
        fused_features = self.cross_modal_transformer(combined)

        # 7. 分离特征
        title_fused = fused_features[:, 0, :]
        cate_fused = fused_features[:,1,:]
        caption_fused = fused_features[:, 2, :]
        image_fused = fused_features[:, 3, :]

        # 8. CTR预测
        concat_features = torch.cat([title_fused, cate_fused, caption_fused, image_fused], dim=1)
        ctr_pred = self.ctr_predictor(concat_features)

        # 预测CTR分数
        if return_emb:
            return ctr_pred,concat_features
        else:
            return ctr_pred


class PairRewardModel(nn.Module):
    def __init__(self,
                 # text_model_name='bert-base-uncased',
                 # image_model_name='microsoft/swin-base-patch4-window7-224',
                 hidden_size=512,
                 fusion_dim=512,
                 lambda_r=0.1,
                 rank_weight=0.7,
                 mse_weight=0.3):
        """
        包装类：用于处理图像对训练
        Args:
            text_model_name: 文本模型名称
            hidden_size: 隐藏层大小
            rank_weight: 排序损失权重
            mse_weight: MSE损失权重
        """
        super(PairRewardModel, self).__init__()
        self.rank_weight = rank_weight
        self.mse_weight = mse_weight
        self.core_model = SingleRewardModel(hidden_size=hidden_size, fusion_dim=fusion_dim, lambda_r=lambda_r)

    def forward(self, item_title, cate_name, image1, caption1, image2, caption2):
        """
        处理图像对（用于训练）
        返回两个创意的预测分数
        """
        score1 = self.core_model(item_title,cate_name, image1, caption1)
        score2 = self.core_model(item_title,cate_name, image2, caption2)
        return score1, score2

    def compute_loss(self, score1, score2, ctr1, ctr2):
        """
        计算组合损失
        Args:
            score1, score2: 模型预测的CTR分数 (batch_size, 1)
            ctr1, ctr2: 真实的CTR分数 (batch_size,)
        Returns:
            total_loss: 总损失
            loss_dict: 各损失分量
        """
        # 1. 偏序关系损失 (Pairwise Ranking Loss)
        true_order = torch.sign(ctr1 - ctr2)  # 1: ctr1>ctr2, -1: ctr1<ctr2, 0: 相等
        pred_diff = score1.squeeze() - score2.squeeze()
        rank_loss = F.relu(1 - true_order * pred_diff).mean()

        # 2. MSE回归损失
        mse_loss1 = F.mse_loss(score1.squeeze(), ctr1)
        mse_loss2 = F.mse_loss(score2.squeeze(), ctr2)
        mse_loss = (mse_loss1 + mse_loss2) / 2

        # 3. 组合损失
        total_loss = self.rank_weight * rank_loss + self.mse_weight * mse_loss

        return total_loss, {
            "total_loss": total_loss,
            "rank_loss": rank_loss,
            "mse_loss": mse_loss
        }

def test_cem_v3():
    device = torch.device('cuda:0')
    model_checkpoint_path = '/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/DiffusionRL/models/cem/best_checkpoint_cem_model_v3.pth'
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model = PairRewardModel(rank_weight=0.7, mse_weight=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    print("Model weights loaded successfully.")

if __name__ == '__main__':
    test_cem_v3()