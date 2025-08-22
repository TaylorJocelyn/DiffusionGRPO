import torch
import os
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
# from src.core.utils import *
from utils.odps_utils import read_odps_table

def filter_data():
    input_table = "odps://ae_ai_intern_dev/tables/ae_p4p_creative_icanvas_data_for_rl_annot_base_detail"
    selected_cols = "item_id,item_title,pcate_lv1_id,pcate_lv1_name,pcate_lv2_id,pcate_lv2_name,pcate_lv3_id,pcate_lv3_name,ori_image_url,seg_mask_url,prompt,prompt_id,negative_prompt,item_image_content_id,gen_type"
    data_info = read_odps_table(input_table, selected_cols=selected_cols)
    df = pd.DataFrame(data_info, columns=[item for item in selected_cols.split(',')])
    tot_cnt, _ = df.shape

    ratio_train, ratio_test = tot_cnt * 1.0 / 12000.0, tot_cnt * 1.0 / 1200.0
    cnt = {}

    group_sz = df.groupby('pcate_lv1_id').size()
    group_sz = group_sz.to_dict()

    sums = 0
    for i in group_sz:
        sums += group_sz[i]
    
    train_cnt, test_cnt = {}, {}

    for i in group_sz:
        cur_cnt = group_sz[i]
        if cur_cnt < 5:
            train_cnt[i] = cur_cnt
        elif cur_cnt < 20:
            train_cnt[i] = int(cur_cnt / 3)
        else:
            train_cnt[i] = int(cur_cnt / ratio_train)

    for i in group_sz:
        test_cnt[i] = int(group_sz[i] / ratio_test)

    train_data, test_data = [], []

    for group_id in group_sz:
        group_data = df[df['pcate_lv1_id'] == group_id]
        remaining_group = group_data
        if train_cnt[group_id] > 0:
            sampled_group = group_data.sample(n=train_cnt[group_id], random_state=42) 
            train_data.append(sampled_group)
            remaining_group = group_data.drop(sampled_group.index)

        if test_cnt[group_id] > 0 and len(remaining_group) > 0:
            if len(remaining_group) >= test_cnt[group_id]:
                sampled_group2 = remaining_group.sample(n=test_cnt[group_id], random_state=42)
                test_data.append(sampled_group2)
            else:
                test_data.append(remaining_group)

    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    csv_save_dir = "/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/Data/diffusion_rl/csv"
    train_df.to_csv(os.path.join(csv_save_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(csv_save_dir, "test.csv"), index=False)

    # flag = train_df[train_df['item_id'].isin(test_df['item_id'])].any()

    # process_data()

if __name__ == '__main__':
    filter_data()