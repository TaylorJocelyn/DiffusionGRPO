import torch
from huggingface_hub import HfApi
import os
# from datetime import datetime

# api = HfApi(token='')
# # api.upload_folder(
# #     folder_path="/mnt/workspace/user/zengdawei.zdw/tmp/OminiControl/runs/train_mulcond_1024_fuse_20250411-021218/ckpt/480000",
# #     repo_id="jinghan1003/FLUX",
# #     repo_type="model",
# # )

# api.upload_folder(
#     folder_path="/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/ImageReward/code",
#     repo_id="jinghan1003/FLUX",
#     repo_type="model",
# )

# current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# print(current_time)

import logging
import datetime

# log_save_dir = os.path.join('/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/DiffusionRL/runs', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
# if not os.path.exists(log_save_dir):
#     os.makedirs(log_save_dir, exist_ok=True)
# log_path = os.path.join(log_save_dir, "run.log")
# logging.basicConfig(
#     format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#     datefmt='%m/%d/%Y %H:%M:%S',
#     level=logging.INFO,
#     handlers=[
#         logging.FileHandler(log_path),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# logger.info('test log...')

params1, params2 = [], []
with open('/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/DiffusionRL/visual_results/flux_transformer_lora.txt', 'r') as file:
    for line in file:
        params1.append(line.strip().split(' ')[0])

with open('/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/DiffusionRL/visual_results/flux_transformer2dmodel.txt', 'r') as file:
    for line in file:
        params2.append(line.strip())

res2 = set(params2)
res1 = set(params1)

cnt = 0
for x in params1:
    if x in res2:
        cnt += 1

print("cnt: ", cnt)
print("len1: ", len(params1))
print("len2: ", len(params2))