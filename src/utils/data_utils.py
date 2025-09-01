import sys
from PIL import Image
import numpy as np
import random
import json
import os
import time
import ast
import base64
from tqdm import tqdm
import requests
import hashlib
from io import BytesIO


# import common_io
# import oss2
from datetime import datetime
import logging

log_dir = './runs'
log_save_dir = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
if not os.path.exists(log_save_dir):
    os.makedirs(log_save_dir, exist_ok=True)
log_path = os.path.join(log_save_dir, "run.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False 
if not logger.handlers: 
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            '%m/%d/%Y %H:%M:%S')
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt)
    sh = logging.StreamHandler();       sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
logger.info('init logger...')

# def read_odps_table(table_path, selected_cols="", slice_id=0, slice_count=1):
#     reader = common_io.table.TableReader(table_path,
#                                          selected_cols=selected_cols,
#                                          slice_id=slice_id,
#                                          slice_count=slice_count)
#     records_cnt = reader.get_row_count()
#     records = reader.read(records_cnt, allow_smaller_final_batch=True)
#     reader.close()
#     return records


# class ossProcessor(object):
#     def __init__(self, bucket_name):
#         # self.auth = oss2.Auth('', '')
#         # self.endpoint = 'oss-ap-southeast-1-internal.aliyuncs.com'

#         self.auth = oss2.Auth('', '')
#         self.endpoint = 'oss-ap-southeast-1-internal.aliyuncs.com'

#         self.expired_time = 777600
#         self.bucket = oss2.Bucket(self.auth, self.endpoint, bucket_name)

#     def oss_upload(self, file_byte, path):
#         self.bucket.put_object(path, file_byte)
#         sign_url = self.bucket.sign_url("GET", path, self.expired_time)
#         return sign_url

#     def oss_download_stream(self, file_path):
#         object_stream = self.bucket.get_object(file_path)
#         return object_stream

#     def oss_download_file(self, file_path, out_path):
#         self.bucket.get_object_to_file(file_path, out_path)

#     def oss_download(self, file_path):
#         object_stream = self.bucket.get_object(file_path)
#         image = Image.open(object_stream)
#         return image

#     def oss_download_base64(self, file_path):
#         object_stream = self.bucket.get_object(file_path)
#         image = Image.open(object_stream)
#         return encode_image_to_base64(image)

#     def get_oss_url(self, file_path):
#         if self.bucket.object_exists(file_path):
#             return self.bucket.sign_url("GET", file_path, self.expired_time)
#         else:
#             return ""

# def filebroker_download(image_url, max_lenght_fliter=False):

#     domain = "filebroker-ofl-sg.aliexpress.com"
#     accessKey = "4ac1bf4c3285449ba93e1d3056426515"  # key
#     accessCode = "AE_AI_P4P_CREATIVE_3"

#     now = datetime.now()
#     # 转换为毫秒级时间戳
#     milliseconds = int(now.timestamp() * 1000)
#     expiredTime = milliseconds + 1000000000
#     file_name = image_url.split("/kf/")[1]
#     origin_value = "&".join([file_name, accessCode, str(expiredTime), accessKey])
#     # 计算originValue的md5，并转换成16进制字符串
#     sign = hashlib.md5(origin_value.encode()).hexdigest()
#     # 拼接URL
#     init_images = None
#     retry_cnt = 3

#     url = f"http://{domain}/ofl/{file_name}?accessCode={accessCode}&expiredTime={expiredTime}&sign={sign}"
#     # 下载图片，使用requests库进行下载URL
#     while retry_cnt > 0:
#         response = requests.get(url)
#         retry_cnt = retry_cnt - 1
#         if response.status_code == 200:
#             content = response.content
#             # init_images = base64.b64encode(content).decode("utf-8")
#             init_images = Image.open(BytesIO(content))
#             break
    
#     if init_images is None:
#         raise Exception("Fail to query filebroker image")

#     # 图片长度限制，在odps中，返回字符串不能大于8388608（8M）
#     # if max_lenght_fliter:
#     #     if len(init_images) >= 8388608:
#     #         init_images = ""
#     return init_images

def get_img_base64(image):
    image = image.byte()
    pil_img = Image.fromarray(image.numpy()) 
    byte_io = BytesIO()
    pil_img.save(byte_io, format='PNG')
    byte_img = byte_io.getvalue() 
    img_base64 = base64.b64encode(byte_img).decode("utf-8")
    return img_base64

# def download_image(image_url, with_alpha=False):
#     # 如果是cdn的图片地址，使用filebroker下载
#     # 如果是oss的图片地址，使用oss的api下载
#     # 否则就用http正常下载
#     # print("download_image: ", image_url)
#     oss_p = ossProcessor("ae-p4p-sg-dev")
#     try:
#         if "alicdn.com" in image_url:
#             image = filebroker_download(image_url)
#         # elif "http" in image_url and "oss" in image_url:
#         #     image = oss_p.oss_download(image_url)
#         else:
#             response = requests.get(image_url)
#             img_data = response.content
#             # image_base64 = base64.b64encode(img_data).decode('utf-8')
#             image = Image.open(BytesIO(img_data))
#     except Exception as e:
#         print("Error: {} !".format(e))
#         if with_alpha:
#             image = Image.fromarray(np.zeros((512,512,4),dtype=np.uint8))
#         else:
#             image = Image.fromarray(np.zeros((512,512,3),dtype=np.uint8))
#     return image

