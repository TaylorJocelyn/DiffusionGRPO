#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/13 15:30
# @Author  : 晨皋
# @File    : query.py
# @Software: PyCharm
import json
from .filebroker_io import filebroker_download
from .oss_process import ossProcessor
from .utils import *
import requests

class Query(object):
    def __init__(self, context):
        request_json_str = context["params"].get("request")
        request_json = json.loads(request_json_str)
        self.batch_size = request_json.get("batch_size", 1)
        self.width = request_json.get("width")
        self.height = request_json.get("height")
        self.prompt = request_json.get("prompt")
        self.negative_prompt = request_json.get("negative_prompt")
        self.image_url = request_json.get("image_url")
        self.image_base64 = self.get_image_data(self.image_url, request_json.get("image_base64",""))
        self.ipa_image_url = request_json.get("ipa_image_url","")
        self.ipa_image_base64 = self.get_image_data(self.ipa_image_url, request_json.get("ipa_image_base64",""))
        self.ipa_fr_image_url = request_json.get("ipa_fr_image_url","")
        self.ipa_fr_image_base64 = self.get_image_data(self.ipa_fr_image_url, request_json.get("ipa_fr_image_base64",""))
        self.mask_image_url = request_json.get("mask_image_url")
        self.mask_image_base64 = self.get_image_data(self.mask_image_url, request_json.get("mask_image_base64",""))

    def get_image_data(self, image_url, image_base64):

        if is_image_base64(image_base64):
            return image_base64
        elif image_url is not None and len(image_url) > 0:
            return self.download_image(image_url)
        else:
            return ""

    def download_image(self, image_url):
        # 如果是cdn的图片地址，使用filebroker下载
        # 如果是oss的图片地址，使用oss的api下载
        # 否则就用http正常下载
        print("download_image: ", image_url)
        if "alicdn.com" in image_url:
            image_base64 = filebroker_download(image_url)
        elif "http" not in image_url:
            oss_p = ossProcessor("ae-p4p-sg-dev")
            image_base64 = oss_p.oss_download_base64(image_url)
        else:
            response = requests.get(image_url)
            img_data = response.content
            image_base64 = base64.b64encode(img_data).decode('utf-8')
        return image_base64


class LocalQuery(Query):
    def __init__(self, 
        batch_size,
        height,
        width,
        prompt,
        negative_prompt,
        image_url="",
        image_base64="",
        mask_image_url="",
        mask_image_base64=""
        ):
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.image_url = image_url
        self.image_base64 = self.get_image_data(self.image_url, image_base64)
        self.mask_image_url = mask_image_url
        self.mask_image_base64 = self.get_image_data(self.mask_image_url, mask_image_base64)
    
    