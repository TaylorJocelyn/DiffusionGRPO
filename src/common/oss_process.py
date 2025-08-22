#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/27 15:13
# @Author  : 晨皋
# @File    : oss_process.py
# @Software: PyCharm
import oss2
import requests
import hashlib
import base64
from io import BytesIO
from PIL import Image
from .utils import *


class ossProcessor(object):
    def __init__(self, bucket_name):
        self.auth = oss2.Auth('', '')
        self.endpoint = 'oss-ap-southeast-1-internal.aliyuncs.com'
        self.expired_time = 777600
        self.bucket = oss2.Bucket(self.auth, self.endpoint, bucket_name)

    def oss_upload(self, file_byte, path):
        self.bucket.put_object(path, file_byte)
        sign_url = self.bucket.sign_url("GET", path, self.expired_time)
        return sign_url

    def oss_download_stream(self, file_path):
        object_stream = self.bucket.get_object(file_path)
        return object_stream

    def oss_download_file(self, file_path, out_path):
        self.bucket.get_object_to_file(file_path, out_path)

    def oss_download_base64(self, file_path):
        object_stream = self.bucket.get_object(file_path)
        image = Image.open(object_stream)
        return encode_image_to_base64(image)

    def get_oss_url(self, file_path):
        if self.bucket.object_exists(file_path):
            return self.bucket.sign_url("GET", file_path, self.expired_time)
        else:
            return ""