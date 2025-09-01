#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 15:29
# @Author  : 晨皋
# @File    : filebroker_io.py.py
# @Software: PyCharm
import hashlib
import base64
import requests
from .utils import *

def filebroker_download(image_url, max_lenght_fliter=False):

    # domain = "filebroker-ofl.aliexpress.com"
    # accessKey = "cf1ee0dc3ba14769a0b781afecfc0642" #key
    # accessCode = "AE_AI_P4P_CREATIVE_2"
    domain = "filebroker-ofl-sg.aliexpress.com"
    accessKey = "4ac1bf4c3285449ba93e1d3056426515" #key
    accessCode = "AE_AI_P4P_CREATIVE_3"

    expiredTime = 1730307600000
    file_name = image_url.split("/kf/")[1]
    origin_value = "&".join([file_name, accessCode, str(expiredTime), accessKey])
    # 计算originValue的md5，并转换成16进制字符串
    sign = hashlib.md5(origin_value.encode()).hexdigest()
    # 拼接URL
    init_images = ""
    retry_cnt = 1
    try:
        url = f"http://{domain}/ofl/{file_name}?accessCode={accessCode}&expiredTime={expiredTime}&sign={sign}"
        # 下载图片，使用requests库进行下载URL
        while retry_cnt > 0:
            response = requests.get(url)
            retry_cnt = retry_cnt - 1
            if response.status_code == 200:
                content = response.content
                init_images = base64.b64encode(content).decode("utf-8")
                break
    except Exception as e:
        print("Error: {} !".format(e))
    # 图片长度限制，在odps中，返回字符串不能大于8388608（8M）
    if max_lenght_fliter:
        if len(init_images) >= 8388608:
            init_images = ""
    return init_images