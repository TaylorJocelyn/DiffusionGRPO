import os
import sys
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件的上一级目录
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
# 将上一级目录加入系统路径
sys.path.append(parent_directory)
import math
import numpy as np
# import torch
# import torchvision.transforms as T
# from decord import VideoReader, cpu
import re
from PIL import Image
# from torchvision.transforms.functional import InterpolationMode
# from transformers import AutoModel, AutoTokenizer
from io import BytesIO
import requests
import os.path as osp
from openai import OpenAI
from uuid import uuid4
import base64
import json
import time
from src.common.filebroker_io import filebroker_download
from src.common.oss_process import ossProcessor
from src.common.utils import decode_base64_to_image, encode_image_to_base64

def download_image(image_url):
    # 如果是cdn的图片地址，使用filebroker下载
    # 如果是oss的图片地址，使用oss的api下载
    # 否则就用http正常下载
    # print("download_image: ", image_url)
    image = None
    if "alicdn.com" in image_url:
        image_base64 = filebroker_download(image_url)
    elif "http" not in image_url:
        oss_p = ossProcessor("ae-p4p-sg-dev")
        image_base64 = oss_p.oss_download_base64(image_url)
        image = oss_p.oss_download_stream(image_url)
    else:
        response = requests.get(image_url)
        img_data = response.content
        image_base64 = base64.b64encode(img_data).decode('utf-8')
    return image_base64, image

class qwenAgent():
    def __init__(self, api_key="", model_name="qwen-max"):
        # If you set `load_in_8bit=True`, you will need two 80GB GPUs.
        # If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
        self.base_url ="https://dashscope.aliyuncs.com/compatible-mode/v1"
        # self.ak = "8a4910ad5dd737a155fb9e096016acdd"
        self.ak = api_key
        self.model_name = model_name

    def encode_image_to_base64(self, image_path):
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        encoded_image = base64.b64encode(image_data)
        return encoded_image.decode('utf-8')

    def make_request(self, message, model_name):
        client = OpenAI(api_key=self.ak, base_url=self.base_url)
        completion = client.chat.completions.create(model=model_name, messages=message)
        response = completion.choices[0].message.content
        return response

    def build_request(self, user_input):
        sys_input = "You are a helpful assistant."
        example_input = """ 
        Translate the following text into Chinese, keeping the format。
        {
            "Scene theme": "Modern living room setting",
            "Background": "Soft, creamy white wall with subtle texture.",
            "Foreground": "A decorative painting is placed on a minimalist wooden easel with a small potted cactus next to it.",
            "Support": "A minimalist wooden easel",
            "Element": "Small potted cactus",
            "Tone": "The overall tone is calm and sophisticated.",
            "Lighting": "Natural light pours in from the side, casting a gentle glow on the painting.",
            "Other settings": "A slight shadow behind the easel creates depth, drawing attention to the artwork."
        }
        """
        sys_output = """ 
        {
            "场景主题": "现代客厅场景",
            "背景": "柔和的奶油白色墙壁,带有微妙的纹理。",
            "前景": "一幅装饰画放在一个极简风格的木制画架上,旁边放着一小盆仙人掌。",
            "支撑物": "极简风格的木制画架",
            "元素": "小盆栽仙人掌",
            "基调": "整体色调平静而高雅。",
            "照明": "自然光从侧面洒入,为画作投下柔和的光辉。",
            "其他设置": "画架后方轻微的阴影增加了深度,使人们的注意力更加集中在艺术品上。"
        }
        """
        user_input = "Translate this paragraph into Chinese, keeping the format。\n {}".format(user_input)

        message = [
            {"role": "system", "content": sys_input},
            {"role": "user", "content": example_input},
            {"role": "assistant", "content": sys_output},
            {"role": "user", "content": user_input}
        ]

        return message

class wanxAgent():
    def __init__(self, api_key="sk-45dedbe241044894b0460105a722a08b", model_name="wanx-background-generation-v2"):
        # If you set `load_in_8bit=True`, you will need two 80GB GPUs.
        # If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
        self.base_url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/background-generation/generation/'
        # self.ak = "8a4910ad5dd737a155fb9e096016acdd"
        self.ak = api_key
        self.model_name = model_name

    
    def make_request(self):
        # 参考代码 https://help.aliyun.com/zh/model-studio/developer-reference/wanx-background-generation-api-reference?spm=a2c4g.11186623.0.0.2f7b7c6arqCXxS#8e702c00ecgnu
        # 定义请求的URL和头部信息
        headers = {
            'X-DashScope-Async': 'enable',
            'Authorization': f'Bearer {self.ak}',
            'Content-Type': 'application/json'
        }

        # 定义请求的JSON数据
        input_image_url = "https://is-content-gen.oss-cn-zhangjiakou.aliyuncs.com/AIGC/wbg_img_gen/result/2024-11-09/wbg_m2f_064055138819499536310a4841bfaf20.png"
        prompt = "a item placed on an old wooden bench with a vintage camera next to it. A rustic brick wall with ivy creeping over it. Vintage urban exploration setting. Earthy and warm, enhancing the vintage feel. no people, realistic, commercial photography, 4k, realistic rendering, RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
        data = {
            "model": self.model_name,
            "input": {
                "base_image_url": input_image_url,
                "ref_prompt": prompt
            },
            "parameters": {
                "n": 4,
                "ref_prompt_weight": 0.5,
                "model_version": "v3"
            }
        }

        # 发送POST请求
        response = requests.post(self.base_url, headers=headers, json=data)
        if response.status_code != 200:
            task_id = ""
        else:
            # 输出返回结果
            try:
                resp_js = response.json()
                task_id = resp_js["output"]["task_id"]
                print(resp_js)
            except:
                task_id = ""
        print("task_id:{}".format(task_id))
        return task_id

    
    def get_response(self, task_id):
        # 定义请求的URL和头部信息
        url = 'https://dashscope.aliyuncs.com/api/v1/tasks/{}'.format(task_id)
        headers = {
            'Authorization': f'Bearer {self.ak}'
        }
        # 发送GET请求
        response = requests.get(url, headers=headers)
        # 几种返回的情况
        # 1. 查询失败
        # 2. 查询成功，任务失败
        # 3. 查询成功，任务执行中
        # 4. 查询成功，任务完成
        image_arr = []
        if response.status_code != 200:
            return image_arr
        else:
            resp_js = response.json()
            task_status = resp_js["output"]["task_status"]
            # 输出返回结果
            print(resp_js)
            if task_status == "RUNNING":
                image_arr = []
            elif task_status == "FAILED":
                image_arr = None 
            elif task_status == "SUCCEEDED":
                results = resp_js["output"]["results"]
                idx = 0  
                for ele in results:
                    image_url = ele["url"]
                    response = requests.get(image_url)
                    img_data = response.content
                    image = Image.open(BytesIO(img_data))
                    image_arr.append(image)
        return image_arr


def qwen_demo():
    # api_key = "8a4910ad5dd737a155fb9e096016acdd"
    # model_name = "qwen-max"
    ida =  qwenAgent()
    user_input = """
        {"Scene theme": "Industrial workspace scene with a minimalist aesthetic", "Background": "A dark gray metal workbench with subtle texture.", "Foreground": "A carbon fiber tube is placed on a metal workbench with a wrench next to it.", "Support": "A metal workbench", "Element": "Wrench", "Tone": "The overall tone is sleek and modern.", "Lighting": "High-contrast lighting with strong shadows to accentuate the tube's pattern.", "Other settings": "A shallow depth of field to blur the surrounding tools and focus on the product."}
    """
    user_js = json.loads(user_input)
    user_str = json.dumps(user_js, indent=4)
    print(user_str)
    prompt = ida.build_request(user_str)
    resp = ida.make_request(prompt, ida.model_name)
    print(resp)

def wanx_demo():
    wxa = wanxAgent()
    task_id = wxa.make_request()
    task_list = []
    task_list.append(task_id)
    time.sleep(60)
    while True:
        if len(task_list) ==0:
            break
        for task_id in task_list:
            images = wxa.get_response(task_id)
            if images is None:
                task_list.remove(task_id)
            if len(images) == 0:
                continue
            if len(images) >0:
                idx = 0
                for image in images:
                    idx +=1
                    image_name = "test_{}.jpg".format(idx)
                    image.save(image_name)
                print("生成图片成功！")
                task_list.remove(task_id)
        print("轮询结束")
        time.sleep(60)


def product_understanding_builder(image_url, item_title):
    messages = ""
    # try:
    # Instruction
    sys_input = """
                You are an excellent and professional e-commerce photographer. You will be provided with a product image and a product title. Your job is to output the product category, product color, product style, product material, product photography height, product photography base view, product real physical width estimate, product real physical height estimate based on the information provided above. The output format is as follows:
                Product category: [product category]
                Product color: [product color]
                Product style: [product style]
                Product material: [product material]
                Product photography height: [product photography angle]
                Product photography view: [product photography base view]
                Notes:
                **Product category**: Determine the product category based on the product title and product features in the image. For example, if the title is "Men's sneakers", then the category is sneakers.
                **Product color**: Determine by observing the product color in the image. If the product is a red sneaker, then the color is red.
                **Product style**: Determine the style based on the product's design and appearance features. For example, if the sneakers have a simple design and smooth lines, they may be classified as modern style.
                **Product material**: Determine the product category based on the product title and product features in the image. For example, if the title is "Ladies wallet, long genuine leather wallet", then the material is leather.
                **Product photography height**: Photographic height refers to the vertical position of the camera relative to the subject. You can only choose one answer from the following four options: Low Angle Shot, Eye Level Shot, High Angle Shot and Overhead Shot.
                - Low Angle Shot: The camera is below the subject and the shot is taken from a low position upwards.
                - Eye Level Shot: The camera is at about the same height as the subject's eyes.
                - High Angle Shot: The camera is above the subject and the shot is taken from a high position downwards.
                - Overhead Shot: The camera is directly above the subject and the shot is taken vertically downwards.
                **Product photography view**: You can only choose one answer from the following five options: Front View, Back View, Side View, Top View and 45-Degree View.
                - front view: Shows the front of the product.
                - back view: Shows the back of the product.
                - side view: Shows the side of the product.
                - top view: Shows the top features of the product.
                - 45-degree view: Shot from a 45-degree angle on one side of the product.
                """
    current_directory = os.path.abspath(os.path.dirname(__file__))
    example_image_path = "product_understand_example.png"
    example_image_path = os.path.join(current_directory, "example", example_image_path)
    example_image = Image.open(example_image_path)
    example_image_base64 = encode_image_to_base64(example_image)

    example_title = "Midea/Midea JSQ30-M5 Household 16L Natural Gas Gas Water Heater Energy Saving Zero Cold Water First-level Noise Reduction."
    example_input = [
        {"type": "text", "text": f"Here is an example:\n The product title: {example_title} "},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64, {example_image_base64}", "image_detail": "low"}}
    ]
    example_output = """
                        Product category: Gas Water Heater
                        Product color: Metallic Gray
                        Product style: Modern and sleek
                        Product material: Metal
                        Product photography height: Eye Level Shot
                        Product photography view: Front View
                    """
    input_text = f"Here is the product title: {item_title}"
    input_image_base64 = download_image(image_url)
    user_input = [
        {"type": "text", "text": input_text},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{input_image_base64}", "image_detail": "low"}}
    ]

    messages = [
        {"role": "system", "content": sys_input},
        {"role": "user", "content": example_input},
        {"role": "assistant", "content": example_output},
        {"role": "user", "content": user_input},
    ]
    # except Exception as e:
    #     print("error message:", e)
    return messages

def creative_advertising_understanding_builder(image, item_title):
    messages = ""
    # try:
    # Instruction
    sys_input = """
                You are an excellent and professional e-commerce photographer. You will be provided with a creative advertising image and a product title. Your job is to describe the foreground, background, scene theme, and tone in concise language. The output format is as follows:
                Foreground: [foreground]
                Background: [background]
                Scene theme: [scene theme]
                Tone: [product material]
                Notes:
                **Foreground**: Describe the foreground content based on the creative advertising image and product title. The foreground content should primarily describe:
                    1. **Product Appearance**: The visual features, design, and physical characteristics of the product.
                    2. **Spatial Arrangement**: The positioning of the product in the scene, including its surrounding context and any relevant elements.
                    For example, if the title is "Educational Toy" and the image shows the toy on a colorful table with other interactive elements, the foreground description might be: "An educational toy is placed on a colorful table with interactive elements around it."
                **Background**: Describe the background content in the creative advertising image. Focus on describing the environment, surroundings, and context behind the main product. This includes elements like decor, landscape, and any other details that enhance or frame the product.
                **Scene theme**: Describe the overall theme or concept of the scene in the creative advertising image.
                **Tone**: Describe the visual tone or color scheme of the creative advertising image. This includes the overall mood created by the colors, lighting, and visual style of the image, such as "bright and inviting," "warm and cozy," or "cool and professional."
                """
    current_directory = os.path.abspath(os.path.dirname(__file__))
    example_image_path = "creative_advertising_understand_example.png"
    example_image_path = os.path.join(current_directory, "example", example_image_path)
    example_image = Image.open(example_image_path)
    example_image_base64 = encode_image_to_base64(example_image)

    example_title = "Nikon Z fc Zfc APS-C Mirrorless Camera Digital Compact Professional Photographer Photography 4K Video Cameras 20.88MP (Renewed)"
    example_input = [
        {"type": "text", "text": f"Here is an example:\n The product title: {example_title} "},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64, {example_image_base64}", "image_detail": "low"}}
    ]
    example_output = """
                        Foreground: A retro camera placed on a vintage wooden table with film rolls and a tripod nearby.
                        Background: Warm gray tones with soft lighting
                        Scene theme: Vintage camera studio setup
                        Tone: Warm and inviting
                    """
    input_text = f"Here is the product title: {item_title}"
    input_image_base64 = encode_image_to_base64(image)
    user_input = [
        {"type": "text", "text": input_text},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{input_image_base64}", "image_detail": "low"}}
    ]

    messages = [
        {"role": "system", "content": sys_input},
        {"role": "user", "content": example_input},
        {"role": "assistant", "content": example_output},
        {"role": "user", "content": user_input},
    ]
    # except Exception as e:
    #     print("error message:", e)
    return messages

# cot 多任务合并推理
def lvlm_critic_builder(image_url):
    # Instruction
    sys_input = """
    你是一个电商图片设计师，善于分析和制作电商广告创意图片。现在你分析一下下面的这张图片，并按下面的步骤逐步进行。
    第一步，描述一下图片的内容。包括：商品主体，摆放位置，背景元素，摆放视角，前后景关系，图片是否是真实拍摄的。
    第二步，判断一下商品主体和背景元素相比，大小是否异常？1-5分之间打分，1表示非常异常，令人感到奇怪，5表示完全正常。
    第三步，根据图片的内容，判断一下商品主体的摆放位置是不是合理的？1-5分之间打分，1表示非常不合理，令人感到奇怪，5分表示完全正常。
    第四步，根据图片的内容，判断一下图片中是否有人体或者人体躯干？0-1打分，1表示有人体或者躯干，0表示没有人体和躯干。
    第五步，根据图片的内容，判断一下图片的颜色搭配是否好看？1-5分之间打分，1表示非常不好看，5表示非常好。
    输出的结果，按照示例，以json格式输出。
    """
    # few-shot learning
    res_image_url = "https://ae01.alicdn.com/kf/S42ccc55fb40443369c03dbcb66be5b863.jpg"
    res_image_base64 = download_image(res_image_url)
    item_image = decode_base64_to_image(res_image_base64)
    # item_image.save("example1.jpg")
    example_input = [
        {"type": "text", "text": "下面是输入的图片，请按任务提示进行分析，并给出结果:\n"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{res_image_base64}", "image_detail": "low"}}
    ]
    example_output = """"{
                        "step1":{
                            "图片描述": "**商品主体**:图片中的商品看起来像一个安装在汽车内部的无线蓝牙耳机。\n**摆放位置**: 耳机被放置在汽车中控台附近，好像是悬浮在空中，利用创造性的方式呈现。\n**背景元素**: 汽车内部环境，包括方向盘、仪表盘、中控屏、挡把等，给人一种科技感和现代感。\n**摆放视角**: 视角是从汽车后座向前方拍摄，展示了汽车内部的设计和耳机的定位。\n**前后景关系**: 耳机在前景中，被放大突出，而汽车内部则是背景元素。\n**图片真实性**: 图片看起来是通过合成或计算机生成设计，而不是真实拍摄的。\n"
                        },
                        "step2":{
                            "商品大小异常判断": 1,
                            "解释":"商品主体（无线蓝牙耳机）和背景元素（汽车内饰）相比，大小明显异常。耳机比正常比例大得多。"
                        },
                        "step3":{
                            "商品摆放位置合理性判断":2,
                            "解释":"商品主体被放在中央扶手的位置，这个位置不太常见，显得有些突兀和不合理。"
                        },
                        "step4":{
                            "人体躯干判断":0,
                            "解释":"图片中没有任何人体或人体躯干可见。"
                        },
                        "step5":{
                            "颜色搭配打分":4,
                            "解释":"图片的颜色搭配总体上是和谐的，看起来比较好看。"
                        }
                    }"""

    res_image_url = "https://ae01.alicdn.com/kf/S265664618bfb43bebe201d87dbf6a036J.jpg"
    res_image_base64 = download_image(res_image_url)
    item_image = decode_base64_to_image(res_image_base64)
    # item_image.save("example2.jpg")
    example_input_2 = [
        {"type": "text", "text": "下面是输入的图片，请按任务提示进行分析，并给出结果:\n"},
        {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{res_image_base64}", "image_detail": "low"}}
    ]
    example_output_2 = """
                    {
                        "step1":{
                            "图片描述": "**商品主体**:图片中的商品是一块手表，具有银色和红色的外观设计，表盘采用镂空设计，显露出内部机械结构。\n**摆放位置**:手表摆放在一个类似大理石纹理的表面上。\n**背景元素**:背景是模糊的，含有一些花卉和绿色植物。\n**摆放视角**:手表的视角是正面略微俯视。\n**前后景关系**:手表作为主体，在前景中清晰显示，背景则较为模糊，呈现景深效果。\n**图片真实性**:看起来像是通过数字手段处理的图像，而不是真实拍摄，更多为广告设计风格。\n"
                        },
                        "step2":{
                            "商品大小异常判断": 4,
                            "解释":"手表与背景植物相比略大，但尚在正常范围内，更体现产品细节。"
                        },
                        "step3":{
                            "商品摆放位置合理性判断":4,
                            "解释":"手表的摆放位置合理，突出主体，背景也不显得杂乱。"
                        },
                        "step4":{
                            "人体躯干判断":0,
                            "解释":"图片中没有任何人体或人体躯干可见。"
                        },
                        "step5":{
                            "颜色搭配打分":4,
                            "解释":"红色与银色腕表搭配周围绿色植物和红色花卉，挺协调。"
                        }
                    }
    """

    # input
    # image_url = "https://ae01.alicdn.com/kf/Sf63b45ba03154d1ebeb26ae8bcef9244z.jpg"
    input_image_base64 = download_image(image_url)
    item_image = decode_base64_to_image(input_image_base64)
    # item_image.save("input.jpg")
    lvlm_input = "下面是输入的图片，请按任务提示进行分析，并给出结果"
    user_input = [
                    {"type": "text", "text": lvlm_input},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_image_base64}", "image_detail": "low"}}
                ]

    message = [
        {"role": "system", "content": sys_input},
        {"role": "user", "content": example_input},
        {"role": "assistant", "content": example_output},
        {"role": "user", "content": example_input_2},
        {"role": "assistant", "content": example_output_2},
        {"role": "user", "content": user_input}
    ]
    return message

#  品类过滤
def category_fliter_builder(image_url, item_title):
    # Instruction
    sys_input = """
    你是一个电商图像审核专家，需要通过多维度分析判断商品是否符合营销背景图标准。请按以下步骤分析：
        1. 基础验证
        - 标题与实物一致性：商品主体是否与标题描述一致？(1/0)
        - 正面视角验证：是否展示产品完整正面形态？(需看到产品全貌，允许15°内角度偏移)(1/0)

        2. 排除项验证（任一为1则淘汰）
        - 工业配件：包含齿轮/轴承/传动装置等机械部件？(1/0)
        - 汽摩配件：涉及车辆动力系统/操控系统/外观部件？（注意完整的车辆不属于配件）(1/0)
        - 电子配件：属于需配合主机使用的附属品？（如充电器/数据线/替换零件）(1/0) 
        - 纺织品：具有是材质柔软的纺织物品？（比如床单、窗帘、毛巾等）(1/0)
        - 人体元素：包含完整/部分人体（包括虚拟形象）？(1/0)

        3. 材质验证
        - 表面材质：用3个关键词描述主体材质（如塑料/金属/玻璃）

        请用JSON格式输出：
        {
        "basic_validation": {
            "title_match": 1,
            "frontal_view": 1
        },
        "exclusion_check": {
            "industrial_parts": 0,
            "vehicle_parts": 0,
            "electronic_accessories": 0,
            "textiles": 0,
            "human_presence": 0
        },
        "material_analysis": ["plastic", "metal", "rubber"]
        }
    """
    # few-shot learning
    current_directory = os.path.abspath(os.path.dirname(__file__))
    example_image_path = "S84ef941b0ab54976aa99952930261a58z_1738965209000_p.png"
    example_image_path = os.path.join(current_directory, "example", example_image_path)
    example_image = Image.open(example_image_path)
    res_image_base64 = encode_image_to_base64(example_image)
    # res_image_base64 = download_image(example_image)
    # item_image = decode_base64_to_image(res_image_base64)
    # example_image.save("example1.png")
    item_title_exp = "Original DJI Mini 1/2/SE/2 SE/4K Gimbal Yaw Motor, Yaw Bracket, Camera Frame, DJI Mini 1/2/SE/2 SE/4K Repair Parts"
    example_input = [
        {"type": "text", "text": f"商品标题是:{item_title_exp},下面输入的图片，请按任务提示进行分析，并给出结果:\n"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{res_image_base64}", "image_detail": "low"}}
    ]
    example_output = """"
            {
            "basic_validation": {
                "title_match": 1,
                "frontal_view": 1
            },
            "exclusion_check": {
                "industrial_parts": 1,
                "vehicle_parts": 0,
                "electronic_accessories": 0,
                "textiles": 0,
                "human_presence": 0
            },
            "material_analysis": ["metal"]
            }
    """

    # res_image_url = "http://ae-p4p-sg-dev.oss-ap-southeast-1.aliyuncs.com/aigc_image%2Fimage_matting%2FSdffd91f3c92c4b7396b52ceedd205695e_1738844394000_p.png?OSSAccessKeyId=LTAI5t8njYs9rQbzcyLBcKB6&Expires=1741840784&Signature=GOgmUAUxBB3KL%2FvP0Y%2FpHJvNa7o%3D"
    example_image_path = "Sdffd91f3c92c4b7396b52ceedd205695e_1738844394000_p.png"
    example_image_path = os.path.join(current_directory, "example", example_image_path)
    example_image2 = Image.open(example_image_path)
    res_image_base64 = encode_image_to_base64(example_image2)
    # item_image = decode_base64_to_image(res_image_base64)
    # example_image2.save("example2.png")
    item_title_exp = "128GB R36MAX Retro Handheld Game Console Linux System 4 Inch IPS Screen Portable Video Players Dual Joystick 64G Games for Kids"
    example_input_2 = [
        {"type": "text", "text": f"商品标题是:{item_title_exp},下面输入的图片，请按任务提示进行分析，并给出结果:\n"},
        {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{res_image_base64}", "image_detail": "low"}}
    ]
    example_output_2 = """
            {
            "basic_validation": {
                "title_match": 1,
                "frontal_view": 1
            },
            "exclusion_check": {
                "industrial_parts": 0,
                "vehicle_parts": 0,
                "electronic_accessories": 0,
                "textiles": 0,
                "human_presence": 0
            },
            "material_analysis": ["plastic"]
            }
    """

    # input
    # image_url = "https://ae01.alicdn.com/kf/Sf63b45ba03154d1ebeb26ae8bcef9244z.jpg"
    input_image_base64 = download_image(image_url)
    item_image = decode_base64_to_image(input_image_base64)
    # item_image.save("input.png")
    lvlm_input = f"商品标题是:{item_title},下面输入的图片，请按任务提示进行分析，并给出结果:\n"
    user_input = [
                    {"type": "text", "text": lvlm_input},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_image_base64}", "image_detail": "low"}}
                ]

    message = [
        {"role": "system", "content": sys_input},
        {"role": "user", "content": example_input},
        {"role": "assistant", "content": example_output},
        {"role": "user", "content": example_input_2},
        {"role": "assistant", "content": example_output_2},
        {"role": "user", "content": user_input}
    ]
    return message


#  抠图结果过滤
def matting_fliter_builder(image_url, image_url_2):
    message = ""
    try:
        # Instruction
        sys_input = """
            你是一位资深的电商商品创意美工，擅长分析电商图美化过程中的细节。user将会上传两张图片，第一张图片是商品原始图，第二张图片是抠图处理后的商品绿色背景图。
            你的任务是通过判断抠图前后的两张图片，对抠图效果进行分析和评判。
            评判逻辑：
                - 抠图不完整：应该抠出的物体内容没有抠出
                - 抠图抠出多余内容：不应该被抠出的内容被抠出了
                - 抠图无问题：以上两项都没有问题，则说明抠图无问题
            一般而言，对于抠图后的结果期望是内容完整、边缘清晰的，如果抠出的物品内容或元素不完整则定义为抠图不完整，如果还包含背景中的内容且内容不完整，则定义为抠出多余内容。
            将抠出元素分为主商品和其他元素，主商品为主要要展示的商品物体，其他元素包含背景中的物体、装饰物、摆台以及说明文字、logo等，抠图无问题的判定要求是主商品必须完整，其他元素如果被抠出也必须完整一致。
            一些抠图常见问题包括[明显锯齿边缘-为可肉眼分辨的阶梯状边缘][毛边-不自然的过渡区域][背景残留(原图背景元素)]、[阴影残留]、[相邻物体残留]等等

            输出规范-这是你需要最终输出的Structured Output JSON SCHEMA:
            ````json
            {
                "抠图不完整": {
                    "type": "boolean",
                    "description": "对比用户上传的两张图，第二张图上抠出的物品和元素有明显残缺部分(如缺角、明显边缘锯齿、内容缺失)。"
                },
                "抠图抠出多余内容": {
                    "type": "boolean",
                    # ，孔洞未抠除
                    "description": "对比用户上传的两张图，第二张图上扣出物品还包含了背景中的部分内容，例如背景残留、阴影，或粘连了其他物体的部分内容。"
                },
                "抠图无问题": {
                    "type": "boolean",
                    "description": "以上两项都没有问题，则说明商品抠图无问题"
                }
            }
            ```
            先对用户上传的两张图片分析思考，最终输出结构化的JSON。
            以下是几个例子(此例子与user上传图片无关): 
                - Example 1
                Thought: 抠图后的商品主体完整，但是还抠出了背景中的桌面的部分内容，但相对原图来看抠出的桌面和原图是一致的。
                Structured Output:
                ```json{
                    "抠图不完整": false,
                    "抠图抠出多余内容": false,
                    "抠图无问题": true
                }```
                - Example 2
                Thought: 商品主体完整干净，也没有多余元素掺入。
                Structured Output: 
                ```json{
                    "抠图不完整": false,
                    "抠图抠出多余内容": false,
                    "抠图无问题": true
                }```
                - Example 3
                Thought: 仔细看第二张图(抠图后结果)，发现第二张图上的商品主体有明显的缺角。
                Structured Output:
                ```json{
                    "抠图不完整": true,
                    "抠图抠出多余内容": false,
                    "抠图无问题": false
                }```
                - Example 4
                Thought: 抠出的商品主体完整，但是还抠出了背景中的Logo内容，Logo相对原图来看存在缺失不完整。
                Structured Output:
                ```json{
                    "商品不完整": true,
                    "抠图抠出多余内容": false,
                    "抠图无问题": false
                }```
                - Example 5
                Thought: 仔细看第二张图(抠图后结果)，发现第二张图上的商品主体抠出了背景的纯色内容。
                Structured Output:
                ```json{
                    "抠图不完整": false,
                    "抠图抠出多余内容": true,
                    "抠图无问题": false
                }```
            """
        # few-shot learning
        current_directory = os.path.abspath(os.path.dirname(__file__))
        res_image_url = "Sd1e5a042929f464fb7574b71ed4323afq.jpg"
        res_image_url = os.path.join(current_directory,"example", res_image_url)
        res_image = Image.open(res_image_url)
        res_image_base64 = encode_image_to_base64(res_image)
        # res_image_base64 = download_image(res_image_url)
        # item_image = decode_base64_to_image(res_image_base64)
        # item_image.save("example1.png")
        res_image_url_2 = "Sd1e5a042929f464fb7574b71ed4323afq_1738921336000_p.png"
        res_image_url_2 = os.path.join(current_directory,"example", res_image_url_2)
        res_image_2 = Image.open(res_image_url_2)
        res_image_base64_2 = encode_image_to_base64(res_image_2)
        # res_image_base64_2 = download_image(res_image_url_2)
        # item_image_2 = decode_base64_to_image(res_image_base64_2)

        example_input = [
            {"type": "text", "text": "下面输入的商品原图和抠图结果，请按任务提示进行分析，并给出结果:\n"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{res_image_base64}", "image_detail": "low"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{res_image_base64_2}", "image_detail": "low"}}
        ]
        example_output = """"
                {
                    "抠图不完整": false,
                    "抠图抠出多余内容": false,
                    "抠图无问题": true
                }
        """

        res_image_url = "S998fef366d26478a9cad5bd585f8bec8X.jpg"
        res_image_url = os.path.join(current_directory,"example", res_image_url)
        res_image = Image.open(res_image_url)
        res_image_base64 = encode_image_to_base64(res_image)
        # res_image_base64 = download_image(res_image_url)
        # item_image = decode_base64_to_image(res_image_base64)
        # item_image.save("example1.png")
        res_image_url_2 = "S998fef366d26478a9cad5bd585f8bec8X_1738888277000_p.png"
        res_image_url_2 = os.path.join(current_directory,"example", res_image_url_2)
        res_image_2 = Image.open(res_image_url_2)
        res_image_base64_2 = encode_image_to_base64(res_image_2)
        # res_image_base64_2 = download_image(res_image_url_2)
        # item_image_2 = decode_base64_to_image(res_image_base64_2)

        example_input_2 = [
            {"type": "text", "text": "下面输入的商品原图和抠图结果，请按任务提示进行分析，并给出结果:\n"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{res_image_base64}", "image_detail": "low"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{res_image_base64_2}", "image_detail": "low"}}
        ]
        example_output_2 = """"
                {
                    "抠图不完整": false,
                    "抠图抠出多余内容": true,
                    "抠图无问题": false
                }
        """

        # input
        # image_url = "https://ae01.alicdn.com/kf/Sf63b45ba03154d1ebeb26ae8bcef9244z.jpg"
        input_image_base64 = download_image(image_url)
        # item_image = decode_base64_to_image(input_image_base64)
        # item_image.save("input.png")
        input_image_base64_2 = download_image(image_url_2)

        user_input = [
                        {"type": "text", "text": "下面输入的商品原图和抠图结果，请按任务提示进行分析，并给出结果:\n"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_image_base64}", "image_detail": "low"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_image_base64_2}", "image_detail": "low"}}
                    ]

        message = [
            {"role": "system", "content": sys_input},
            {"role": "user", "content": example_input},
            {"role": "assistant", "content": example_output},
            {"role": "user", "content": example_input_2},
            {"role": "assistant", "content": example_output_2},
            {"role": "user", "content": user_input}
        ]
    except Exception as e:
        print("error message:", e)
    return message

def parse_matting_fliter_resp(input_text):
    json_formatted = ""
    exclu_flag = 1
    try:
        parsed_json = get_json(input_text)
        json_formatted = json.dumps(parsed_json,indent=4, ensure_ascii=False)
        exclu_value = 0
        for key, value in parsed_json.items():
            # print(key)
            # print(value)
            if key == "抠图不完整":
                if value:
                    exclu_value +=1
            if key == "抠图抠出多余内容":
                if value:
                    exclu_value +=1
        if exclu_value == 0:
            exclu_flag = 0
    except Exception as e:
        print("error message:", e)

    return json_formatted, exclu_flag

def get_json(input_text):
    # 使用正则表达式提取JSON部分
    json_match = re.search(r'{.*}', input_text, re.DOTALL)
    parsed_json = {}
    if json_match:
        json_data = json_match.group(0)
        # 解析JSON数据
        parsed_json = json.loads(json_data)
        # 格式化输出JSON
        # json_formatted = json.dumps(parsed_json, indent=4, ensure_ascii=False)
        # print(json_formatted)
    else:
        print("未找到JSON数据")
    return parsed_json

def parse_prodcut_understand_resp(input_text):
    json_formatted = ""
    try:
        product_info_key_list = [
            'Product category', 'Product color', 'Product style', 'Product material',
            'Product photography height', 'Product photography view',
        ]
        product_info = {}
        for line in input_text.split('\n'):
            for key in product_info_key_list:
                if key in line:
                    value = line.split(":", 1)[1]
                    if 'physical' in key:
                        value = value.split()[0]
                        product_info[key.strip()] = float(value)
                    else:
                        product_info[key.strip()] = value.strip()
        json_formatted = json.dumps(product_info,indent=4, ensure_ascii=False)
    except Exception as e:
        print("error message:", e)

    return product_info

def parse_creative_advertising_understand_resp(input_text):
    json_formatted = ""
    try:
        product_info_key_list = [
            'Foreground', 'Background', 'Scene theme', 'Tone'
        ]
        product_info = {}
        for line in input_text.split('\n'):
            for key in product_info_key_list:
                if key in line:
                    value = line.split(":", 1)[1]
                    if 'physical' in key:
                        value = value.split()[0]
                        product_info[key.strip()] = float(value)
                    else:
                        product_info[key.strip()] = value.strip()
        json_formatted = json.dumps(product_info,indent=4, ensure_ascii=False)
    except Exception as e:
        print("error message:", e)

    return product_info

def qwenvl_product_understand(image_url, item_title):
    message = product_understanding_builder(image_url, item_title)
    model_name = "qwen-vl-max"
    ida = qwenAgent(model_name=model_name)
    resp = ida.make_request(message, ida.model_name)
    # print(resp)
    parsed_json = parse_prodcut_understand_resp(resp)
    print(parsed_json)
    return parsed_json

def qwenvl_creative_advertising_understand(image, item_title):
    message = creative_advertising_understanding_builder(image, item_title)
    model_name = "qwen-vl-max"
    ida = qwenAgent(model_name=model_name)
    resp = ida.make_request(message, ida.model_name)
    # print(resp)
    parsed_json = parse_creative_advertising_understand_resp(resp)
    print(parsed_json)
    return parsed_json

def get_img_base64(image):
    image = image.byte()
    pil_img = Image.fromarray(image.numpy()) 
    byte_io = BytesIO()
    pil_img.save(byte_io, format='PNG')
    byte_img = byte_io.getvalue() 
    img_base64 = base64.b64encode(byte_img).decode("utf-8")
    return img_base64

def qwenvl_demo():
    image_url = "https://ae01.alicdn.com/kf/Sf63b45ba03154d1ebeb26ae8bcef9244z.jpg"
    message = lvlm_critic_builder(image_url)
    model_name = "qwen-vl-max-2025-01-25"
    ida =  qwenAgent(model_name=model_name)
    resp = ida.make_request(message, ida.model_name)
    print(resp)

def qwenvl_cate_fliter(image_url, item_title):
    # image_url = "http://ae-p4p-sg-dev.oss-ap-southeast-1.aliyuncs.com/aigc_image%2Fimage_matting%2FSbd5129b5839e4b82bfff479171deb20eI_1738922167000_p.png?OSSAccessKeyId=LTAI5t8njYs9rQbzcyLBcKB6&Expires=1741840763&Signature=BDcqc59AZxXR3uoEIQxTavslr4g%3D"
    # item_title = "UV Protection Hat Fisherman Cap Sun Hat Portable Foldable Wide Brim Sun Protection Hats Summer Adjustable Size Cap for Women"
    
    message = category_fliter_builder(image_url,item_title)
    model_name = "qwen-vl-max-latest"
    ida =  qwenAgent(model_name=model_name)
    resp = ida.make_request(message, ida.model_name)
    print(resp)
    return resp

def qwenvl_matting_fliter(origin_image_url, image_url):
    resp = ""
    message = matting_fliter_builder(origin_image_url, image_url)
    model_name = "qwen-vl-max"
    ida = qwenAgent(model_name=model_name)
    resp = ida.make_request(message, ida.model_name)
    result = parse_matting_fliter_resp(resp)
    parsed_json, exclu_flag = result
    print(parsed_json)
    return parsed_json


def ocr_request_builder(image_url):
    input_image_base64 = download_image(image_url)
    # item_image = decode_base64_to_image(input_image_base64)
    # item_image.save("input.png")

    user_input = [
        {"type": "text", "text": "Read all the text in the image."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_image_base64}"}},
        ]
    message = [
        {"role": "user", "content": user_input}
    ]
    return message

def qwenvl_ocr(image_url):
    resp = ""
    message = ocr_request_builder(image_url)
    model_name = "qwen-vl-ocr"
    api_key = "sk-324ab44172634701ae9075685ded872c"
    ida = qwenAgent(api_key=api_key, model_name=model_name)
    resp = ida.make_request(message, ida.model_name)
    print(resp)

if __name__ == "__main__":  

    # img_url = 'aigc_image/zq_dev/rl_annot_filtered_s/infer_Sd348f184193546608f5cfe8b834ddf28i_1752982848003.png'
    # base64_str, _ = download_image(img_url)
    # byte_data = base64.b64decode(base64_str)
    # byte_io = BytesIO(byte_data)
    # pil_img = Image.open(byte_io)
    # pil_img.save('/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/DiffusionRL/src/vlm_api/example/creative_advertising_understand_example.png')

    # local_api_request()
    # demo()
    # image_base64, image = download_image(img_url)

    from torchvision import transforms
    import torch
    
    device = torch.device('cuda')
    pil_img = Image.open('/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/DiffusionRL/data/rl_embeddings/ori_img/1005009263398714.png')
    qwenvl_creative_advertising_understand(pil_img, 'Beige sofa set')
    
    # qwenvl_cate_fliter(image_url, item_title)
    # image_url = "https://ae04.alicdn.com/kf/S6d2519cee0df4ce084ef4b7481324c09a.jpg"
    # image_url_2 = "http://ae-p4p-sg-dev.oss-ap-southeast-1.aliyuncs.com/aigc_image%2Fimage_matting%2FS6d2519cee0df4ce084ef4b7481324c09a_1738845556000_p.png?OSSAccessKeyId=LTAI5t8njYs9rQbzcyLBcKB6&Expires=1741840767&Signature=l3CEMv2ZlzU7C6UOQIt2fn%2F%2FwyA%3D"
    # qwenvl_matting_fliter(image_url, image_url_2)

    # image_url = "https://ae01.alicdn.com/kf/S9c4359c554ce499ab309a8adb0f34068t.jpg"
    # image_url = "https://ae01.alicdn.com/kf/Sb66e2bfef0ac431e8fbde5adff89a67cO.jpg"
    # image_url = "https://ae01.alicdn.com/kf/S3f6dea16dbfe43a3b1e45397acbb0963n.jpg"
    # qwenvl_ocr(image_url)