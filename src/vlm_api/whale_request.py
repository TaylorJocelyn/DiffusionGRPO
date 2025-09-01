import json

from whale import TextGeneration
from whale.util import Timeout
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from openai import OpenAI
import time
import re
# from TestConfigForDemo import *


def build_prompt(image_content_js):
    # 请求模型
    # prompt = [ 
    #             {"role": "system", "content": "你是一个人工智能助手，能够准确的理解并回答我的问题"},
    #             {"role": "user", "content": "美国的首都是哪个城市？"}
    #          ]
    sys_prompt_4 = """
                    You are a professional advertising poster designer. Very good at designing the background or environment that matches the product to enhance the attractiveness                   and expression of the product.
                    Your job is to design reasonable, high-quality scenes and matching elements for the product based on your understanding of the product. Note that the product                     image will be used as the foreground and cannot be modified in any way, such as making the model wear the product unable to output. 
                    Note:
                    - You can deeply understand the characteristics of the product, including product category, product color, product style, product material, product shooting                         height, product shooting Angle, etc.
                    - You can determine a scene theme based on the product category. For example, if the product is outdoor sports equipment, the scene might be an outdoor mountain                     or forest.
                    - Select a color that coordinates or contrasts with the product color. For example, if the product is bright red, the background can be a neutral color like gray                   or white to highlight the product, or a complementary color like green to create a visual impact. The background color should not be too striking, nor should it                   be too close to the product color, so as not to cause visual confusion.
                    - Suitable scene elements can be selected according to the product style to create an atmosphere. For example, a retro-inspired product may require a retro-                         inspired background, such as an old record player or vintage furniture.
                    - You can select a background material that matches or compares the product material based on the product material. For example, if the product is made of smooth                   metal, the background may contain some rough texture, such as concrete or stone, to create visual contrast.
                    - Can reasonably design the surrounding environment of the product according to the shooting height. For example, a product photo taken from above can choose a                     background that shows the full picture of the product, such as a desktop layout, a solid color background, or a texture that harmonizes with the pattern on the                   product.
                    - You can design a reasonable product environment according to the product shooting Angle. If the product image shows the top features of the product, the                           background can be a background that complements those features, such as using a texture or pattern that harmonizes with the top pattern of the product.
                    - Reasonable light and shadow effects can be designed according to product characteristics to increase the realism and three-dimensional sense of the scene.
                    - The Settings of scene background, theme, tone, lighting etc. in the output content are controlled within 10 words. The foreground of the scene is controlled                       within 20 words.
                    - The foreground must follow the format of "[product category] placed on [support] with an [Element 1] next to it", where the support is a table, ground, grass or                   similar object. For example, "A diving mask is placed on a colorful striped beach towel with a shell next to it."


                    An example is as follows:
                    Product image understanding:
                    Product Category: Diving Mask
                    Product color: Black
                    Product style: Animation
                    Material: Plastic
                    Product height: eye level shooting
                    Product shooting Angle: front

                    Output result

                    "Scene theme": "Relaxing summer beach scene at sunset",
                    "Background": "The sky is painted in warm hues of orange, pink and purple.",
                    "Foreground": "A diving mask is placed on a colorful striped beach towel next to a shell.",
                    "Support": "A colorful striped beach towel",
                    "Element": "Shells",
                    "Tone": "The overall tone is bright and cheerful.",
                    "Lighting": "To highlight the product, the beach towel and surrounding elements are in pastel colors.",
                    "Other settings": "Shallow depth of field blur the background to highlight the product."

                    Note: Don't output the thought process!
                    """
    sys_input = sys_prompt_4
    user_input = "OK, Here is the product image understanding: {}\n Note: Don't output the thought process!"
    # image_content_js = {
    #           "Product category": "Handbag",
    #           "Product color": "Brown",
    #           "Product style": "Casual and fashion",
    #           "Product material": "Leather",
    #           "Product photography height": "Eye Level Shot",
    #           "Product photography view": "Front View"
    #         }
    product_category = image_content_js.get("Product category", "")
    product_color = image_content_js.get("Product color", "")
    product_style = image_content_js.get("Product style", "")
    product_material = image_content_js.get("Product material", "")
    product_photography_height = image_content_js.get("Product photography height", "")
    photography_view = image_content_js.get("Product photography view", "")
    item_image_content_format = """Product image understanding: 
                                    Product category: {}
                                    Product color: {}
                                    Product style: {}
                                    Product material: {} 
                                    Product shooting height: {}
                                    Product shooting angle: {}     
                                    """
    item_image_content = item_image_content_format.format(product_category, product_color, product_style,
                                                                  product_material, product_photography_height,
                                                                  photography_view)
    user_input = user_input.format(item_image_content)
    messages = [
                {"role": "system", "content": sys_input},
                {"role": "user", "content": user_input},
            ]
    return messages

# 参考文档 https://aliyuque.antfin.com/gdorir/ahrwo0/whale-inference-input-output-format
# 请求自己部署的deepseek8b模型
def send_request(args):
    messages = args
    
    # apikey = TestConfigForDemo.get_api_key()
    apikey = ""
    base_url = "https://offline-whale-wave.alibaba-inc.com"  # 生产网离线域名
    # base_url = "https://internal-offline-whale-wave.alibaba-inc.com" # 办公网离线域名

    TextGeneration.set_api_key(apikey, base_url=base_url)
    # 设置模型生成结果过程中的参数
    config = {
        "max_new_tokens": 1000,
        "top_p": 0.8,
        "temperature": 0,
        "topk":1
    }

    response = TextGeneration.call(
        model="openlm_ae_p4p_creative_deepseek8b",
        prompt=messages,
        streaming=False,
        timeout=Timeout(100, -1),
        generate_config=config)
    res_js = response.data
    response = res_js["output"]["response"]
    ans = response.split("</think>")[-1]
    ans.replace("\n","")
    return ans

# 请求官方提供的deepseek满血模型
# 使用流失方式，不推荐
def sene_deepseek_request(msgs):

    # apikey = TestConfigForDemo.get_api_key()
    apikey = ""
    base_url = "https://offline-whale-wave.alibaba-inc.com"  # 生产网离线域名
    # base_url = "https://internal-offline-whale-wave.alibaba-inc.com" # 办公网离线域名

    TextGeneration.set_api_key(apikey, base_url=base_url)

    extend_fields= {"top_k": 1}
    response = TextGeneration.chat(
        model="DeepSeek-R1",
        messages=msgs,
        stream=True,
        temperature=1.0,
        max_tokens=2000,
        timeout=Timeout(60, 20),
        top_p=0.8,
        extend_fields=extend_fields)

    result = ""
    # 处理流式结果
    for event in response:
        if event.error_code is not None:
            print(f'Error: {event.error_code, event.message}')
        else:
            if event.choices[0].finish_reason is not None and event.choices[0].finish_reason != '':
                print(f'Finished Reason: {event.choices[0].finish_reason}')
                break
            content = event.choices[0].delta.content
            reasoning_content = event.choices[0].delta.reasoning_content
            if reasoning_content is not None:
                print("思考：")
                print(json.dumps(reasoning_content, ensure_ascii=False))
            if content is not None:
                print("回答：")
                print(json.dumps(content, ensure_ascii=False))
                result += content

            print('====================')

    print(result)

# 使用openai的格式请求deepseek r1
def sene_deepseek_request_withopenai(message):

    apikey = ""
    base_url = "https://offline-whale-wave.alibaba-inc.com"  # 生产网离线域名

    client = OpenAI(
        api_key=apikey,
        base_url=f"{base_url}/api/v2/services/aigc/text-generation/chat/completions",# 查阅Whale API 服务文档确认域名 https://aliyuque.antfin.com/gdorir/ahrwo0/gcwgz78cmz15l8r2
    )
    completion = client.chat.completions.create(
        model="DeepSeek-R1",
        stream=True,
        messages= message,
        extra_body={
            "extend_fields": {
            }
        }
    )
    for chunk in completion:
        print("========================")
        print(chunk)
        if chunk.choices:
            if chunk.choices[0].delta.dict().get("reasoning_content"):
                print("【思考】: " + chunk.choices[0].delta.dict().get("reasoning_content"))
            if chunk.choices[0].delta.content:
                print("【回答】: " + chunk.choices[0].delta.dict().get("content"))

def parse_resp(input_text):
    # 输入字符串
    # input_text = '''
    # Just provide the output result.
    # "Scene theme": "Elegant leather handbag in a sophisticated setting",  
    # "Background": "A neutral-colored wall with subtle texture resembling aged leather.",  
    # "Foreground": "A brown leather handbag is placed on a sleek, modern table with a small decorative bowl next to it.",  
    # "Support": "A sleek, modern table",  
    # "Element": "A small decorative bowl",  
    # "Tone": "The overall tone is sophisticated and understated.",  
    # "Lighting": "Soft, diffused lighting highlights the texture of the handbag and table.",  
    # "Other settings": "Shallow depth of field blur the background to highlight the product."
    # '''
    # 使用正则表达式提取键值对 
    pattern = r'\"(.*?)\": \"(.*?)\"' 
    matches = re.findall(pattern,input_text)
    # print(matches)
    # 将输入字符串转换为字典
    key_set = {"Scene theme","Background","Foreground","Support","Element","Tone","Lighting","Other settings"}
    data_dict = {}
    for key,value in matches:
        if key in data_dict:
            data_dict = {}
            break
        if key in key_set:
            data_dict[key] = value

    if not data_dict:
        lines = [line.strip() for line in input_text.split('\n') if line.strip()]
        for line in lines:
            line = line.replace("**","").replace("-","")
            # 分割键值对
            if '": ' not in line:
                continue  # 跳过无效行

            # 提取键和值
            key_part, value_part = line.split('": ', 1)
            key = key_part.strip('"')  # 去除双引号
            value = value_part.strip()
            data_dict[key] = value

    # 将字典转换为 JSON 格式
    # json_data = json.dumps(data_dict, indent=4, ensure_ascii=False)
    print(data_dict)
    return data_dict


def parse_text_to_json(text):
    pattern = r'-\s*\*\*([^:]+)\*\*:\s*(.*)'
    result = {}

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(pattern, line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            result[key] = value

    return result

def batch_process():
    start_time = time.time()
    num_workers = 6
    print("num_workers: ", num_workers)
    
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        all_task = list()
        results = list()
        step = 100
        for idx in range(step):
            message = build_prompt()
            if len(all_task) < num_workers:
                all_task.append(executor.submit(send_request, (message)))
                print('Create task:', idx)
            else:
                for future in as_completed(all_task):
                    all_task.remove(future)
                    all_task.append(executor.submit(send_request, (message)))
                    print('Create task:', idx)
                    results.append(future.result()[1])
                    break
        for future in as_completed(all_task):
            results.append(future.result()[1])
    
    print("result cnt:{}".format(len(results)))
    
    end_time = time.time() 
    elapsed_time = end_time - start_time 
    print(f"代码执行时间: {elapsed_time} 秒")


def scene_generation(item_image_content):
    message = build_prompt(item_image_content)
    resp = send_request(message)
    print(resp)
    # 按行分割并过滤空行
    resp_dict = parse_resp(resp)

    if resp_dict == "{}":
        resp_dict = parse_text_to_json(resp)
    print(resp_dict)
    return resp_dict

if __name__ == "__main__":
    item_image_content = {
                        "Product category": "Circuit board",
                        "Product color": "Black",
                        "Product style": "Modern",
                        "Product material": "Plastic and metal",
                        "Product photography height": "Eye Level Shot",
                        "Product photography view": "Top View"
                    }
    scene_generation(item_image_content)
    # parse_resp(res)
    
