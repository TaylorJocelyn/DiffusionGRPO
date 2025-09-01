import os
import sys
import yaml
# sys.path.append("/mnt/workspace/zengdawei.zdw/tmp/OminiControl/src/test")
from src.test.test_infer import TestPipe

def get_config(config_path):
    # config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config



config_path = "/mnt/workspace/zengdawei.zdw/tmp/OminiControl/train/config/mulcond_512.yaml"
config = get_config(config_path)
model_config = config.get("model", {})

test_pipe = TestPipe(
    lora_path="/mnt/workspace/zengdawei.zdw/tmp/OminiControl/runs/20250313-034856/ckpt/20000",
    lora_name="pytorch_lora_weights.safetensors",
    model_config=model_config,
    condition_types=["subject"]
)

save_path = "./"
if not os.path.exists(save_path):
    os.makedirs(save_path)
file_name="test_step20000_161"

input_image = "/mnt/workspace/zengdawei.zdw/Data/BG60K/bg60k_masks_1/161_mask.png"

prompt = "Minimalist setting with a flowing white fabric background, enhancing the product's sleek design and modern appeal."



res_images = test_pipe.generate_a_sample(
        image_path=input_image,
        prompt=prompt,
        condition_size=512,
        target_size=512,
)


res_images[0].save(
    os.path.join(save_path, f"{file_name}_multi_cond_sub.jpg")
)
