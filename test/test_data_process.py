import numpy as np
import pandas as pd

def process():
    file_path = '/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/Data/BG60K/bg60k_caption_res.csv'
    df = pd.read_csv(file_path)
    df["image_url"] = df["image_url"].str.replace("user", "workspace/user", regex=False)
    df["mask_image_url"] = df["mask_image_url"].str.replace("user", "workspace/user", regex=False)
    df.to_csv("/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/Data/BG60K/bg60k_caption_img.csv", index=False)
    

if __name__ == '__main__':
    process()