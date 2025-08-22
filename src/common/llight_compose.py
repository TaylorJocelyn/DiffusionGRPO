import numpy as np
from common.utils import resize_img,blendImgWithType,imgBlend
import os
import cv2

# import pdb

def compose_light_bloom(fr_img, bg_img, mask_img, skip=False, ratio=0.5, ttype="HSV"):
    assert len(fr_img.shape) == len(bg_img.shape), print("shape of input1 and input2 is not equal")
    height, width = mask_img.shape[:2]
    mask_img = np.reshape(mask_img, [height, width])
    bg_img = resize_img(bg_img, width, height)

    if skip:
        new_img = imgBlend(bg_img, fr_img, mask_img)
        return new_img.astype('uint8')

    # get Y channel of bg_img within mask
    if ttype == "HSV":
        cv_type = cv2.COLOR_BGR2HSV
        rcv_type = cv2.COLOR_HSV2BGR
    else:
        cv_type = cv2.COLOR_BGR2HLS
        rcv_type = cv2.COLOR_HLS2BGR

    object_hue = blendImgWithType(bg_img, fr_img, cv_type, ratio)
    object_hue_l = object_hue[:,:,1]
    object_hue_s = object_hue[:,:,2]

    fr_img_hsl = cv2.cvtColor(fr_img, cv_type)
    fr_img_hsl[:,:,1] = object_hue_l
    if ttype == "HSV":
        fr_img_hsl[:,:,2] = object_hue_s
    fr_img[:,:,:3] = cv2.cvtColor(fr_img_hsl, rcv_type)

    # orignal composed image
    new_img = imgBlend(bg_img, fr_img, mask_img).astype('uint8')

    return new_img


    # droplet_img_name = "/mnt/workspace/aigc/image_aigc_service_icanvas/dataset/amazon/res_imgs/droplet.png"
    # droplet_img = cv2.imread(droplet_img_name, cv2.IMREAD_UNCHANGED)
    # resized = cv2.resize(droplet_img, (256, 256), interpolation=cv2.INTER_AREA)
    # # factor = max(height, width) // 128
    # factor = 3
    # tile_droplet = np.tile(resized, (factor,factor,1))
    # tile_droplet = cv2.resize(tile_droplet, (height, width), interpolation=cv2.INTER_AREA)
    # new_droplet_img = imgBlend(new_img, tile_droplet[:,:,:3], tile_droplet[:,:,3])
    # new_droplet_img = np.where(mask_img > 0, new_droplet_img, new_img)
