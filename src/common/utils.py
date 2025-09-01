import base64
import cv2
from io import BytesIO
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from random import choice
import math

def imgBlend(img1, img2, alpha):
    alpha_map = alpha / np.max(alpha) * 1.0
    if len(alpha.shape) == 2 and len(img1.shape) == 3:
        alpha_map = np.expand_dims(alpha_map, axis=-1)
    return img1 * (1-alpha_map) + img2 * alpha_map

def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as err:
        print("Invalid base64 image")
        return None
        
# def encode_image_to_base64(image):
#     buffered = BytesIO()
#     image.save(buffered, format="PNG")
#     return "data:image/png;base64,"+base64.b64encode(buffered.getvalue()).decode("utf-8")

# 定义一个函数，检验 base64 串是否是图片数据
def is_image_base64(base64_string):
  # 尝试对 base64 串进行解码
    if base64_string.startswith("data:image/"):
        base64_string = base64_string.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(base64_string)))
        return True
    except Exception as err:
        return False
def encode_image_to_base64(image,if_uri = False):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    if if_uri:
        return "data:image/png;base64,"+base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
def padding_image(img):
    # 获取图片的宽和高
    width, height = img.size

    # 计算需要的背景的宽和高，使其能被64整除
    bg_width = (width + 63) // 64 * 64
    bg_height = (height +63) // 64 * 64

    # 创建一个白色的背景图片
    bg = Image.new("RGB", (bg_width, bg_height), "white")

    # 计算图片在背景中的位置，使其居中
    x = (bg_width - width) // 2
    y = (bg_height - height) // 2

    # 把图片粘贴到背景上
    bg.paste(img, (x, y))
    return bg,(x,y,width,height)

def crop_image(image_base64,bound,is_encode=True):
    # 打开b.png文件
    img = decode_base64_to_image(image_base64)
    # print(image_base64)
    # 裁剪图片
    cropped_img = img.crop(bound)
    if is_encode:
        return encode_image_to_base64(cropped_img,if_uri=False)
    else:
        return cropped_img

def mergeBlankObject(fr_image, bg_image, height, width, color=[255, 255, 255]):
    # Get the height and width of the fr_image image
    h, w = fr_image.shape[:2]

    # Calculate the top, bottom, left and right padding to make the small image center-aligned with the pure color image
    top = int( (height - h) * (2/3) )
    bottom = height - h - top
    left = (width - w) // 2
    right = width - w - left
    top = int((height - h) * (2 / 3))
    bottom = height - h - top
    left = (width - w) // 2
    right = width - w - left

    # Use cv2.copyMakeBorder to add padding to the small image with the same color as the pure color image
    small_padded = cv2.copyMakeBorder(fr_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # Use cv2.addWeighted to blend the two images with equal weights
    merged = cv2.addWeighted(bg_image, 0.0, small_padded, 1.0, 0)

    # cv2.imwrite("merged.jpg", merged)
    return merged

def fr_area_ratio(image, width, height, ratio=1.0):
    h, w = image.shape[:2]
    scale = min(height / h, width / w) * 0.75

    # 如果主体占原图的面积大于0.4，则将主体缩放，增加背景区域面积
    roi_ratio_thres = ratio
    roi_ratio = (w * h) / (height * width)
    new_scale = scale
    if (roi_ratio >= roi_ratio_thres):
        new_scale = (roi_ratio_thres / roi_ratio) ** 0.5
    scale = min(new_scale, scale)
    # Use cv2.resize to resize the image with the scaling factor and the cv2.INTER_AREA interpolation method
    resized_ratio = (w * scale * h * scale) / (width * height)
    return resized_ratio

def adjuest_img(image, width, height, ratio=1.0):
    h, w = image.shape[:2]
    scale = min(height / h, width / w) * 0.75

    # 如果主体占原图的面积大于0.4，则将主体缩放，增加背景区域面积
    roi_ratio_thres = ratio
    roi_ratio = (w * h) / (height * width)
    new_scale = scale
    if (roi_ratio >= roi_ratio_thres):
        new_scale = (roi_ratio_thres / roi_ratio) ** 0.5
    scale = min(new_scale, scale)
    # Use cv2.resize to resize the image with the scaling factor and the cv2.INTER_AREA interpolation method
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized

def img_layout(image, width, height, ratio=1.0):
    h, w = image.shape[:2]
    scale = min(height/h, width/w) * 0.9
    # 如果主体占原图的面积大于阈值，则将主体缩放，增加背景区域面积
    roi_ratio_thres = ratio
    roi_ratio = (scale * scale * w * h) / (height * width)
    new_scale = scale
    if (roi_ratio >= roi_ratio_thres):
        new_scale = scale * (roi_ratio_thres / roi_ratio) ** 0.5
    scale = min(new_scale, scale)

    new_w = int(scale * w)
    new_h = int(scale * h)
    x = int((width - new_w)/2)
    y = int((height - new_h)*2/3)
    return [x, y, new_w, new_h]


def getMidBoundingBox(fr_image, width, height):
    h, w = fr_image.shape[:2]
    top = int((height - h) * (2 / 3))
    bottom = height - h - top
    left = (width - w) // 2
    right = width - w - left
    return [top, bottom, left, right]

def extendBackgroudv3(fr_image, width, height, bbox, color=[255,255,255,0]):
    b_x, b_y, b_w, b_h = bbox
    fr_h, fr_w = fr_image.shape[:2]
    scale = min(b_w / fr_w, b_h / fr_h)
    new_w = int(fr_w * scale)
    new_h = int(fr_h * scale)
    resize_img = cv2.resize(fr_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    roi_x = b_x + int((b_w - new_w) / 2)
    roi_y = b_y + int((b_h - new_h) / 2)
    channel = len(color)
    if channel > 1 :
        bg_img = np.zeros((height, width, channel), dtype=np.uint8)
    if channel == 1:
        bg_img = np.zeros((height, width), dtype=np.uint8)
    # Fill the array with the color
    bg_img[:] = color
    print("roi op, bg image shape: {}, fr image shape: {}".format(bg_img.shape, resize_img.shape))
    bg_img[roi_y : roi_y + new_h, roi_x : roi_x+ new_w] = resize_img
    return bg_img

def extendBackgroudv2(fr_image, width, height, color=[255,255,255,0], ratio=1.0):
    result = adjuest_img(fr_image, width, height, ratio)
    channel = len(color)
    bg_img = np.zeros((height, width, channel), dtype=np.uint8)
    # Fill the array with the color
    bg_img[:] = color
    h, w = result.shape[:2]
    x = (width - w) // 2
    y = int((height - h) * (2 / 3))
    bg_img[y:y+h,x:x+w,:] = result
    return bg_img
    
def extendBackgroud(image, width, height, color=[255, 255, 255], ratio=1.0):
    # adjuest original image
    result = adjuest_img(image, width, height, ratio)
    channel = len(color)
    bg_img = np.zeros((height, width, channel), dtype=np.uint8)
    # Fill the array with the color
    bg_img[:] = color
    boundingBox = getMidBoundingBox(result, width, height)
    return mergeBlankObjectWithBox(result, bg_img, boundingBox, color)


def mergeBlankObjectWithBox(fr_image, bg_image, bounding_box, color=[255, 255, 255]):
    top = bounding_box[0]
    bottom = bounding_box[1]
    left = bounding_box[2]
    right = bounding_box[3]
    # Use cv2.copyMakeBorder to add padding to the small image with the same color as the pure color image
    small_padded = cv2.copyMakeBorder(fr_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # Use cv2.addWeighted to blend the two images with equal weights
    merged = cv2.addWeighted(bg_image, 0.0, small_padded, 1.0, 0)
    return merged

def resize_img(image, width, height, ratio=1.0):
    h, w = image.shape[:2]
    # Calculate the scaling factor based on the ratio of the longest side to the height or width
    scale = min(height / h, width / w) * min(ratio, 1.0)

    # Use cv2.resize to resize the image with the scaling factor and the cv2.INTER_AREA interpolation method
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized

def generateBlankBg(image, width, height, color=[255, 255, 255], ratio=1.0):
    # resize original image
    result = resize_img(image, width, height, ratio)
    # generate blank image
    # Create an empty numpy array of shape (height, width, 3)
    channel = len(color)
    bg_img = np.zeros((height, width, channel), dtype=np.uint8)
    # Fill the array with the color
    bg_img[:] = color
    return mergeBlankObject(result, bg_img, height, width, color)

def blendImgWithType(img1, img2, dtype, alpha=0.5):
    img1_dtype = cv2.cvtColor(img1, dtype)
    img2_dtype = cv2.cvtColor(img2, dtype)
    return (img1_dtype * alpha + img2_dtype * (1 - alpha)).astype(img1.dtype)

def fill(image, mask):
    """fills masked regions with colors from image using blur. Not extremely effective."""

    image_mod = Image.new('RGBA', (image.width, image.height))

    image_masked = Image.new('RGBa', (image.width, image.height))
    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert('L')))

    image_masked = image_masked.convert('RGBa')

    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert('RGBA')
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)
    return image_mod.convert("RGB")


def getAdaptiveBgConf(bg_confs, img_size):
    # 尝试上限
    retry_cnt = 3
    cnt = 0
    while True:
        bg_conf = choice(bg_confs)
        bg_roi = bg_conf["roi"]
        roi_x = int(bg_roi.split(',')[0])
        roi_y = int(bg_roi.split(',')[1])
        roi_w = int(bg_roi.split(',')[2])
        roi_h = int(bg_roi.split(',')[3])
        ori_h = img_size[3]
        ori_w = img_size[2]
        rz_ratio = min(roi_w / ori_w, roi_h / ori_h)
        new_h = int(ori_h * rz_ratio)
        new_w = int(ori_w * rz_ratio)
        cnt = cnt + 1
        # 当商品缩放之后面积占比大于50% 或者尝试次数超过上限，返回选择的背景
        if (new_w * new_h) / (ori_h * ori_w) >= 0.5 or cnt > retry_cnt:
            return bg_conf


def getObjShape(mask, roi):
    # 检测底部宽度
    l_x = roi[0]
    l_y = roi[1]
    r_x = roi[0] + roi[2]
    r_y = roi[1] + roi[3]
    roi_w = roi[2]
    thres = 3
    max_bottom_w = 0
    for i in range(r_y, r_y - thres, -1):
        bottom_w = 0
        for j in range(r_x, l_x, -1):
            if mask[i, j] == 255:
                bottom_w = bottom_w + 1
        max_bottom_w = max(max_bottom_w, bottom_w)

    objShapTypeWide = 1  # 宽底边，适合前视图 高度/贴地宽度 <= objShapThres
    objShapTypeNarrow = 2  # 债底边，适合顶视图或者悬浮,  高度/贴地宽度 > objShapThres
    # 判断形状
    if roi_w / max_bottom_w >= 20:
        return objShapTypeNarrow
    else:
        return objShapTypeWide


def getAdaptiveBgView(mask, roi):
    objShapeType = getObjShape(mask, roi)
    frontView = "front_view"
    flotageView = "flotage_view"
    if objShapeType == 1:
        return frontView
    else:
        return flotageView


def getMaskRect(mask):
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find contour with maximum area
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    # Get bounding rectangle of contour
    x, y, w, h = cv2.boundingRect(max_contour)
    return x, y, w, h, max_contour


def imgCut(rect, img, mask):
    SIZE = (1, 65)
    bgdModle = np.zeros(SIZE, np.float64)
    fgdModle = np.zeros(SIZE, np.float64)
    cv2.grabCut(img, mask, rect, bgdModle, fgdModle, 50, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    a = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    cutImg = img * mask2[:, :, np.newaxis]
    b, g, r = cv2.split(cutImg)
    bgra = cv2.merge([b, g, r, a])
    return bgra


# def imgBlend(img1, img2):
#     for i in range(img1.shape[0]):
#         for j in range(img1.shape[1]):
#             alpha = img2[i, j][3] / 255.0
#             for c in range(0, 3):
#                 img1[i, j][c] = img1[i, j][c] * (1 - alpha) + img2[i, j][c] * alpha
#     return img1

def edgeSoomth(img, mask):
    # 轮廓检测
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        for j in range(len(cnt)):
            x, y = cnt[j][0]
            print(x, y)


def maskShrink(mask, shrink_width):
    # mask = cv2.bitwise_not(mask)
    # 对图像进行腐蚀操作，迭代次数为1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    shrink_width = min(max(int(shrink_width), 1), 4)
    erosion = cv2.erode(mask, kernel, iterations=shrink_width)
    # random = np.random.randint(0, 10, size=erosion.shape)
    # # 将随机矩阵中的1变为255，得到一个随机的二值图像
    # random = np.where(random == 1, 0, 255)
    # # 将结果图像和随机二值图像进行按位与操作，得到一个部分白色变为黑色的图像
    # random = random.astype(np.uint8)

    # erosion = cv2.bitwise_and(erosion, random)

    # 对图像进行膨胀操作，迭代次数为1
    # dilation = cv2.dilate(erosion, kernel, iterations=10)
    # dilation = cv2.bitwise_not(dilation)
    return erosion


def maskDilate(mask, width):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    width = min(max(int(width), 1), 5)
    dilated_mask = cv2.dilate(mask, kernel, iterations=width)
    return dilated_mask


def imgCompose(bg_img, img, bg_roi_args):
    roi_x = int(bg_roi_args.split(',')[0])
    roi_y = int(bg_roi_args.split(',')[1])
    roi_w = int(bg_roi_args.split(',')[2])
    roi_h = int(bg_roi_args.split(',')[3])
    ori_h, ori_w = img.shape[:2]
    rz_ratio = min(roi_w / ori_w, roi_h / ori_h)
    img_dst = cv2.resize(img, None, fx=rz_ratio, fy=rz_ratio, interpolation=cv2.INTER_LINEAR)

    bg_h, bg_w = bg_img.shape[:2]
    img_h, img_w = img_dst.shape[:2]
    # 计算图像在背景图中的位置
    x = roi_x + (roi_w - img_w) // 2
    y = roi_y + (roi_h - img_h)
    # 将图像叠加到背景图的中心位置
    bg_b, bg_g, bg_r = cv2.split(bg_img)
    bg_alpha = np.ones(bg_b.shape, dtype=bg_b.dtype) * 255
    bg_img = cv2.merge([bg_b, bg_g, bg_r, bg_alpha])
    bg_roi = bg_img[y:y + img_h, x:x + img_w]
    # img_compose = cv2.addWeighted(bg_roi, 0, img_dst, 1, 0)
    img_compose = imgBlend(bg_roi, img_dst, img_dst[:, :, 3])
    bg_img[y:y + img_h, x:x + img_w] = img_compose

    # 建立对应的mask图
    bg_black = np.zeros((bg_h, bg_w), dtype="uint8")
    bg_black[y:y + img_h, x:x + img_w] = img_dst[:, :, 3]
    return bg_img, bg_black


def convertMat2Image(image):
    # 将图像从BGR格式转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 将图像从NumPy数组转换为PIL.Image格式
    image = Image.fromarray(image)
    return image

def convertMat2ImageWithALpha(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    # 将图像从NumPy数组转换为PIL.Image格式
    image = Image.fromarray(image)
    return image
def areaNMS(ori_mask, max_contour):
    # Assume you have an image called img and a contour called cnt
    mask = np.zeros(ori_mask.shape, dtype=np.uint8)  # Create a blank mask
    # Assume you have an image called img and a contour called cnt
    mask = cv2.fillPoly(mask, [max_contour], (1, 1, 1)).astype('uint8')  # Fill the polygon with label
    return mask * ori_mask

# def get_prompt(bg_view_type, fuse_type, prompt):
#     base_prompt = "Product photography , placeholder , natural light and shadow , realistic , isolated on light gray background , commercial photography ,4k, Unreal Engine , realistic rendering"
#     if prompt is not None and len(prompt) > 0:
#         promptStyle = "customized"
#         newprompt = base_prompt.replace("placeholder", prompt)
#         return newprompt, promptStyle
#     elif bg_view_type in self.promptConf.keys():
#         promptDicts = self.promptConf[bg_view_type]
#         newprompt = ""
#         promptStyle = ""
#         if fuse_type is not None and len(fuse_type) > 0:
#             for promptDict in promptDicts:
#                 if fuse_type == promptDict["promptStyle"]:
#                     prompt = promptDict["prompt"]
#                     promptStyle = promptDict["promptStyle"]
#                     newprompt = base_prompt.replace("placeholder", prompt)
#         if len(newprompt) == 0:
#             promptDict = choice(promptDicts)
#             promptStyle = promptDict["promptStyle"]
#             prompt = promptDict["prompt"]
#             newprompt = base_prompt.replace("placeholder", prompt)
#         return newprompt, promptStyle
#     raise Exception("Invalid fuse_type")

def get_negative_prompt():
    negative_prompt = "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), low res, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, bad legs, error legs, bad feet,low-res, bad anatomy, bad hands, text error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, fused body, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation, UnrealisticDream,(nsfw:2),(nude)"
    return negative_prompt

def outputImgIO(img, image_format="JPEG"):
    buf = BytesIO()
    # Save the image to the buffer with a format
    img.save(buf, format=image_format)
    # Get the bytes from the buffer
    byte_im = buf.getvalue()
    return byte_im

def getMaskOuterRect(mask):
    # Find external contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Merge all contours into one
    merged_contour = cv2.convexHull(np.concatenate(contours))
    # Get bounding rectangle of merged contour
    x, y, w, h = cv2.boundingRect(merged_contour)
    return x, y, w, h, merged_contour

def pad_image(image,image_size):
    # image = Image.fromarray(image)
    width, height = image.size
    # 指定输出的大小
    output_size = image_size
    # 计算缩放比例
    scale = min(output_size[0] / width, output_size[1] / height)
    # 调整图像大小
    resized_image = image.resize((int(width * scale), int(height * scale)))
    # 获取调整后的宽高
    new_width, new_height = resized_image.size
    # 创建一个白色的背景图像
    padded_image = Image.new("RGB", output_size, (255, 255, 255))
    # 计算需要粘贴的位置
    x = (output_size[0] - new_width) // 2
    y = (output_size[1] - new_height) // 2
    # 将调整后的图像粘贴到背景图像上
    padded_image.paste(resized_image, (x, y))
    output_array = np.array(padded_image)
    return output_array

def select_max_region_v2(mask, onlyMax=True):
    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    #stats表示每个连通组件的统计信息，包括左上角坐标、宽度、高度和面积，centroids表示每个连通组件的质心坐标
    #labels是一个与image一样大小的矩形（labels.shape = image.shape），其中每一个连通区域会有一个唯一标识，标识从0开始。
    # print(stats)
    background = 0
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            background = row
    stats_no_bg = np.delete(stats, background, axis=0)
    max_idx = stats_no_bg[:, 4].argmax()
    max_region = np.where(labels==max_idx+1, 1, 0)
    max_area = stats[max_idx+1,:][4]

    if not onlyMax:
        second_idx = np.argsort(stats_no_bg[:, 4])[-2]
        second_region = np.where(labels==second_idx+1, 1, 0)
        second_area = stats[second_idx+1,:][4]
        if second_area >= 0.7 *max_area :
            return max_region + second_region
        else:
            return max_region
    else:
        return max_region

def imgBlendv2(bg_img, fr_img):
    ## 先将通道分离
    b,g,r,a = cv2.split(fr_img)
    #得到PNG图像前景部分，在这个图片中就是除去Alpha通道的部分
    foreground = cv2.merge((b,g,r))
    #得到PNG图像的alpha通道，即alpha掩模
    alpha = cv2.merge((a,a,a))
    #因为下面要进行乘法运算故将数据类型设为float，防止溢出
    foreground = foreground.astype(float)
    background = bg_img.astype(float)
    print("imgBlendv2 bg shape: ", background.shape, "bg dtype: ", background.dtype)
    print("imgBlendv2 fr shape: ", foreground.shape, "fr dtype: ", foreground.dtype)
    #将alpha的值归一化在0-1之间，作为加权系数
    alpha = alpha.astype(float)/255
    #将前景和背景进行加权，每个像素的加权系数即为alpha掩模对应位置像素的值，前景部分为1，背景部分为0
    foreground = cv2.multiply(alpha,foreground)
    background = cv2.multiply(1-alpha,background)
    outImage = foreground + background
    outImage = outImage.astype('uint8')
    # print("outImage shape: ", outImage.shape, "outImage dtype: ", outImage.dtype)
    return outImage

def obj_compose(fr_img, bg_color):
    height, width = fr_img.shape[:2]
    bg_img = np.ones((height, width, 3), dtype=np.uint8)
    bg_img[:] = bg_color
    image_compose = imgBlendv2(bg_img, fr_img)
    # cv2.imwrite("res.jpg", image_compose)
    return image_compose

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

def rgb2hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df / mx
    v = mx
    return h, s, v

def hex2rgb(hex):
    hex = hex.replace('#','')
    r = int(hex[0:2], 16)
    g = int(hex[2:4], 16)
    b = int(hex[4:6], 16)
    #     rgb = str(r)+','+str(g)+','+str(b)
    rgb = (r, g, b)
    return rgb

def rgb2hex(r,g,b):
    colors = [r,g,b]
    hex_str = '#'
    for i in colors:
        num = int(i)
        hex_str += str(hex(num))[-2:].replace('x', '0').upper()
    #     print(color)
    return hex_str

def rgba2rgb(image, background=(255,255,255)):
    template = Image.new("RGB", image.size, background)
    template.paste(image, mask=image.split()[3]) #split alpha channel
    return template

