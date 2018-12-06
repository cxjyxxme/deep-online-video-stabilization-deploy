from configs.v2_93 import *
import cv2
from PIL import Image
import numpy as np
DEBUG = True
def cvt_img2train(img, crop_rate = 1):
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    if (crop_rate != 1):
        h = int(height / crop_rate)
        dh = int((h - height) / 2)
        w = int(width / crop_rate)
        dw = int((w - width) / 2)

        img = img.resize((w, h), Image.BILINEAR)
        img = img.crop((dw, dh, dw + width, dh + height))
    else:
        img = img.resize((width, height), Image.BILINEAR)
    img = np.array(img)
    img = img * (1. / 255) - 0.5
    img = img.reshape((1, height, width, 1))
    return img

