from datetime import datetime
import os
import random
import torch
import numpy as np
import math
import skimage
import cv2


def tprint(s):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {s}")


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dict2str(d, indent_l=1):
    """
    dict to string for logger
    """
    msg = ''
    for k, v in d.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + '\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def tensor2img(tensor, out_type=np.uint8, min_max=(-0.5, 0.5)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(1,C,H,W) or 3D(C,H,W), any range, RGB channel order
    Output: 3D(H,W,C) [0,255], np.uint8 (default)
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

    img_np = tensor.numpy()
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)


def save_img(img, img_path):
    cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2, edge_trim=0):
    """
    calculate PSNR
    pixel range: [0, 255]
    """
    if edge_trim != 0:
        img1 = img1[edge_trim:-edge_trim, edge_trim:-edge_trim, :]
        img2 = img2[edge_trim:-edge_trim, edge_trim:-edge_trim, :]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        # return float('inf')
        return 10000.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2, edge_trim=0):
    """
    calculate SSIM
    pixel range: [0, 255]
    """
    if edge_trim != 0:
        img1 = img1[edge_trim:-edge_trim, edge_trim:-edge_trim, :]
        img2 = img2[edge_trim:-edge_trim, edge_trim:-edge_trim, :]

    return skimage.metrics.structural_similarity(img1, img2, channel_axis=2)


class AvgMeter:
    def __init__(self):
        self.a = 0.0
        self.b = 0.0
        self.n = 0

    def update(self, x, y):
        self.a += x
        self.b += y
        self.n += 1

    def reset(self):
        self.__init__()

    def result(self):
        return (0, 0) if self.n == 0 else (self.a / self.n, self.b / self.n)
    

class AvgMeter3:
    def __init__(self):
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.n = 0

    def update(self, x, y, j):
        self.a += x
        self.b += y
        self.c += j
        self.n += 1

    def reset(self):
        self.__init__()

    def result(self):
        return (0, 0, 0) if self.n == 0 else (self.a / self.n, self.b / self.n, self.c / self.n)
    

class AvgMeter4:
    def __init__(self):
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.d = 0.0
        self.n = 0

    def update(self, x, y, j, k):
        self.a += x
        self.b += y
        self.c += j
        self.d += k
        self.n += 1

    def reset(self):
        self.__init__()

    def result(self):
        return (0, 0, 0) if self.n == 0 else (self.a / self.n, self.b / self.n, self.c / self.n, self.d / self.n)
