import torch
import cv2 as cv
import json
import numpy as np
import scipy.stats as st
import os
from config import Config

CONFIG = Config()

def print_params_size(file_path):

    dic_params = torch.load(file_path)
    for i, key in enumerate(dic_params,0):
        print(i, ',', key, ',', dic_params[key].shape)

def print_net_ParamDict(net):

    dic = net.state_dict()
    for i, key in enumerate(dic,0):
        print(i, ',', key, ',', dic[key].shape)

def load_json(j_fn):

    with open(j_fn,'r') as f:
        data = json.load(f)
    return data

def save_json(dic,j_fn):

    json_str = json.dumps(dic)
    with open(j_fn,'w') as json_file:
        json_file.write(json_str)


def normalizeF(input_img, mean, std):

    output_img = torch.zeros(input_img.shape)
    input_img = (input_img * 2 + 128)/255.0
    output_img[:, 0, :, :] = (input_img[:, 0, :, :] - mean) / std
    return output_img


def get_GaussKernel(kernlen, nsig):

    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2.,nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)

    return kernel

def get_GaussKernel_2(kernlen, nsig):
    kx = cv.getGaussianKernel(kernlen, nsig)
    ky = cv.getGaussianKernel(kernlen, nsig)
    return torch.FloatTensor(np.multiply(kx,np.transpose(ky))).unsqueeze(0).unsqueeze(0)


def calc_psnr(img1, img2, ignore=0, cpsnr=False):

    if cpsnr:
        if ignore > 0:
            rpsnr = 10. * torch.log10(1. / torch.mean((img1[:, 0, ignore:-ignore, ignore:-ignore] - img2[:, 0, ignore:-ignore, ignore:-ignore]) ** 2))
            gpsnr = 10. * torch.log10(1. / torch.mean((img1[:, 1, ignore:-ignore, ignore:-ignore] - img2[:, 1, ignore:-ignore, ignore:-ignore]) ** 2))
            bpsnr = 10. * torch.log10(1. / torch.mean((img1[:, 2, ignore:-ignore, ignore:-ignore] - img2[:, 2, ignore:-ignore, ignore:-ignore]) ** 2))
            psnr = 10. * torch.log10(1. / torch.mean((img1[:, :, ignore:-ignore, ignore:-ignore] - img2[:, :, ignore:-ignore, ignore:-ignore]) ** 2))
        else:
            rpsnr = 10. * torch.log10(1. / torch.mean((img1[:, 0, :, :] - img2[:, 0, :, :]) ** 2))
            gpsnr = 10. * torch.log10(1. / torch.mean((img1[:, 1, :, :] - img2[:, 1, :, :]) ** 2))
            bpsnr = 10. * torch.log10(1. / torch.mean((img1[:, 2, :, :] - img2[:, 2, :, :]) ** 2))
            psnr = 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))
        return np.array((rpsnr.cpu(), gpsnr.cpu(), bpsnr.cpu(), psnr.cpu()), dtype = np.float32)
    else:
        if ignore > 0:
            psnr = 10. * torch.log10(1. / torch.mean((img1[:, :, ignore:-ignore, ignore:-ignore] - img2[:, :, ignore:-ignore, ignore:-ignore]) ** 2))
        else:
            psnr = 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))
        return psnr.cpu()


def rgb2RGGB(img):

    r = img[:, 0, 1::2, 0::2].unsqueeze(1)
    g1 = img[:, 1, 0::2, 0::2].unsqueeze(1)
    g2 = img[:, 1, 1::2, 1::2].unsqueeze(1)
    b = img[:, 2, 0::2, 1::2].unsqueeze(1)

    return torch.cat((r,g1,g2,b),1)


def imgSIZEnormalize(img):  # Correction of images with a size of an odd number
    if img.shape[2] % 2 != 0:
        img = img[:,:,1:,:]
    if img.shape[3] % 2 != 0:
        img = img[:, :, :, 1:]
    return img


def ResizeCrop(input_im, reSIZE):
    input_img = np.array(input_im)
    h, w, c = input_img.shape
    # h = float(h)
    # w = float(w)
    if h / w == reSIZE[0] / reSIZE[1]:  # important!!!
        img_crop = cv.resize(input_img, (reSIZE[1], reSIZE[0]))
    else:
        fac = np.array([reSIZE[0] / h, reSIZE[1] / w], dtype='float32')
        scale, idx = torch.max(torch.from_numpy(fac), 0)
        scale = scale.numpy()
        if idx == 1:
            input_img = cv.resize(input_img, (reSIZE[1], int(np.ceil(scale * h))))
            start_pos = int(np.floor(int(np.floor(scale * h)) / 2 - reSIZE[0] / 2))
            img_crop = input_img[start_pos: start_pos + reSIZE[0], :, :]
        else:
            input_img = cv.resize(input_img, (int(np.ceil(scale * w)), reSIZE[0]))
            start_pos = int(np.floor(int(np.floor(scale * w)) / 2 - reSIZE[1] / 2))
            img_crop = input_img[:, start_pos: start_pos + reSIZE[1], :]
    return img_crop


def Crop(input_image, c=48):
    I = np.array(input_image).squeeze(0)
    I = np.pad(I, [(0, 0), (c, c), (c, c)], 'symmetric')
    I_tensor = torch.from_numpy(I).unsqueeze(0)
    return I_tensor


def unCrop(input_images, c=48):
    for i, img_L in enumerate(input_images):
        input_images[i] = img_L[:, :, c:-c, c:-c]
    return input_images


def _Debug_show(I):
    import matplotlib.pyplot as plt
    I = np.array(I.cpu())
    pic = (I * 255).clip(0, 255).astype(np.uint8).squeeze(0)
    pic = pic.transpose(1, 2, 0)
    plt.imshow(pic)
    plt.show()


if __name__ == "__main__":
    pass