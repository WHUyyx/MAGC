import pyiqa
import torch
import os
import numpy as np
import torch

from PIL import Image

import shutil
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
# from pytorch_fid import fid_score
import pytorch_lightning as pl
import torch.nn as nn
import torch_fidelity


def get_three_imgs_names(root_path):
    filenames = os.listdir(root_path)
    filenames = [img for img in filenames if img.lower().endswith('.png')]
    return filenames


def save_to_single_img(root_path):
    img_or_path = os.path.join(root_path,'or')
    if not os.path.exists(img_or_path):
        os.mkdir(img_or_path)
    img_re_path = os.path.join(root_path,'re')
    if not os.path.exists(img_re_path):
        os.mkdir(img_re_path)
    three_imgs_names = get_three_imgs_names(root_path)

    w = 256

    # 把每个结果拆分成3个单独的，并分别保存
    for filename in three_imgs_names:
        _filename = filename.split('.')[0]
        img_path_three = os.path.join(root_path,filename)
        img_path_or = os.path.join(img_or_path,_filename+'.png')
        img_path_re = os.path.join(img_re_path,_filename+'.png')

        # 以此读取每张图像
        img_three = Image.open(img_path_three)

        # 提取第一张图片
        img_or_box = (0, 0, w, w)
        img_or = img_three.crop(img_or_box)

        # 提取第二张图片
        img_re_box = (w + 10, 0, w + w + 10, w)
        img_re = img_three.crop(img_re_box)

        img_or.save(img_path_or)
        img_re.save(img_path_re)

    return img_or_path, img_re_path

def train_files(file_,patch_size,overlap,p_max,lr_output_path,hr_output_path):
    lr_file, hr_file = file_
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)
    num_patch = 0
    w, h = lr_img.shape[:2]
    # if w > p_max and h > p_max:
    #     w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=np.int32)) # 192 32
    #     h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=np.int32))
    #     w1.append(w-patch_size)
    #     h1.append(h-patch_size)
    #     for i in w1:
    #         for j in h1:
    #             num_patch += 1
    #             lr_patch = lr_img[i:i+patch_size, j:j+patch_size,:]
    #             lr_savename = os.path.join(lr_output_path, filename + '-' + str(num_patch) + '.png')
    #             cv2.imwrite(lr_savename, lr_patch)
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-patch_size, patch_size, dtype=np.int32)) # 192 32
        h1 = list(np.arange(0, h-patch_size, patch_size, dtype=np.int32))
        w1.append(w-patch_size)
        h1.append(h-patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1
                lr_patch = lr_img[i:i+patch_size, j:j+patch_size,:]
                lr_savename = os.path.join(lr_output_path, filename + '-' + str(num_patch) + '.png')
                cv2.imwrite(lr_savename, lr_patch)

        w1 = list(np.arange(overlap, w-overlap, patch_size, dtype=np.int32)) 
        h1 = list(np.arange(overlap, h-overlap, patch_size, dtype=np.int32))
        for i in w1:
            for j in h1:
                num_patch += 1
                lr_patch = lr_img[i:i+patch_size, j:j+patch_size,:]
                lr_savename = os.path.join(lr_output_path, filename + '-' + str(num_patch) + '.png')
                cv2.imwrite(lr_savename, lr_patch)



    w, h = hr_img.shape[:2]
    # if w > p_max and h > p_max:
    #     w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=np.int32))
    #     h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=np.int32))
    #     w1.append(w-patch_size)
    #     h1.append(h-patch_size)
    #     for i in w1:
    #         for j in h1:
    #             num_patch += 1
    #             hr_patch = hr_img[i:i+patch_size, j:j+patch_size,:]
    #             hr_savename = os.path.join(hr_output_path, filename + '-' + str(num_patch) + '.png')
    #             cv2.imwrite(hr_savename, hr_patch)


    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-patch_size, patch_size, dtype=np.int32)) # 192 32
        h1 = list(np.arange(0, h-patch_size, patch_size, dtype=np.int32))
        w1.append(w-patch_size)
        h1.append(h-patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1
                hr_patch = hr_img[i:i+patch_size, j:j+patch_size,:]
                hr_savename = os.path.join(hr_output_path, filename + '-' + str(num_patch) + '.png')
                cv2.imwrite(hr_savename, hr_patch)

        w1 = list(np.arange(overlap, w-overlap, patch_size, dtype=np.int32)) 
        h1 = list(np.arange(overlap, h-overlap, patch_size, dtype=np.int32))
        for i in w1:
            for j in h1:
                num_patch += 1
                hr_patch = hr_img[i:i+patch_size, j:j+patch_size,:]
                hr_savename = os.path.join(hr_output_path, filename + '-' + str(num_patch) + '.png')
                cv2.imwrite(hr_savename, hr_patch)



def get_fid_from_path(root_path,patch_size = 128):
    # 将图像分别保存
    hr_input_path, lr_input_path = save_to_single_img(root_path)


    # 将分开的图像进行分块
    overlap = patch_size // 2
    p_max = 0
    base_input_path = os.path.dirname(lr_input_path)
    base_out_path = base_input_path + "/patches"
    lr_output_path = os.path.join(base_out_path, 'lr_{}'.format(patch_size))
    hr_output_path = os.path.join(base_out_path, 'hr_{}'.format(patch_size))

    os.makedirs(lr_output_path, exist_ok=True)
    os.makedirs(hr_output_path, exist_ok=True)

    lr_files = sorted(glob(os.path.join(lr_input_path, '*.png')))
    sr_files = sorted(glob(os.path.join(hr_input_path, '*.png')))

    files = [(i, j) for i, j in zip(lr_files, sr_files)]


    for file_ in tqdm(files):
        train_files(file_,patch_size,overlap,p_max,lr_output_path,hr_output_path) 


    print("-----------------Cropping finished!!! Start calculating FID--------------")
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=lr_output_path, 
        input2=hr_output_path, 
        cuda=True, 
        fid=True, 
        kid=True, 
        verbose=False,
    )
    print(metrics_dict)
    fid = metrics_dict['frechet_inception_distance']
    kid = metrics_dict['kernel_inception_distance_mean']
    # metrics_dict = torch_fidelity.calculate_metrics(
    #     input1=lr_output_path, 
    #     input2=hr_output_path, 
    #     cuda=True, 
    #     fid=True, 
    #     verbose=False,
    # )
    # print(metrics_dict)
    # fid = metrics_dict['frechet_inception_distance']

    # delete crops
    shutil.rmtree(base_out_path)
    shutil.rmtree(hr_input_path)
    shutil.rmtree(lr_input_path)
    return fid, kid
    # return fid


class single_iqa():
    def __init__(self,device=torch.device('cuda')):
        self.iqa_metrics = dict()
        self.iqa_metrics['lpips'] = pyiqa.create_metric('lpips', device=device)
        self.iqa_metrics['psnr'] = pyiqa.create_metric('psnr',test_y_channel=True, device=device)
        self.iqa_metrics['dists'] = pyiqa.create_metric('dists', device=device)
        self.iqa_metrics['msssim'] = pyiqa.create_metric('ms_ssim',test_y_channel=True, device=device)  
        self.iqa_metrics['niqe'] = pyiqa.create_metric('niqe', device=device)
        self.iqa_metrics['clipiqa'] = pyiqa.create_metric('clipiqa', device=device)
        self.iqa_metrics['musiq'] = pyiqa.create_metric('musiq', device=device)
        # self.iqa_metrics['maniqa'] = pyiqa.create_metric('maniqa', device=device)




    def cal_metrics(self, or_01, re_01):
        metrics_single = {}
        metrics_single["psnr"] = float(self.iqa_metrics['psnr'](or_01, re_01).item()),
        metrics_single["msssim"] = float(self.iqa_metrics['msssim'](or_01, re_01).item()),
        metrics_single["lpips"] = float(self.iqa_metrics['lpips'](or_01, re_01).item()),
        metrics_single["dists"] = float(self.iqa_metrics['dists'](or_01, re_01).item()),
        metrics_single["musiq"] = float(self.iqa_metrics['musiq'](re_01).item()),
        # metrics_single["niqe"] = float(self.iqa_metrics['niqe'](re_01).item()),
        metrics_single["clipiqa"] = float(self.iqa_metrics['clipiqa'](re_01).item()),
        # metrics_single["maniqa"] = float(self.iqa_metrics['maniqa'](re_01).item()),
        

        for i in metrics_single: # 去括号
            metrics_single[i] = metrics_single[i][0]

        return metrics_single
    
