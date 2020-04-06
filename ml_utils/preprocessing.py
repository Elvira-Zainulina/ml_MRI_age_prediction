import torchio
from torchio.transforms import HistogramStandardization
import numpy as np
import pandas as pd
import os
from scipy import ndimage as nd
from scipy.stats import mode
from torch.utils.data import DataLoader
import multiprocessing
from tqdm.notebook import tqdm


lab2num = {'22-25': 0, '26-30': 1, '31-35': 2, '36+': 3}
num2lab = {0: '22-25', 1: '26-30', 2: '31-35', 3: '36+'}
num2gender = {0: 'M', 1: 'F'}

def map_gender(genders):
    return np.array(list(map(lambda x: 1 if x=='F' else 0, genders)))


def process_ages(ages):
    return np.array(list(map(lambda x: lab2num[x], ages)))


def regroup_ages(ages):
    return np.array(list(map(lambda x: 0 if x < 2 else 1, ages)))


def cut_img_mask(img):
    mask = np.ones_like(img)
    condition = (np.sum(img, axis=(0, 2)) == 0)
    mask[:, condition, :] = 0
    condition = (np.sum(img, axis=(0, 1)) == 0)
    mask[:, :, condition] = 0
    condition = (np.sum(img, axis=(1, 2)) == 0)
    mask[condition, :, :] = 0
    return mask.astype(bool)


def get_hist_landmarks(img_path, lndmrk_path, names):
    image_paths = list(map(lambda x: os.path.join(img_path, str(x), 'T1w', 
                                                  'T1w_acpc_dc_restore_brain.nii.gz'), 
                       names))
    landmarks = HistogramStandardization.train(image_paths, 
                                               masking_function=cut_img_mask,
                                               output_path=lndmrk_path)
    return landmarks


def rescale_crop_img(img, size=128):
    H, Z, W = img.shape
    img_scale = size / min(H, Z, W)
    img = nd.interpolation.zoom(img, img_scale)
    h_center = img.shape[0] // 2
    z_center = img.shape[1] // 2
    w_center = img.shape[2] // 2
    s = size // 2
    img = img[max(0, h_center - s):min(h_center + s, img.shape[0]),
              max(0, z_center - s):min(z_center + s, img.shape[1]), 
              max(0, w_center - s):min(w_center + s, img.shape[2])]
    return img


def rescale_pad_img(img, size=128):
    H, Z, W = img.shape
    img_scale = size / max(H, Z, W)
    img = nd.interpolation.zoom(img, img_scale)
    h_pad = (size - img.shape[0]) // 2
    z_pad = (size - img.shape[1]) // 2
    w_pad = (size - img.shape[2]) // 2
    h_v1 = mode(img[0, :, :].ravel())[0][0]
    z_v1 = mode(img[:, 0, :].ravel())[0][0]
    w_v1 = mode(img[:, :, 0].ravel())[0][0]
    h_v2 = mode(img[-1, :, :].ravel())[0][0]
    z_v2 = mode(img[:, -1, :].ravel())[0][0]
    w_v2 = mode(img[:, :, -1].ravel())[0][0]
    img = np.pad(img, ((h_pad, h_pad + img.shape[0] % 2),
                       (z_pad, z_pad + img.shape[1] % 2),
                       (w_pad, w_pad + img.shape[2] % 2)),
                 constant_values=((h_v1, h_v2), 
                                  (z_v1, z_v2),
                                  (w_v1, w_v2)))
    return img


def cut_img(img):
    condition = (np.sum(img, axis=(0, 2)) != 0)
    img = img[:, condition, :]
    condition = (np.sum(img, axis=(0, 1)) != 0)
    img = img[:, :, condition]
    condition = (np.sum(img, axis=(1, 2)) != 0)
    img = img[condition, :, :]
    return img


def process_dataset(dataset, out_path=None):
    imgs = []
    genders = []
    ages = []
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, 
                            num_workers=multiprocessing.cpu_count())
    for sample in tqdm(dataloader):
        img, gender, age = sample
        imgs.append(img['MRI'][torchio.DATA].numpy())
        ages.append(age.numpy())
        genders.append(gender.numpy())
    imgs = np.concatenate(imgs, axis=0)
    genders = np.concatenate(genders)
    ages = np.concatenate(ages)
    if not (out_path is None):
        with open(out_path, 'wb') as data_file:
            np.savez(data_file, images=imgs, genders=genders, ages=ages)
    return imgs, genders, ages