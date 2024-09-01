import cv2
import torch
import numpy as np
from torch.utils import data
import random
from config import config
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

import torchvision.transforms as transforms

from torch.nn.utils.rnn import pad_sequence
import albumentations as A

def random_mirror(rgb, gt):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)

    return rgb, gt
def random_mirror_cls(rgb):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)

    return rgb

def random_scale(rgb, gt, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    # print(type(rgb), type(gt), type(modal_x))
    # print(type(depth))
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    return rgb, gt, scale
def random_scale_cls(rgb, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    # print(type(rgb), type(gt), type(modal_x))
    # print(type(depth))
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, scale

def collate_fn(batch):
  return {
      'data': torch.stack([x['data'] for x in batch]),
      # 'label': torch.stack([x['label'] for x in batch]),
      'label': [x['label'] for x in batch],
      'path': [x['path'] for x in batch],
      'ix': [x['ix'] for x in batch]
  }
def collate_fn_val(batch):
  return {
      'data': torch.stack([x['data'] for x in batch]),
      # 'label': torch.stack([x['label'] for x in batch])
      'label': [x['label'] for x in batch]
  }

# class TrainPre(object):
#     def __init__(self, norm_mean, norm_std):
#         self.norm_mean = norm_mean
#         self.norm_std = norm_std
#         self.to_tensor = transforms.ToTensor()
#     def __call__(self, rgb, gt):
#         # rgb, gt = random_mirror(rgb, gt)
#         rgb = random_mirror_cls(rgb)
#         if config.train_scale_array is not None:
#             # rgb, gt, scale = random_scale(rgb, gt, config.train_scale_array)
#             rgb, scale = random_scale_cls(rgb, config.train_scale_array)
#         rgb = normalize(rgb, self.norm_mean, self.norm_std)
#         crop_size = (config.image_height, config.image_width)
#         crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)
#         p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
#         p_rgb = p_rgb.transpose(2, 0, 1)
#         # gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 0)
#
#         return p_rgb, gt
class TrainPre(object):
    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.to_tensor = transforms.ToTensor()
        self.num = 0

        basic_transform = [
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(350, 560),
        ]
        self.basic_transform = basic_transform

    def __call__(self, rgb, rgb_path):
        # rgb = random_mirror_cls(rgb)
        # cv2.imwrite(f'/media/jslee/Data2/jslee_two/jisu/RGBX_scene/12/rgb_image/sss{self.num}.jpg', rgb)
        # if config.train_scale_array is not None:
        #     rgb, scale = random_scale_cls(rgb, config.train_scale_array)
        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        # crop_size = (config.image_height, config.image_width)
        # crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)
        # p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)

        aug = A.Compose(transforms=self.basic_transform)
        augmented = aug(image=rgb)
        p_rgb = augmented['image']
        # print(p_rgb.shape) h * w * c
        # cv2.imwrite(f'/media/jslee/Data2/jslee_two/jisu/RGBX_scene/12/rgb_image/sss{self.num}_crop.jpg', p_rgb)
        # self.num += 1

        p_rgb = p_rgb.transpose(2, 0, 1)
        return p_rgb
class ValPre(object):
    def __call__(self, rgb, gt, modal_x):
        return rgb, gt, modal_x

def get_train_loader(engine, dataset):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_root_n': config.rgb_root_folder_n,
                    'rgb_root_eval': config.rgb_root_folder_eval,
                    'rgb_root_eval_n': config.rgb_root_folder_eval_n,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_root_eval': config.gt_root_folder_eval,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root': config.x_root_folder,
                    'x_root2': config.x_root_folder2,

                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    train_preprocess = TrainPre(config.norm_mean, config.norm_std)
    train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)
    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler,
                                   collate_fn=collate_fn)

    return train_loader, train_sampler


def get_val_loader(engine, dataset):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_root_n': config.rgb_root_folder_n,
                    'rgb_root_eval': config.rgb_root_folder_eval,
                    'rgb_root_eval_n': config.rgb_root_folder_eval_n,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_root_eval': config.gt_root_folder_eval,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root': config.x_root_folder_eval,
                    'x_root2': config.x_root_folder_eval2,

                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    val_pre = TrainPre(config.norm_mean, config.norm_std)
    # val_pre = ValPre()
    val_dataset = dataset(data_setting, 'val', val_pre, 652)

    val_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    val_loader = data.DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 num_workers=config.num_workers,
                                 drop_last=True,
                                 shuffle=is_shuffle,
                                 pin_memory=True,
                                 sampler=val_sampler,
                                 collate_fn=collate_fn_val
                                 )
    return val_loader, val_sampler