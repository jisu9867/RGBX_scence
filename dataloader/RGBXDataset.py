import os
from pickletools import uint8
import cv2
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data

import h5py
def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s

class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting['rgb_root']
        self._rgb_path_n = setting['rgb_root_n']
        self._rgb_path_eval = setting['rgb_root_eval']
        self._rgb_path_eval_n = setting['rgb_root_eval_n']
        self._rgb_format = setting['rgb_format']

        self._gt_path = setting['gt_root']
        self._gt_path_eval = setting['gt_root_eval']
        self._gt_format = setting['gt_format']
        self._transform_gt = setting['transform_gt']

        self._x_path = setting['x_root']
        self._x_path2 = setting['x_root2']
        self._x_format = setting['x_format']
        self._x_single_channel = setting['x_single_channel']

        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self.class_names = setting['class_names']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        # if self._file_length is not None:
        #     item_name = self._construct_new_file_names(self._file_length)[index]
        # else:
        #     item_name = self._file_names[index]
        item_name = self._file_names[index]
        # print("item name2: ", item_name)
        if self._split_name == 'train':
            if index < 2400:
                rgb_path = os.path.join(self._rgb_path, remove_leading_slash(item_name.split()[0]))
                # depth_gt_path = os.path.join(self._rgb_path, remove_leading_slash(item_name.split()[1])) # for no depth (different txt file)
                gt = 0
                # print("kitti")
            else:
                rgb_path = os.path.join(self._rgb_path_n, remove_leading_slash(item_name.split()[0]))
                # depth_gt_path = os.path.join(self._rgb_path_n, remove_leading_slash(item_name.split()[1])) # different txt file
                gt = 1
                # print("nyu")
        else:
            if index < 652:
                rgb_path = os.path.join(self._rgb_path_eval, remove_leading_slash(item_name.split()[0]))
                # depth_gt_path = os.path.join(self._rgb_path_eval, remove_leading_slash(item_name.split()[1])) # different txt file
                gt = 0
                # print("kitti")
            else:
                rgb_path = os.path.join(self._rgb_path_eval_n, remove_leading_slash(item_name.split()[0]))
                # depth_gt_path = os.path.join(self._rgb_path_eval_n, remove_leading_slash(item_name.split()[1])) # different txt file
                gt = 1
                # print("nyu")
        rgb_original = self._open_image(rgb_path, cv2.COLOR_BGR2RGB, dtype=np.float32)
        # depth = cv2.imread(depth_gt_path, cv2.IMREAD_UNCHANGED).astype('float32') # different txt file
        if gt == 1:
            rgb_original = rgb_original[45:471, 41:601]
        # print(rgb_original.shape) h * w * c
        if self.preprocess is not None:
            rgb = self.preprocess(rgb_original, rgb_path)
            # depth = depth / 1000 # different txt file
        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float() #for collate_fn_val
        output_dict = dict(data=rgb, label=gt, fn=str(item_name), n=len(self._file_names), ix= index, path= rgb_path) #, depth= depth)

        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names
    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)                          
        new_file_names = self._file_names * (length // files_len)   

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    @staticmethod
    def _gt_transform(gt):
        return gt - 1 

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors
