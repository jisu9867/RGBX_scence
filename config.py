import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 12345

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'NYUDepthv2'
# C.dataset_path = "/media/jslee/Data2/jslee_two/jisu/data/kitti"
# C.rgb_root_folder = "/media/jslee/Data2/jslee_two/jisu/data/kitti/"
# C.rgb_root_folder_eval = "/media/jslee/Data2/jslee_two/jisu/data/kitti/"
C.dataset_path = "/media/jslee/Data2/jslee_two/jisu/data/kitti/kitti2/"
C.rgb_root_folder = "/media/jslee/Data2/jslee_two/jisu/data/kitti/kitti2/input/"
C.rgb_root_folder_eval = "/media/jslee/Data2/jslee_two/jisu/data/kitti/"
C.rgb_root_folder_n = "/media/jslee/Data2/jslee_two/jisu/VPD/depth/dataset/nyu_depth_v2/sync/"
C.rgb_root_folder_eval_n = "/media/jslee/Data2/jslee_two/workspace/dataset/nyu_depth_v2/official_splits/test/"
C.rgb_format = '.jpg'

# C.gt_root_folder = "/media/jslee/Data2/jslee_two/jisu/MIM-Depth-Estimation/label2/train"
C.gt_root_folder = "/media/jslee/Data2/jslee_two/jisu/MIM-Depth-Estimation/label_scene2/train"
# C.gt_root_folder_eval = "/media/jslee/Data2/jslee_two/jisu/MIM-Depth-Estimation/label2/val/"
C.gt_root_folder_eval = "/media/jslee/Data2/jslee_two/jisu/MIM-Depth-Estimation/label_scene2/val/"
C.gt_format = '.png'

C.gt_transform = False
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
C.x_root_folder = "/media/jslee/Data2/jslee_two/jisu/VPD/depth/dataset/nyu_depth_v2/hha/hha_train/"
C.x_root_folder2 = "/media/jslee/Data2/jslee_two/jisu/VPD/depth/dataset/nyu_depth_v2/hha/hha_train_v_d/"

C.x_root_folder_eval = "/media/jslee/Data2/jslee_two/jisu/VPD/depth/dataset/nyu_depth_v2/hha/hha_val/"
C.x_root_folder_eval2 = "/media/jslee/Data2/jslee_two/jisu/VPD/depth/dataset/nyu_depth_v2/hha/hha_val_v_d/"

C.x_format = '.png'
C.x_is_single_channel = False # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.train_source = "/media/jslee/Data2/jslee_two/jisu/MIM-Depth-Estimation/dataset/filenames/eigen_benchmark/train_list_nk_2400.txt"
C.eval_source = "/media/jslee/Data2/jslee_two/jisu/MIM-Depth-Estimation/dataset/filenames/eigen_benchmark/test_list_nk.txt"
C.is_test = False
C.num_train_imgs = 4790
C.num_eval_imgs = 1306
C.num_classes = 2
C.class_names = ['mim', 'ieb']

"""Image Config"""
C.background = 2
C.image_height = 352
C.image_width = 1216
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'mit_b4' # Remember change the path below.
C.pretrained_model = "/media/jslee/Data2/jslee_two/jisu/RGBX_Semantic_Segmentation/pretrained_models/mit_b4.pth"
# C.pretrained_model = "/media/jslee/Data2/jslee_two/jisu/RGBX_Semantic_Segmentation/pretrained_models/NYUDV2_CMX+Segformer-B2.pth"
# C.pretrained_model = None
C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
C.lr_power = 0.9
C.lr = 8e-4
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 16
C.nepochs = 5
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 4
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 1

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] # [0.75, 1, 1.25] # 
C.eval_flip = False # True # 
C.eval_crop_size = [480, 640] # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 5
C.checkpoint_step = 5

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath('log_' + C.dataset_name + '_' + C.backbone)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()