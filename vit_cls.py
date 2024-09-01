import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader.dataloader import get_train_loader, get_val_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor, ensure_dir
from criterion import SiLogLoss, AverageMeter, MetricLogger

import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from tensorboardX import SummaryWriter
from earlystop import EarlyStopping
import numpy as np
from utils.metric import *
parser = argparse.ArgumentParser()
logger = get_logger()

from transformers import ViTImageProcessor, ViTForImageClassification
import torch.nn as nn
from PIL import Image
import requests



with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)
    val_loader, val_sampler = get_val_loader(engine, RGBXDataset)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier.out_features
    model.fc = nn.Linear(num_ftrs, 2)
    criterion = nn.CrossEntropyLoss()

    # group weight and config optimizer in RGBX config
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr

    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)

    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank],
                                            output_device=engine.local_rank, find_unused_parameters=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print("=========================no distributed=========================")

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    logger.info('begin trainning:')

    early_stopping = EarlyStopping(patience =7, verbose = True)

    for epoch in range(engine.state.epoch, config.nepochs + 1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        pbar2 = tqdm(range(config.num_eval_imgs//config.batch_size), file=sys.stdout, bar_format=bar_format)
        dataloader = iter(train_loader)
        val_dataloader = iter(val_loader)

        # from torchvision import transforms
        # from PIL import Image
        model.train()
        # depth_loss = AverageMeter()
        sum_loss = 0
        sum_loss_val = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = next(dataloader)

            imgs = minibatch['data'].cuda(non_blocking=True)
            depth = minibatch['depth'].cuda()
            gts = minibatch['label'].cuda()


            aux_rate = 0.2
            optimizer.zero_grad()

            # inputs = processor(images=imgs, return_tensors="pt") #0-255values input
            # print(type(inputs))
            # outputs = model(**inputs)
            outputs = model(imgs)
            logits = outputs.logits
            # model predicts one of the 1000 ImageNet classes
            predicted_class_idx = logits.argmax(-1).item()
            print(predicted_class_idx)
            # print("Predicted class:", model.config.id2label[predicted_class_idx])
            loss = criterion(predicted_class_idx, gts)
            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            loss.backward()
            optimizer.step()

            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)
            # scheduler.step()
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr
            # lr = optimizer.param_groups[0]['lr']
            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))

            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
            del loss
            pbar.set_description(print_str, refresh=False)
        with torch.no_grad():
            model.eval()
            for idx2 in pbar2:
                minibatch = next(val_dataloader)
                imgs = minibatch['data'] # 4 480 640 3
                depth = minibatch['depth'].cuda()
                gts = minibatch['label'].cuda()

                imgs = imgs.cuda(non_blocking=True)

                aux_rate = 0.2
                inputs = processor(images=imgs, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits
                # model predicts one of the 1000 ImageNet classes
                predicted_class_idx = logits.argmax(-1).item()
                # print("Predicted class:", model.config.id2label[predicted_class_idx])
                loss_val = criterion(predicted_class_idx, gts)

                # reduce the whole loss over multi-gpu
                if engine.distributed:
                    reduce_loss = all_reduce_tensor(loss_val, world_size=engine.world_size)

                # pred_crop, gt_crop = cropping_img(pred_d, depth)
                # computed_result = eval_depth(pred_crop, gt_crop)

                if engine.distributed:
                    sum_loss_val += reduce_loss.item()
                    print_str2 = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                                + ' Iter {}/{}:'.format(idx2 + 1, config.num_eval_imgs//config.batch_size) \
                                + ' lr=%.4e' % lr \
                                + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss_val / (idx2 + 1)))
                else:
                    sum_loss_val += loss_val
                    print_str2 = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                                + ' Iter {}/{}:'.format(idx2 + 1, config.num_eval_imgs//config.batch_size) \
                                + ' lr=%.4e' % lr \
                                + ' loss=%.4f total_loss=%.4f' % (loss_val, (sum_loss_val / (idx2 + 1)))
                del loss_val
                pbar2.set_description(print_str2, refresh=False)

        early_stopping(sum_loss_val / len(pbar2))
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
            tb.add_scalar('Val_loss', sum_loss_val / len(pbar2), epoch)

        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (
                epoch == config.nepochs) or (early_stopping.save):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)

        early_stopping.save = False
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break





