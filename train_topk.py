############################################################
#   File: train_bap.py                                     #
#   Created: 2019-11-06 13:22:23                           #
#   Author : wvinzh                                        #
#   Email : wvinzh@qq.com                                  #
#   ------------------------------------------             #
#   Description:train_bap.py                               #
#   Copyright@2019 wvinzh, HUST                            #
############################################################

# system
import os
import time
import shutil
import random
import numpy as np

# my implementation
from model.inception_topk import inception_v3_topk
from model.resnet import resnet50
from dataset.custom_dataset import CustomDataset

from utils import calculate_pooling_center_loss, mask2bbox
from utils import attention_crop, attention_drop, attention_crop_drop
from utils import getDatasetConfig, getConfig
from utils import accuracy, get_lr, save_checkpoint, AverageMeter, set_seed
from utils import Engine

# pytorch
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import copy
import cProfile

GLOBAL_SEED = 1231
def _init_fn(worker_id):
    set_seed(GLOBAL_SEED+worker_id)


def set_parameter_requires_grad(model, feature_extracting):
    parameters_count, trainable_count, non_trainable_count = 0, 0, 0
    if feature_extracting:
        for name, parameter in model.named_parameters():
            parameters_count += 1
            # parameter.requires_grad = True
            # if 'sac' not in name and 'fc_new' not in name:
            #     parameter.requires_grad = False
            #     print('non-trainable parameter: ', name)
            #     non_trainable_count += 1
            # else:
            parameter.requires_grad = True
            print('trainable parameter: ', name)
            trainable_count += 1
        print('parameters count: %s, trainable count: %s, non-trainable count: %s' % (parameters_count, trainable_count, non_trainable_count))

def inception_v3_topk_init(net, feature_extract):
    set_parameter_requires_grad(net, feature_extract)
    return net


def train():
    # input params
    set_seed(GLOBAL_SEED)
    config = getConfig()
    data_config = getDatasetConfig(config.dataset)
    sw_log = 'logs/%s' % config.dataset
    sw = SummaryWriter(log_dir=sw_log)
    best_prec1 = 0.
    best_prec_test = 0.
    rate = 0.875

    # define train_dataset and loader
    transform_train = transforms.Compose([
        transforms.Resize((int(config.input_size//rate), int(config.input_size//rate))),
        transforms.RandomCrop((config.input_size,config.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=32./255.,saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset = CustomDataset(
        data_config['train'], data_config['train_root'], transform=transform_train)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=True, worker_init_fn=_init_fn)

    transform_test = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.CenterCrop(config.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_dataset = CustomDataset(
        data_config['val'], data_config['val_root'], transform=transform_test)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers, pin_memory=True, worker_init_fn=_init_fn)
    # logging dataset info
    print('Dataset Name:{dataset_name}, Train:[{train_num}], Val:[{val_num}]'.format(
        dataset_name=config.dataset,
        train_num=len(train_dataset),
        val_num=len(val_dataset)))
    print('Batch Size:[{0}], Total:::Train Batches:[{1}],Val Batches:[{2}]'.format(
        config.batch_size, len(train_loader), len(val_loader)
    ))
    use_gpu = torch.cuda.is_available() and config.use_gpu
    # define model
    if config.model_name == 'inception':
        net = inception_v3_topk(pretrained=True, aux_logits=False, num_parts=config.parts, dataset_name=config.dataset, batch_size=config.batch_size, use_gpu=use_gpu, prefix=config.prefix)
    elif config.model_name == 'resnet50':
        net = resnet50(pretrained=True,use_bap=True)

    
    in_features = net.fc_new.in_features
    new_linear = torch.nn.Linear(
        in_features=in_features, out_features=train_dataset.num_classes)
    net.fc_new = new_linear
    # feature center
    feature_len = 768 if config.model_name == 'inception' else 512
    center_dict = {'center': torch.zeros(
        train_dataset.num_classes, feature_len*config.parts)}

    # gpu config
    if use_gpu:
        net = net.cuda()
        center_dict['center'] = center_dict['center'].cuda()
    gpu_ids = [int(r) for r in config.gpu_ids.split(',')]
    if use_gpu and config.multi_gpu:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)

    # define loss
    criterion = torch.nn.CrossEntropyLoss()
    if use_gpu:
        criterion = criterion.cuda()
        # torch.cuda.set_per_process_memory_fraction(0.99, 0)
        # torch.cuda.empty_cache()
        # # 计算一下总内存有多少。
        # total_memory = torch.cuda.get_device_properties(0).total_memory
        # # 使用0.499的显存:
        # tmp_tensor = torch.empty(int(total_memory * 0.95), dtype=torch.int8, device='cuda')
        # # 清空该显存：
        # del tmp_tensor
        # torch.cuda.empty_cache()

    # Load checkpoint if exists
    if config.resume_bap:
        if os.path.isfile(config.resume_bap):
            print("=> loading checkpoint bap '{}'".format(config.resume_bap))
            checkpoint = torch.load(config.resume_bap)
            # start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            if 'best_prec_test' in checkpoint:
                best_prec_test = checkpoint['best_prec_test']
            pretrained_dict = checkpoint['state_dict']
            net_dict = net.state_dict()
            state_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict.keys()}
            net_dict.update(state_dict)
            net.load_state_dict(net_dict)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            center_dict['center'] = checkpoint['center']
            print("=> loaded checkpoint bap '{}' best_prec:{} (epoch {})"
                  .format(config.resume_bap, best_prec1, checkpoint['epoch']))
        else:
            print("=> no checkpoint bap found at '{}'".format(config.resume_bap))

    net = inception_v3_topk_init(net, config.feature_extract)
    print('\n==============================================================================================\n')
    # 是否训练所有层
    print("Params to learn:")
    if config.feature_extract:
        params_to_update = []
        for name, param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        params_to_update = net.parameters()
        for name, param in net.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # define optimizer
    assert config.optim in ['sgd', 'adam'], 'optim name not found!'
    if config.optim == 'sgd':
        optimizer = torch.optim.SGD(
            params_to_update, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optim == 'adam':
        optimizer = torch.optim.Adam(
            params_to_update, lr=config.lr, weight_decay=config.weight_decay)

    # define learning scheduler
    assert config.scheduler in ['plateau',
                                'step', 'none'], 'scheduler not supported!!!'
    if config.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.1)
    elif config.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.9)

    start_epoch = 0
    # Load checkpoint if exists
    if config.resume:
        if os.path.isfile(config.resume):
            print("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            best_prec_test = checkpoint['best_prec_test']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            center_dict['center'] = checkpoint['center']
            print("=> loaded checkpoint '{}' best_prec:{} best_prec_test: {} (epoch {})"
                  .format(config.resume, best_prec1, best_prec_test, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))

    # train val parameters dict
    state = {'model': net, 'train_loader': train_loader,
             'val_loader': val_loader, 'criterion': criterion,
             'center': center_dict['center'], 'config': config,
             'optimizer': optimizer}

    # 重置学习率
    new_lr = config.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    print(f'Resumed training from epoch {start_epoch} with learning rate reset to {new_lr}')

    ## train and val
    engine = Engine(config.dataset, config.prefix)
    print(config)
    # (first_best_prec1, first_best_prec1_k, first_best_prec1_0, first_best_prec1_001, first_best_prec1_01, first_best_prec1_05), val_loss = engine.validate(state)
    # prev_state_dict = copy.deepcopy(state['model'].state_dict())
    for e in range(start_epoch, config.epochs):
        if config.scheduler == 'step':
            scheduler.step()
        lr_val = get_lr(optimizer)
        print("Start epoch %d ==========,lr=%f" % (e, lr_val))
        (train_prec1, train_prec1_k, train_prec1_0, train_prec1_001, train_prec1_01, train_prec1_05), train_loss = engine.train(state, e)
        (prec1, prec1_k, prec1_0, prec1_001, prec1_01, prec1_05), val_loss, cos_sim_loss = engine.validate(state)
        # test_prec1, test_prec1_k, test_prec1_0, test_prec1_001, test_prec1_01, test_prec1_05, prec5 = engine.test(val_loader, net, config, sw, e)
        cur_best = sorted([prec1, prec1_k, prec1_0, prec1_001, prec1_01, prec1_05], key=lambda x: -x)[0]
        is_best = cur_best > best_prec1
        # cur_best_test = sorted([test_prec1, test_prec1_k, test_prec1_0, test_prec1_001, test_prec1_01, test_prec1_05],
        #        key=lambda x: -x)[0]
        # is_best_test = cur_best_test > best_prec_test
        # best_prec_test = max(cur_best_test, best_prec_test)
        is_best_test = False
        best_prec1 = max(cur_best, best_prec1)
        save_checkpoint({
            'epoch': e + 1,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
            'best_prec_test': best_prec_test,
            'optimizer': optimizer.state_dict(),
            'center': center_dict['center']
        }, is_best, is_best_test, config.checkpoint_path)
        sw.add_scalars("Accurancy", {'train': train_prec1, 'val': prec1}, e)
        sw.add_scalars("Accurancy_k", {'train': train_prec1_k, 'val': prec1_k}, e)
        sw.add_scalars("Accurancy_0", {'train': train_prec1_0, 'val': prec1_0}, e)
        sw.add_scalars("Accurancy_001", {'train': train_prec1_001, 'val': prec1_001}, e)
        sw.add_scalars("Accurancy_01", {'train': train_prec1_01, 'val': prec1_01}, e)
        sw.add_scalars("Accurancy_05", {'train': train_prec1_05, 'val': prec1_05}, e)
        sw.add_scalars("Loss", {'train': train_loss, 'val': val_loss, 'cos_sim_loss': cos_sim_loss}, e)
        # sw.add_scalars("best_prec_test", {'test': best_prec_test}, e)
        if config.scheduler == 'plateau':
            scheduler.step(val_loss)

def test():
    ##
    config = getConfig()
    data_config = getDatasetConfig(config.dataset)
    engine = Engine(config.dataset, config.prefix)
    # define dataset
    transform_test = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.CenterCrop(config.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_dataset = CustomDataset(
        data_config['val'], data_config['val_root'], transform=transform_test)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers, pin_memory=True)
    # define model
    if config.model_name == 'inception':
        net = inception_v3_topk(pretrained=True, aux_logits=False,num_parts=config.parts, dataset_name=config.dataset, batch_size=config.batch_size, use_gpu=config.use_gpu, prefix=config.prefix)
    elif config.model_name == 'resnet50':
        net = resnet50(pretrained=True)

    in_features = net.fc_new.in_features
    new_linear = torch.nn.Linear(
        in_features=in_features, out_features=val_dataset.num_classes)
    net.fc_new = new_linear

    # load checkpoint
    use_gpu = torch.cuda.is_available() and config.use_gpu
    if use_gpu:
        net = net.cuda()
        # torch.cuda.set_per_process_memory_fraction(0.5, 0)
        # torch.cuda.empty_cache()
    gpu_ids = [int(r) for r in config.gpu_ids.split(',')]
    if use_gpu and len(gpu_ids) >= 1:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)
    #checkpoint_path = os.path.join(config.checkpoint_path,'model_best.pth.tar')
    # net.load_state_dict(torch.load(config.checkpoint_path)['state_dict'])
    checkpoint = torch.load(config.checkpoint_path)
    # start_epoch = checkpoint['epoch']
    pretrained_dict = checkpoint['state_dict']
    net_dict = net.state_dict()
    state_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict.keys()}
    net_dict.update(state_dict)
    net.load_state_dict(net_dict)

    # define loss
    # define loss
    criterion = torch.nn.CrossEntropyLoss()
    if use_gpu:
        criterion = criterion.cuda()
    sw_log = 'logs/ckpt/%s' % config.dataset
    sw = SummaryWriter(log_dir=sw_log, flush_secs=30)
    prec1, prec1_k, prec1_0, prec1_001, prec1_01, prec1_05, prec5 = engine.test(val_loader, net, config, sw, 100)
    sw.close()
    print((prec1, prec1_k, prec1_0, prec1_001, prec1_01, prec1_05))


if __name__ == '__main__':
    config = getConfig()
    if config.action == 'train':
        cProfile.run('train()', 'output.prof')
    else:
        cProfile.run('test()', 'output.prof')
