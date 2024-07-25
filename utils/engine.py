############################################################
#   File: engine.py                                        #
#   Created: 2019-11-20 15:02:13                           #
#   Author : wvinzh                                        #
#   Email : wvinzh@qq.com                                  #
#   ------------------------------------------             #
#   Description:engine.py                                  #
#   Copyright@2019 wvinzh, HUST                            #
############################################################
import math
import time
from utils import calculate_pooling_center_loss, mask2bbox
from utils import attention_crop, attention_drop, attention_crop_drop, attention_crop_drop2
from utils import getDatasetConfig, getConfig, getLogger
from utils import accuracy, get_lr, save_checkpoint, AverageMeter, set_seed
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import _pickle as pickle


def std_img(A):
    return (A-A.min()) / (A.max()-A.min())


class Engine():
    def __init__(self, dataset_name, prefix):
        dataset_name = dataset_name[0].upper() + dataset_name[1:]
        self.classes = np.array(pickle.load(open(prefix + 'data/%s_classes.pkl' % dataset_name, 'rb')))

    def train(self, state, epoch):
        config = state['config']
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        top1_0 = AverageMeter()
        top5_0 = AverageMeter()
        if config.feature_extract:
            top1_k = AverageMeter()
            top5_k = AverageMeter()
            top1_001 = AverageMeter()
            top5_001 = AverageMeter()
            top1_01 = AverageMeter()
            top5_01 = AverageMeter()
            top1_05 = AverageMeter()
            top5_05 = AverageMeter()

        print_freq = config.print_freq
        model = state['model']
        criterion = state['criterion']
        optimizer = state['optimizer']
        train_loader = state['train_loader']
        model.train()
        # if config.feature_extract:
        #     """冻结模块中的所有 BN 层的 running_mean 和 running_var"""
        #     for m in model.modules():
        #         if isinstance(m, torch.nn.BatchNorm2d):
        #             m.eval()  # 设置为评估模式，防止更新 running_mean 和 running_var

        end = time.time()
        for i, (img, label) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = label.cuda()
            input = img.cuda()
            # compute output
            if config.feature_extract:
                attention_maps1, raw_features1, logits1, bap_logits1, att_w1, topk_cls1, ps1 = model(input, label)
            else:
                attention_maps1, raw_features1, logits1 = model(input)
            features = raw_features1.reshape(raw_features1.shape[0], -1)

            feature_center_loss, center_diff = calculate_pooling_center_loss(
                features, state['center'], target, alfa=config.alpha)

            # update model.centers
            state['center'][target] += center_diff

            # compute refined loss
            # img_drop = attention_drop(attention_maps,input)
            # img_crop = attention_crop(attention_maps, input)
            img_crop, img_drop = attention_crop_drop2(attention_maps1, input)
            if config.feature_extract:
                attention_maps2, raw_features2, logits2, bap_logits2, att_w2, topk_cls2, ps2 = model(img_drop, label)
                attention_maps3, raw_features3, logits3, bap_logits3, att_w3, topk_cls3, ps3 = model(img_crop, label)
            else:
                attention_maps2, raw_features2, logits2 = model(img_drop)
                attention_maps3, raw_features3, logits3 = model(img_crop)

            loss1 = criterion(logits1, target)
            loss2 = criterion(logits2, target)
            loss3 = criterion(logits3, target)
            if config.feature_extract:
                topk_loss1 = criterion(bap_logits1, target)
                topk_loss2 = criterion(bap_logits2, target)
                topk_loss3 = criterion(bap_logits3, target)

            loss = (loss1 + loss2 + loss3) / 3 + feature_center_loss
            if config.feature_extract:
                loss += (topk_loss1 + topk_loss2 + topk_loss3) / 3
                # measure accuracy and record loss
                ens_logits_001 = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1) + F.softmax(logits3, dim=-1)) * 0.33 + (0.01/3) * (F.softmax(bap_logits1, dim=-1) + F.softmax(bap_logits2, dim=-1) + F.softmax(bap_logits3, dim=-1))
                ens_logits_01 = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1) + F.softmax(logits3, dim=-1)) * 0.3 + (0.1/3) * (F.softmax(bap_logits1, dim=-1) + F.softmax(bap_logits2, dim=-1) + F.softmax(bap_logits3, dim=-1))
                ens_logits_05 = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1) + F.softmax(logits3, dim=-1) + F.softmax(bap_logits1, dim=-1) + F.softmax(bap_logits2, dim=-1) + F.softmax(bap_logits3, dim=-1)) / 6
                prec1_k, prec5_k = accuracy(bap_logits1, target, topk=(1, 5))
                prec1_001, prec5_001 = accuracy(ens_logits_001, target, topk=(1, 5))
                prec1_01, prec5_01 = accuracy(ens_logits_01, target, topk=(1, 5))
                prec1_05, prec5_05 = accuracy(ens_logits_05, target, topk=(1, 5))

            logits = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1) + F.softmax(logits3, dim=-1)) / 3
            prec1, prec5 = accuracy(logits1, target, topk=(1, 5))
            prec1_0, prec5_0 = accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            top1_0.update(prec1_0[0], input.size(0))
            top5_0.update(prec5_0[0], input.size(0))
            if config.feature_extract:
                top1_k.update(prec1_k[0], input.size(0))
                top5_k.update(prec5_k[0], input.size(0))
                top1_001.update(prec1_001[0], input.size(0))
                top5_001.update(prec5_001[0], input.size(0))
                top1_01.update(prec1_01[0], input.size(0))
                top5_01.update(prec5_01[0], input.size(0))
                top1_05.update(prec1_05[0], input.size(0))
                top5_05.update(prec5_05[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                if config.feature_extract:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                          'Prec@1     {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5     {top5.val:.3f} ({top5.avg:.3f})\n'
                          'Prec@1   k {top1_k.val:.3f} ({top1_k.avg:.3f})\t'
                          'Prec@5   k {top5_k.val:.3f} ({top5_k.avg:.3f})\n'
                          'Prec@1   0 {top1_0.val:.3f} ({top1_0.avg:.3f})\t'
                          'Prec@5   0 {top5_0.val:.3f} ({top5_0.avg:.3f})\n'
                          'Prec@1 001 {top1_001.val:.3f} ({top1_001.avg:.3f})\t'
                          'Prec@5 001 {top5_001.val:.3f} ({top5_001.avg:.3f})\n'
                          'Prec@1  01 {top1_01.val:.3f} ({top1_01.avg:.3f})\t'
                          'Prec@5  01 {top5_01.val:.3f} ({top5_01.avg:.3f})\n'
                          'Prec@1  05 {top1_05.val:.3f} ({top1_05.avg:.3f})\t'
                          'Prec@5  05 {top5_05.val:.3f} ({top5_05.avg:.3f})\n'  # *
                          .format(
                        epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                            top1=top1, top5=top5, top1_k=top1_k, top5_k=top5_k, top1_0=top1_0, top5_0=top5_0,
                            top1_001=top1_001, top5_001=top5_001, top1_01=top1_01, top5_01=top5_01,
                            top1_05=top1_05, top5_05=top5_05
                            ))
                else:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                          'Prec@1   0 {top1_0.val:.3f} ({top1_0.avg:.3f})\t'
                          'Prec@5   0 {top5_0.val:.3f} ({top5_0.avg:.3f})'.format(
                            epoch, i, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, top1=top1, top5=top5, top1_0=top1_0, top5_0=top5_0))
                print("loss1,loss2,loss3,feature_center_loss", loss1.item(), loss2.item(), loss3.item(),
                      feature_center_loss.item())
        if config.feature_extract:
            return (top1.avg, top1_k.avg, top1_0.avg, top1_001.avg, top1_01.avg, top1_05.avg), losses.avg
        else:
            return top1.avg, losses.avg

    def validate(self, state):
        config = state['config']
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        if config.feature_extract:
            top1_k = AverageMeter()
            top5_k = AverageMeter()
            top1_0 = AverageMeter()
            top5_0 = AverageMeter()
            top1_001 = AverageMeter()
            top5_001 = AverageMeter()
            top1_01 = AverageMeter()
            top5_01 = AverageMeter()
            top1_05 = AverageMeter()
            top5_05 = AverageMeter()

        print_freq = config.print_freq
        model = state['model']
        val_loader = state['val_loader']
        criterion = state['criterion']
        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda()
                input = input.cuda()
                # forward
                if config.feature_extract:
                    attention_maps1, raw_features1, logits1, bap_logits1, att_w1, topk_cls1, ps1 = model(input, target)
                else:
                    attention_maps1, raw_features1, logits1 = model(input)
                features = raw_features1.reshape(raw_features1.shape[0], -1)
                feature_center_loss, _ = calculate_pooling_center_loss(
                    features, state['center'], target, alfa=config.alpha)

                img_crop, img_drop = attention_crop_drop2(attention_maps1, input)
                # img_drop = attention_drop(attention_maps,input)
                # img_crop = attention_crop(attention_maps,input)
                if config.feature_extract:
                    attention_maps2, raw_features2, logits2, bap_logits2, att_w2, topk_cls2, ps2 = model(img_drop, target)
                    attention_maps3, raw_features3, logits3, bap_logits3, att_w3, topk_cls3, ps3 = model(img_crop, target)
                else:
                    attention_maps2, raw_features2, logits2 = model(img_drop)
                    attention_maps3, raw_features3, logits3 = model(img_crop)
                loss1 = criterion(logits1, target)
                loss2 = criterion(logits2, target)
                loss3 = criterion(logits3, target)
                if config.feature_extract:
                    topk_loss1 = criterion(bap_logits1, target)
                    topk_loss2 = criterion(bap_logits2, target)
                    topk_loss3 = criterion(bap_logits3, target)
                # loss = loss1 + feature_center_loss
                loss = (loss1 + loss2 + loss3) / 3 + feature_center_loss
                if config.feature_extract:
                    loss += (topk_loss1 + topk_loss2 + topk_loss3) / 3

                    # measure accuracy and record loss
                    logits = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1) + F.softmax(logits3, dim=-1)) / 3
                    ens_logits_001 = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1) + F.softmax(logits3, dim=-1)) * 0.33 + (0.01 / 3) * (F.softmax(bap_logits1, dim=-1) + F.softmax(bap_logits2, dim=-1) + F.softmax(bap_logits3, dim=-1))
                    ens_logits_01 = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1) + F.softmax(logits3, dim=-1)) * 0.3 + (0.1/3) * (F.softmax(bap_logits1, dim=-1) + F.softmax(bap_logits2, dim=-1) + F.softmax(bap_logits3, dim=-1))
                    ens_logits_05 = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1) + F.softmax(logits3, dim=-1) + F.softmax(bap_logits1, dim=-1) + F.softmax(bap_logits2, dim=-1) + F.softmax(bap_logits3, dim=-1)) / 6

                prec1, prec5 = accuracy(logits1, target, topk=(1, 5))
                if config.feature_extract:
                    prec1_k, prec5_k = accuracy(bap_logits1, target, topk=(1, 5))
                    prec1_0, prec5_0 = accuracy(logits, target, topk=(1, 5))
                    prec1_001, prec5_001 = accuracy(ens_logits_001, target, topk=(1, 5))
                    prec1_01, prec5_01 = accuracy(ens_logits_01, target, topk=(1, 5))
                    prec1_05, prec5_05 = accuracy(ens_logits_05, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))
                if config.feature_extract:
                    top1_k.update(prec1_k[0], input.size(0))
                    top5_k.update(prec5_k[0], input.size(0))
                    top1_0.update(prec1_0[0], input.size(0))
                    top5_0.update(prec5_0[0], input.size(0))
                    top1_001.update(prec1_001[0], input.size(0))
                    top5_001.update(prec5_001[0], input.size(0))
                    top1_01.update(prec1_01[0], input.size(0))
                    top5_01.update(prec5_01[0], input.size(0))
                    top1_05.update(prec1_05[0], input.size(0))
                    top5_05.update(prec5_05[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    if config.feature_extract:
                        print('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                              'Prec@1     {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5     {top5.val:.3f} ({top5.avg:.3f})\n'
                              'Prec@1   k {top1_k.val:.3f} ({top1_k.avg:.3f})\t'
                              'Prec@5   k {top5_k.val:.3f} ({top5_k.avg:.3f})\n'
                              'Prec@1   0 {top1_0.val:.3f} ({top1_0.avg:.3f})\t'
                              'Prec@5   0 {top5_0.val:.3f} ({top5_0.avg:.3f})\n'
                              'Prec@1 001 {top1_001.val:.3f} ({top1_001.avg:.3f})\t'
                              'Prec@5 001 {top5_001.val:.3f} ({top5_001.avg:.3f})\n'
                              'Prec@1  01 {top1_01.val:.3f} ({top1_01.avg:.3f})\t'
                              'Prec@5  01 {top5_01.val:.3f} ({top5_01.avg:.3f})\n'
                              'Prec@1  05 {top1_05.val:.3f} ({top1_05.avg:.3f})\t'
                              'Prec@5  05 {top5_05.val:.3f} ({top5_05.avg:.3f})\n'
                            .format(
                            i, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5, top1_k=top1_k, top5_k=top5_k, top1_0=top1_0, top5_0=top5_0,
                            top1_001=top1_001, top5_001=top5_001, top1_01=top1_01, top5_01=top5_01,
                            top1_05=top1_05, top5_05=top5_05))
                    else:
                        print('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5))
            if config.feature_extract:
                print(' * Prec@1     {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
                      ' * Prec@1   k {top1_k.avg:.3f} Prec@5 001 {top5_k.avg:.3f}\n'
                      ' * Prec@1   0 {top1_0.avg:.3f} Prec@5 001 {top5_0.avg:.3f}\n'
                      ' * Prec@1 001 {top1_001.avg:.3f} Prec@5 001 {top5_001.avg:.3f}\n'
                      ' * Prec@1  01 {top1_01.avg:.3f} Prec@5 01 {top5_01.avg:.3f}\n'
                      ' * Prec@1  05 {top1_05.avg:.3f} Prec@5 05 {top5_05.avg:.3f}'
                      .format(top1=top1, top5=top5, top1_k=top1_k, top5_k=top5_k, top1_0=top1_0, top5_0=top5_0,
                              top1_001=top1_001, top5_001=top5_001, top1_01=top1_01, top5_01=top5_01,
                              top1_05=top1_05, top5_05=top5_05))
                return (top1.avg, top1_k.avg, top1_0.avg, top1_001.avg, top1_01.avg, top1_05.avg), losses.avg
            else:
                print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                      .format(top1=top1, top5=top5))
                return top1.avg, losses.avg

    def sac_drop(self, att_w):
        d_phi = 0.5
        batch_size, height, width, num_parts = att_w.shape
        masks = []
        att_w_max = torch.max_pool2d(att_w.permute(0, 3, 1, 2), kernel_size=26).permute(0, 2, 3, 1)
        for i in range(batch_size):
            current_att_w = att_w[i]
            # 299 * 299 * 5
            M_D = current_att_w <= (1 - d_phi) * att_w_max[i]
            # 299 * 299 * 1
            mask = torch.any(M_D, dim=2, keepdim=True)
            masks.append(mask)
        # 12 * 299 * 299 * 1
        masks = torch.stack(masks).type(torch.float32)
        return masks

    def denormalize(self, tensor, mean, std):
        mean = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
        std = torch.tensor(std, device=tensor.device).view(3, 1, 1)
        tensor = tensor * std + mean
        return tensor

    def generate_attw_image(self, image, attention_map):
        h, w, _ = image.shape
        attention_images = []
        image = (image / 2.0 + 0.5) * 255.0
        image = image.type(torch.uint8)
        for i in range(attention_map.shape[-1]):
            mask = attention_map[:, :, i]
            mask = (mask / mask.max() * 255.0).type(torch.uint8)
            mask = mask.cpu().numpy()
            mask = cv2.resize(mask, (w, h))
            color_map = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
            attention_image = cv2.addWeighted(image.cpu().numpy(), 0.5, color_map.astype(np.uint8), 0.5, 0)
            attention_image = cv2.cvtColor(attention_image, cv2.COLOR_BGR2RGB)
            attention_images.append(np.transpose(attention_image, (2, 0, 1)))
        return torch.tensor(attention_images, device=image.device)

    def test(self, val_loader, model, config, sw, e):
        if config.action == 'train':
            freq_times = 20
        else:
            freq_times = 1
        top1 = AverageMeter()
        top5 = AverageMeter()
        top1_0 = AverageMeter()
        top5_0 = AverageMeter()
        if config.feature_extract:
            top1_k = AverageMeter()
            top5_k = AverageMeter()
            top1_001 = AverageMeter()
            top5_001 = AverageMeter()
            top1_01 = AverageMeter()
            top5_01 = AverageMeter()
            top1_05 = AverageMeter()
            top5_05 = AverageMeter()
        print_freq = 10
        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda()
                input = input.cuda()
                input_d = self.denormalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # forward
                if config.feature_extract:
                    attention_maps1, raw_features1, logits1, bap_logits1, att_w1, topk_cls1, ps = model(input, target)
                else:
                    attention_maps1, raw_features1, logits1 = model(input)
                refined_input = mask2bbox(attention_maps1, input)
                if config.feature_extract:
                    attention_maps2, raw_features2, logits2, bap_logits2, att_w2, topk_cls2, _ = model(refined_input, target)
                else:
                    attention_maps2, raw_features2, logits2 = model(refined_input)
                logits = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1)) / 2
                if config.feature_extract:
                    ens_logits_001 = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1)) * 0.495 + 0.005 * (F.softmax(bap_logits1, dim=-1) + F.softmax(bap_logits2, dim=-1))
                    ens_logits_01 = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1)) * 0.45 + 0.05 * (F.softmax(bap_logits1, dim=-1) + F.softmax(bap_logits2, dim=-1))
                    ens_logits_05 = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1) + F.softmax(bap_logits1, dim=-1) + F.softmax(bap_logits2, dim=-1)) / 4
                # measure accuracy and record loss
                prec1, prec5 = accuracy(logits1, target, topk=(1, 5))
                prec1_0, prec5_0 = accuracy(logits, target, topk=(1, 5))
                if config.feature_extract:
                    prec1_k, prec5_k = accuracy(bap_logits1, target, topk=(1, 5))
                    prec1_001, prec5_001 = accuracy(ens_logits_001, target, topk=(1, 5))
                    prec1_01, prec5_01 = accuracy(ens_logits_01, target, topk=(1, 5))
                    prec1_05, prec5_05 = accuracy(ens_logits_05, target, topk=(1, 5))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))
                top1_0.update(prec1_0[0], input.size(0))
                top5_0.update(prec5_0[0], input.size(0))
                if config.feature_extract:
                    top1_k.update(prec1_k[0], input.size(0))
                    top5_k.update(prec5_k[0], input.size(0))
                    top1_001.update(prec1_001[0], input.size(0))
                    top5_001.update(prec5_001[0], input.size(0))
                    top1_01.update(prec1_01[0], input.size(0))
                    top5_01.update(prec5_01[0], input.size(0))
                    top1_05.update(prec1_05[0], input.size(0))
                    top5_05.update(prec5_05[0], input.size(0))

                if i % print_freq == 0:
                    if config.feature_extract:
                        print('Test: [{0}/{1}]\n'
                              'Prec@1     {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5     {top5.val:.3f} ({top5.avg:.3f})\n'
                              'Prec@1   k {top1_k.val:.3f} ({top1_k.avg:.3f})\t'
                              'Prec@5   k {top5_k.val:.3f} ({top5_k.avg:.3f})\n'
                              'Prec@1   0 {top1_0.val:.3f} ({top1_0.avg:.3f})\t'
                              'Prec@5   0 {top5_0.val:.3f} ({top5_0.avg:.3f})\n'
                              'Prec@1 001 {top1_001.val:.3f} ({top1_001.avg:.3f})\t'
                              'Prec@5 001 {top5_001.val:.3f} ({top5_001.avg:.3f})\n'
                              'Prec@1  01 {top1_01.val:.3f} ({top1_01.avg:.3f})\t'
                              'Prec@5  01 {top5_01.val:.3f} ({top5_01.avg:.3f})\n'
                              'Prec@1  05 {top1_05.val:.3f} ({top1_05.avg:.3f})\t'
                              'Prec@5  05 {top5_05.val:.3f} ({top5_05.avg:.3f})'
                        .format(
                            i, len(val_loader),
                            top1=top1, top5=top5, top1_k=top1_k, top5_k=top5_k, top1_0=top1_0, top5_0=top5_0, top1_001=top1_001, top5_001=top5_001, top1_01=top1_01, top5_01=top5_01, top1_05=top1_05, top5_05=top5_05))
                    else:
                        print('Test: [{0}/{1}]\t'
                              'Prec@1 {top1_0.val:.3f} ({top1_0.avg:.3f})\t'
                              'Prec@5 {top5_0.val:.3f} ({top5_0.avg:.3f})'.format(
                                i, len(val_loader),
                                top1_0=top1_0, top5_0=top5_0))
            print(' * Prec@1   0 {top1_0.avg:.3f} Prec@5   0 {top5_0.avg:.3f}'
                  .format(top1_0=top1_0, top5_0=top5_0))
            print(' * Prec@1     {top1.avg:.3f}   Prec@5     {top5.avg:.3f}\n'
                  ' * Prec@1   0 {top1_0.avg:.3f}   Prec@5   0 {top5_0.avg:.3f}\n'
                  ' * Prec@1   k {top1_k.avg:.3f}   Prec@5   k {top5_k.avg:.3f}\n'
                  ' * Prec@1 001 {top1_001.avg:.3f}   Prec@5 001 {top5_001.avg:.3f}\n'
                  ' * Prec@1  01 {top1_01.avg:.3f}  Prec@5  01 {top5_01.avg:.3f}\n'
                  ' * Prec@1  05 {top1_05.avg:.3f}  Prec@5  05 {top5_05.avg:.3f}\n'
            .format(
                top1=top1, top5=top5, top1_k=top1_k, top5_k=top5_k, top1_0=top1_0, top5_0=top5_0, top1_001=top1_001,
                top5_001=top5_001, top1_01=top1_01, top5_01=top5_01, top1_05=top1_05, top5_05=top5_05))
        if config.feature_extract:
            return top1.avg, top1_k.avg, top1_0.avg, top1_001.avg, top1_01.avg, top1_05.avg, top5.avg
        else:
            return top1_0.avg, top5_0.avg


if __name__ == '__main__':
    engine = Engine()
    engine.train()
