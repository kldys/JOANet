
import argparse
import datetime
import time
import logging
import os.path as osp
from logger import setup_logger
from model_seg import BiSeNet
from cityscapes import CityScapes
from loss import OhemCELoss
from optimizer import Optimizer

import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from model import SISR
from Dataset import Train_Data
from config import opt
from torch.utils.data import DataLoader
from skimage import io, color
import os
from skimage.metrics import structural_similarity as compare_ssim
import cv2 as cv

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x, y):
        tensor_gpu = torch.tensor(x).cuda()
        tensor_cpu = tensor_gpu.cpu()
        img1 = tensor_cpu.numpy()
        tensor_gpu = torch.tensor(y).cuda()
        tensor_cpu = tensor_gpu.cpu()
        img2 = tensor_cpu.numpy()
        edge_output1 = cv.Canny(img1.astype(np.uint8), 50, 150)
        edge_output1[edge_output1 == 255] = 1

        edge_output2 = cv.Canny(img2.astype(np.uint8), 50, 150)
        edge_output2[edge_output2 == 255] = 1
        a3 = abs(edge_output1.astype(np.float32) - edge_output2.astype(np.float32))
        edge_loss = torch.tensor(sum(a3)).cuda(0)
        mse_loss = torch.mean(torch.pow((x - y), 2))+0.6*edge_loss
        return mse_loss*10

def train_sisr(num):
    train_data = Train_Data()
    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True)

    net = SISR()
    net = net.cuda()
    net = nn.DataParallel(net)
    if num>0:
        net.load_state_dict(torch.load(opt.load_model_path))
    else:
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

        torch.save(net.state_dict(), opt.save_model_path)


    criterion =CustomLoss()
    criterion = criterion.cuda()

    optimizer = optim.SGD(net.parameters(), lr=opt.lr)

    num_show = 0
    psnr_best = 0

    if num > 0:
        modelpth = './model'
        Method = 'X4'
        modelpth = os.path.join(modelpth, Method)
        n_classes = 2
        segmodel = BiSeNet(n_classes=n_classes)
        save_pth = osp.join(modelpth, 'BTM_model_final.pth')
        segmodel.load_state_dict(torch.load(save_pth))
        segmodel.cuda()
        segmodel.eval()
        for p in segmodel.parameters():
            p.requires_grad = False
        print('Load Segmentation Model {} Sucessfully~'.format(save_pth))

    if num > 0:
        score_thres = 0.7
        ignore_idx = 255
        n_min = 8 * 256 * 256 // 8
        criteria_p = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        criteria_16 = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    for epoch in range(opt.max_epoch):
        for i, (data, HR, label) in enumerate(train_loader):
            data = data.cuda()
            HR = HR.cuda()
            label = label.cuda().long()
            lb=torch.squeeze(label, 1)
            torch.cuda.empty_cache()


            optimizer.zero_grad()
            output = net(data)
            if num > 0:
                sisr_out=output.repeat(1, 3, 1, 1)
                out, mid = segmodel(sisr_out)


                lossp = criteria_p(out, lb)
                loss2 = criteria_16(mid, lb)
                seg_loss = lossp + 0.1 * loss2

            loss = criterion(output, HR)
            if num > 0:
                loss_total = loss +  seg_loss*num
            else:
                loss_total = loss

            loss_total.backward()
            optimizer.step()

            if i % 20 == 0:
                mse_loss, psnr_now, ssim = val(net, epoch, i)
                print('[%d, %5d] loss:%.10f PSNR:%.3f SSIM:%.3f' % (epoch + 1, (i + 1)*opt.batch_size, mse_loss, psnr_now, ssim))
                num_show += 1

                if psnr_best < psnr_now:
                    psnr_best = psnr_now
                    torch.save(net.state_dict(), opt.save_model_path)

        if (epoch+1) % 3 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * opt.lr_decay
            print('learning rate: ', optimizer.param_groups[0]['lr'])
    print('Finished Training')

def val(net1, epoch, i):
    with torch.no_grad():
        psnr_ac = 0
        ssim_ac = 0
        for j in range(5):
            label = io.imread('./data/test_data/X%d/BTM/'%opt.scaling_factor + str(j + 1) + '_HR.png')
            test = io.imread('./data/test_data/X%d/BTM/'%opt.scaling_factor + str(j + 1) + '_LR.png')

            label_y = label / 255
            test_y = test / 255

            label = torch.FloatTensor(label_y).unsqueeze(0).unsqueeze(0).cuda()
            test = torch.FloatTensor(test_y).unsqueeze(0).unsqueeze(0).cuda()

            output = net1(test)
            output = torch.clamp(output, 0.0, 1.0)
            loss = (output*255 - label*255).pow(2).sum() / (output.shape[2]*output.shape[3])
            psnr = 10*np.log10(255*255 / loss.item())

            output = output.squeeze(0).squeeze(0).cpu()
            label = label.squeeze(0).squeeze(0).cpu()

            output_array = np.array(output * 255).astype(np.float32)
            label_array = np.array(label * 255).astype(np.float32)
            ssim = compare_ssim(output_array, label_array, data_range=255)

            psnr_ac += psnr
            ssim_ac += ssim

            if i % 100 == 0:
                SR_image = output_array
                save_index = str(int(epoch * (opt.num_data / opt.batch_size / 100) + (i + 1) / 100))
                SR_image = np.clip(SR_image, a_min=0., a_max=255.)
                SR_image = SR_image.astype(np.uint8)
                io.imsave('./data/test_data/X%d/BTM_test_output/' % opt.scaling_factor + save_index + '.png',
                          SR_image)

    return loss, psnr_ac/5, ssim_ac/5

def run_sisr():
    with torch.no_grad():
        net = SISR()
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(opt.load_model_path))

        data_num = 3064
        for j in range(data_num):
            test = io.imread('./data/train_data/X%d/BTM_LR/' % opt.scaling_factor + '%d' % (j + 1) + '.png')
            test = torch.FloatTensor(test/255).unsqueeze(0).unsqueeze(0).cuda()

            output = net(test)
            output = torch.clamp(output, 0.0, 1.0)
            output = output.squeeze(0).squeeze(0).cpu()

            output_array = np.array(output * 255).astype(np.float32)

            SR_image = output_array
            SR_image = np.clip(SR_image, a_min=0., a_max=255.)
            SR_image = SR_image.astype(np.uint8)
            io.imsave('./data/train_data/X4/BTM_sisr_result/%d.png' % (j + 1),
                      SR_image)


def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()

def train_seg(i=0, logger=None, args=None):
    load_path = './model/X4/BTM_model_final.pth'
    modelpth = './model'
    Method = 'X4'
    modelpth = os.path.join(modelpth, Method)
    os.makedirs(modelpth, mode=0o777, exist_ok=True)


    n_classes = 2
    n_img_per_gpu = args.batch_size
    n_workers = 4
    cropsize = [512, 512]
    ds = CityScapes('./data/', cropsize=cropsize, mode='train_data', Method=Method)
    dl = DataLoader(
        ds,
        batch_size=n_img_per_gpu,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )

    ignore_idx = 255
    net = BiSeNet(n_classes=n_classes)
    if i > 0:
        net.load_state_dict(torch.load(load_path))
    net.cuda()
    net.train()
    print('Load Pre-trained Segmentation Model:{}!'.format(load_path))
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    criteria_p = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 0.01

    max_iter = 40000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    it_start = i * 4000
    iter_nums = 4000

    optim = Optimizer(
        model=net,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power,
        it=it_start,
    )

    msg_iter = 10
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(iter_nums):
        try:
            im, lb, _ = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            diter = iter(dl)
            im, lb, _ = next(diter)
        im = im.cuda()
        im = im.repeat(1, 3, 1, 1)
        lb = lb.cuda()
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, mid = net(im)

        lossp = criteria_p(out, lb)
        loss2 = criteria_16(mid, lb)
        loss = lossp + loss2
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            if lr <= 1e-6: lr = 1e-6
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join(
                [
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]
            ).format(
                it=it_start + it + 1, max_it=max_iter, lr=lr, loss=loss_avg, time=t_intv, eta=eta
            )
            logger.info(msg)
            loss_avg = []
            st = ed

    save_pth = osp.join(modelpth, 'BTM_model_final.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    logger.info(
        'Segmentation Model Training done~, The Model is saved to: {}'.format(
            save_pth)
    )
    logger.info('\n')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=32)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    logpath = './logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    for i in range(8):
        train_sisr(i)
        print("|{0} Train sisr Model Sucessfully~!".format(i + 1))
        run_sisr()
        print("|{0} test sisr Model Sucessfully~!".format(i + 1))
        train_seg(i, logger, args)
        print("|{0} Train Segmentation Model Sucessfully~!".format(i + 1))
    print("training Done!")
