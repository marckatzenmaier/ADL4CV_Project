
import os
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from yolo.yolo_utils import *
from yolo.yolo_net import Yolo
from yolo.loss import YoloLoss as yloss
import torchvision
from tensorboardX import SummaryWriter
import shutil
import time
from dataset_utils.datasets.MOT_bb_singleframe import MOT_bb_singleframe
from dataset_utils.datasets.MOT_bb_singleframe import MOT_bb_singleframe_eval
import dataset_utils.MOT_utils as motu


class Opt(object):
    def __init__(self):
        self.batch_size = 1
        self.reduction = 32
        self.num_epoches = 20
        self.momentum = 0.9
        self.decay = 0.0005
        self.image_size = 416
        self.encoding_size = self.image_size // self.reduction
        self.log_path = './log/test'
        self.pre_trained_yolo_path = './models/yolo80_coco.pt'
        self.pre_trained_flownet_path = './models/flownets_EPE1.951.pth.tar'
        self.dataset_file = 'dataset_utils/Mot17_test_single.txt'
        self.learning_rate = 1e-6
        self.useCuda = True  # for easy disabeling cuda
        self.num_workers = 4
        self.reinitailise_last_layer = True
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.conf_threshold = 0.25


def writeLossToSummary(writer, prefix, loss, loss_coord, loss_conf, index):
    writer.add_scalar(prefix + '/Total_loss', loss, index)
    writer.add_scalar(prefix + '/Coordination_loss', loss_coord, index)
    writer.add_scalar(prefix + '/Confidence_loss', loss_conf, index)


def loadTrainEvalSet(opt):
    trans = torchvision.transforms.Compose([torchvision.transforms.Resize((opt.image_size, opt.image_size))])
    training_set = MOT_bb_singleframe(opt.dataset_file, transform=trans)
    training_loader = DataLoader(training_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                                 collate_fn=custom_collate_fn)

    eval_set = MOT_bb_singleframe_eval(opt.dataset_file, transform=trans)
    eval_loader = DataLoader(eval_set, batch_size=opt.batch_size, num_workers=opt.num_workers,
                             collate_fn=custom_collate_fn)
    return training_set, training_loader, eval_set, eval_loader


def loadYoloBaseWeights(yolo_model, opt):
    # load the model
    # save convention: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    load_strict = False
    if torch.cuda.is_available() and opt.useCuda:
        device = torch.device("cuda")
        model_state_dict = torch.load(opt.pre_trained_yolo_path, map_location="cuda:0")
    else:
        device = torch.device('cpu')
        model_state_dict = torch.load(opt.pre_trained_yolo_path, map_location='cpu')
    del model_state_dict["stage3_conv2.weight"]
    yolo_model.load_state_dict(model_state_dict, strict=load_strict)
    yolo_model.to(device)
    if opt.reinitailise_last_layer:
        nn.init.normal_(list(yolo_model.modules())[-1].weight, 0, 0.01)


def train(opt):
    # setup train and eval set
    if torch.cuda.is_available() and opt.useCuda:
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    training_set, training_loader, eval_set, eval_loader = loadTrainEvalSet(opt)

    # log stuff
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    if torch.cuda.is_available() and opt.useCuda:
        writer.add_graph(opt.model.cpu(), torch.rand(opt.batch_size, 3, opt.image_size, opt.image_size))
        opt.model.cuda()
    else:
        writer.add_graph(opt.model, torch.rand(opt.batch_size, 3, opt.image_size, opt.image_size))

    # write the hyperparams lr, batchsize  and imagesize in the tensorboard file
    writer.add_text('Hyperparams',
                    'lr: {}, \nbatchsize: {}, \nimg_size:{}'.format(opt.learning_rate, opt.batch_size, opt.image_size))

    # loss and optimize
    if opt.optimizer is None:
        opt.optimizer = torch.optim.Adam(opt.model.parameters(), lr=opt.learning_rate, betas=(opt.momentum, 0.999),
                                         weight_decay=opt.decay)

    epoch_len = len(training_loader)
    for epoch in range(opt.num_epoches):
        print('num epoch: {:4d}'.format(epoch))
        opt.model.train()
        for img_nr, (img, gt) in enumerate(training_loader):
            break
            if torch.cuda.is_available() and opt.useCuda:
                img = Variable(img.cuda(), requires_grad=True)
            else:
                img = Variable(img, requires_grad=True)
            opt.optimizer.zero_grad()
            logits = opt.model(img)
            loss, loss_coord, loss_conf = opt.criterion(logits, gt)
            writeLossToSummary(writer, 'Train', loss.item(),
                               loss_coord.item(), loss_conf.item(), epoch*epoch_len + img_nr)
            loss.backward()
            opt.optimizer.step()

        # eval stuff
        opt.model.eval()
        loss_ls = []
        loss_coord_ls = []
        loss_conf_ls = []
        all_ap = []
        all_ap1 = []
        for te_iter, te_batch in enumerate(eval_loader):
            te_image, te_label = te_batch
            num_sample = len(te_label)
            if torch.cuda.is_available() and opt.useCuda:
                te_image = te_image.cuda()
            with torch.no_grad():
                te_logits = opt.model(te_image)
                batch_loss, batch_loss_coord, batch_loss_conf = opt.criterion(te_logits, te_label)
                for i in range(num_sample):
                    ap = get_ap(te_logits[0].unsqueeze(0), gt[0], opt.image_size, opt.image_size,opt.model.anchors, .5)
                    ap1 = get_ap(te_logits[0].unsqueeze(0), gt[0], opt.image_size, opt.image_size,opt.model.anchors, .8)
                    all_ap.append(ap)
                    all_ap1.append(ap1)
                #if te_iter % 10 ==0:
                #    img = np.array(draw_img(te_logits, te_image[0], opt.image_size, opt.model.anchors))
                #    print(img.shape)
                #    print(img.dtype)
                #    print(type(img))
                #    writer.add_image(f'Val/{epoch}', img)
            loss_ls.append(batch_loss * num_sample)
            loss_coord_ls.append(batch_loss_coord * num_sample)
            loss_conf_ls.append(batch_loss_conf * num_sample)
        te_loss = sum(loss_ls) / eval_set.__len__()
        te_coord_loss = sum(loss_coord_ls) / eval_set.__len__()
        te_conf_loss = sum(loss_conf_ls) / eval_set.__len__()
        print('{}  {}   {}'.format(te_loss, te_coord_loss, te_conf_loss))
        writer.add_scalar('Val/AP0.5', np.mean(np.array(all_ap)), epoch * epoch_len)
        writer.add_scalar('Val/AP0.8', np.mean(np.array(all_ap1)), epoch * epoch_len)
        writeLossToSummary(writer, 'Val', te_loss.item(),
                           te_coord_loss.item(), te_conf_loss.item(), epoch * epoch_len)

        torch.save({'epoch': epoch, 'model_state_dict': opt.model.state_dict(),
                    'optimizer_state_dict': opt.optimizer.state_dict()},
                   opt.log_path+'/snapshot{:04d}.tar'.format(epoch))
    writer.close()


if __name__ == "__main__":
    opt = Opt()
    opt.useCuda = True
    opt.learning_rate = 1e-5
    opt.batch_size = 1
    opt.model = Yolo(0, anchors=[(0.215, 0.8575), (0.3728125, 1.8225), (0.621875, 2.96625),
                                            (1.25, 6.12), (3.06125, 11.206875)])
    loadYoloBaseWeights(opt.model, opt)
    opt.criterion = yloss(opt.model.anchors, opt.reduction)
    train(opt)
