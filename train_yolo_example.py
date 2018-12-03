"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import argparse
import numpy as np
import torch.nn as nn
import cv2
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

def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=10, help="The number of images per batch")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_epoches", type=int, default=160)
    parser.add_argument("--test_interval", type=int, default=5, help="Number of epoches between testing phases")
    parser.add_argument("--object_scale", type=float, default=1.0)
    parser.add_argument("--noobject_scale", type=float, default=0.5)
    parser.add_argument("--class_scale", type=float, default=1.0)
    parser.add_argument("--coord_scale", type=float, default=5.0)
    parser.add_argument("--reduction", type=int, default=32)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="train")
    parser.add_argument("--test_set", type=str, default="val")
    parser.add_argument("--year", type=str, default="2014", help="The year of dataset (2014 or 2017)")
    parser.add_argument("--data_path", type=str, default="data/COCO", help="the root folder of dataset")
    parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="model")
    parser.add_argument("--pre_trained_model_path", type=str, default="trained_models/whole_model_trained_yolo_coco")
    parser.add_argument("--log_path", type=str, default="tensorboard/yolo_coco")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


class Opt(object):
    def __init__(self):
        self.batch_size = 1
        self.reduction = 32
        self.num_epoches = 20
        self.momentum = 0.9
        self.decay = 0.0005
        self.image_size = 416
        self.log_path = './log/test'
        self.pre_trained_model_path = './models/yolo80_coco.pt'
        self.dataset_file = 'dataset_utils/Mot17_test_single.txt'
        self.learning_rate = 1e-6
        self.useCuda = True  # for easy disabeling cuda
        self.num_workers = 4
        self.reinitailise_last_layer = True
        self.model = None
        self.criterion = None
        self.optimizer = None

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
        model_state_dict = torch.load(opt.pre_trained_model_path, map_location="cuda:0")
    else:
        device = torch.device('cpu')
        model_state_dict = torch.load(opt.pre_trained_model_path, map_location='cpu')
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

    #model = Yolo(0, anchors=[(6.88, 27.44), (11.93, 58.32), (19.90, 94.92), (40.00, 195.84), (97.96, 358.62)])
    #loadYoloBaseWeights(model, opt)

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

    # loss and optimizer
    #criterion = yloss(0, opt.model.anchors, opt.reduction)
    if opt.optimizer is None:
        opt.optimizer = torch.optim.Adam(opt.model.parameters(), lr=opt.learning_rate, betas=(opt.momentum, 0.999),
                                         weight_decay=opt.decay)
    #optimizer = torch.optim.SGD(opt.model.parameters(), lr=opt.learning_rate,
    # momentum=opt.momentum, weight_decay=opt.decay)

    #timing stuff for speed eval
    load_img = []
    run_network =[]
    run_loss=[]
    optim_backward = []
    log_time = []

    epoch_len = len(training_loader)
    for epoch in range(opt.num_epoches):
        print('num epoch: {:4d}'.format(epoch))
        opt.model.train()
        #temp_time = time.time()
        for img_nr, (img, gt) in enumerate(training_loader):
            if torch.cuda.is_available() and opt.useCuda:
                img = Variable(img.cuda(), requires_grad=True)
            else:
                img = Variable(img, requires_grad=True)
            #load_img.append(time.time()-temp_time)
            opt.optimizer.zero_grad()
            #temp_time = time.time()
            logits = opt.model(img)
            #run_network.append(time.time()-temp_time)
            #temp_time = time.time()
            loss, loss_coord, loss_conf = opt.criterion(logits, gt)
            #run_loss.append(time.time()-temp_time)
            #temp_time = time.time()
            writeLossToSummary(writer, 'Train', loss.item(),
                               loss_coord.item(), loss_conf.item(), epoch*epoch_len + img_nr)
            #nr = epoch*epoch_len + img_nr
            #writer.add_scalar('Train/Total_loss', loss.item(), nr)
            #writer.add_scalar('Train/Coordination_loss', loss_coord.item(), nr)
            #writer.add_scalar('Train/Confidence_loss', loss_conf.item(), nr)
            #log_time.append(time.time()-temp_time)
            #temp_time = time.time()
            loss.backward()
            opt.optimizer.step()
            #optim_backward.append(time.time()-temp_time)
            if img_nr % 1000 == 0:
                print('Iteration {0:7d} loss: {1:8.4f}, \tcoord_loss: {2:8.4f}, \tconf_loss: {3:8.4f}'
                      .format(epoch*epoch_len + img_nr,
                              loss.detach().cpu().numpy(),
                              loss_coord.detach().cpu().numpy(),
                              loss_conf.detach().cpu().numpy()))

            #if img_nr > 99:
            #    break
            #time.sleep(1)
            #temp_time = time.time()
        #print('load_time: {}'.format(np.mean(np.array(load_img))))
        #print('network_time: {}'.format(np.mean(np.array(run_network))))
        #print('loss_time: {}'.format(np.mean(np.array(run_loss))))
        #print('optim_time: {}'.format(np.mean(np.array(optim_backward))))
        #print('log_time: {}'.format(np.mean(np.array(log_time))))
        #writer.close()
        #exit()

        # eval stuff

        # model, eval_loader, criterion, writer/out_losses
        opt.model.eval()
        loss_ls = []
        loss_coord_ls = []
        loss_conf_ls = []
        for te_iter, te_batch in enumerate(eval_loader):
            te_image, te_label = te_batch
            num_sample = len(te_label)
            if torch.cuda.is_available() and opt.useCuda:
                te_image = te_image.cuda()
            with torch.no_grad():
                te_logits = opt.model(te_image)
                batch_loss, batch_loss_coord, batch_loss_conf = opt.criterion(te_logits, te_label)
            loss_ls.append(batch_loss * num_sample)
            loss_coord_ls.append(batch_loss_coord * num_sample)
            loss_conf_ls.append(batch_loss_conf * num_sample)
        te_loss = sum(loss_ls) / eval_set.__len__()
        te_coord_loss = sum(loss_coord_ls) / eval_set.__len__()
        te_conf_loss = sum(loss_conf_ls) / eval_set.__len__()
        print('{}  {}   {}'.format(te_loss, te_coord_loss, te_conf_loss))

        writeLossToSummary(writer, 'Test', te_loss.item(),
                           te_coord_loss.item(), te_conf_loss.item(), epoch * epoch_len)
        #writer.add_scalar('Test/Total_loss', te_loss, epoch*epoch_len)
        #writer.add_scalar('Test/Coordination_loss', te_coord_loss, epoch*epoch_len)
        #writer.add_scalar('Test/Confidence_loss', te_conf_loss, epoch*epoch_len)

        torch.save({'epoch': epoch, 'model_state_dict': opt.model.state_dict(),
                    'optimizer_state_dict': opt.optimizer.state_dict()},
                   opt.log_path+'/snapshot{:04d}.tar'.format(epoch))

    #writer.export_scalars_to_json(log_path + os.sep + "all_logs.json")
    writer.close()


if __name__ == "__main__":
    #opt = get_args()
    opt = Opt()
    opt.useCuda = True
    opt.learning_rate = 1e-5
    opt.batch_size = 1
    opt.model = Yolo(0, anchors=[(6.88, 27.44), (11.93, 58.32), (19.90, 94.92), (40.00, 195.84), (97.96, 358.62)])
    loadYoloBaseWeights(opt.model, opt)
    opt.criterion = yloss(0, opt.model.anchors, opt.reduction)
    train(opt)
