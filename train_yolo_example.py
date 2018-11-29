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


def train():
    useCuda = True
    # setup train and eval set
    if torch.cuda.is_available() and useCuda:
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    opt.batch_size = 10
    opt.reduction = 32
    opt.num_epoches = 3
    opt.momentum = 0.9
    opt.decay = 0.0005
    opt.image_size = 416  #todo hardcoded bad
    dataset_file = 'dataset_utils/Mot17_test_single.txt'
    #dataset_file = 'dataset_utils/Mot17_1_video.txt'
    trans = torchvision.transforms.Compose([torchvision.transforms.Resize((opt.image_size, opt.image_size))])
    training_set = MOT_bb_singleframe(dataset_file, transform=trans)
    training_loader = DataLoader(training_set, shuffle=True)

    eval_set = MOT_bb_singleframe_eval(dataset_file, transform=trans)
    eval_loader = DataLoader(eval_set)

    # load the model
    opt.log_path = './log/test'
    opt.pre_trained_model_path = './models/yolo80_coco.pt'
    # save convention: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    load_strict = False
    model = Yolo(0, anchors=[(6.88, 27.44), (11.93, 58.32), (19.90, 94.92), (40.00, 195.84), (97.96, 358.62)])
    if torch.cuda.is_available() and useCuda:
        device = torch.device("cuda")
        model_state_dict = torch.load(opt.pre_trained_model_path, map_location="cuda:0")
        #model.load_state_dict(torch.load(opt.pre_trained_model_path, map_location="cuda:0"), strict=load_strict)
    else:
        device = torch.device('cpu')
        model_state_dict = torch.load(opt.pre_trained_model_path, map_location='cpu')
        #model.load_state_dict(torch.load(opt.pre_trained_model_path, map_location='cpu'), strict=load_strict)
    del model_state_dict["stage3_conv2.weight"]  # todo hacki
    model.load_state_dict(model_state_dict, strict=load_strict)
    model.to(device)

    # The following line will re-initialize weight for the last layer, which is useful
    # when you want to retrain the model based on my trained weights. if you uncomment it,
    # you will see the loss is already very small at the beginning.
    nn.init.normal_(list(model.modules())[-1].weight, 0, 0.01)

    # log stuff
    log_path = opt.log_path
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    if torch.cuda.is_available() and useCuda:
        writer.add_graph(model.cpu(), torch.rand(opt.batch_size, 3, opt.image_size, opt.image_size))
        model.cuda()
    else:
        writer.add_graph(model, torch.rand(opt.batch_size, 3, opt.image_size, opt.image_size))

    # loss and optimizer
    criterion = yloss(training_set.num_classes, model.anchors, opt.reduction)
    #  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(opt.momentum, 0.999), weight_decay=opt.decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=opt.momentum, weight_decay=opt.decay)
    epoch_len = len(training_loader)
    for epoch in range(opt.num_epoches):
        print('num epoch: {:4d}'.format(epoch))
        model.train()
        for img_nr, (img, gt) in enumerate(training_loader):
            if torch.cuda.is_available() and useCuda:
                img = Variable(img.cuda(), requires_grad=True)
            else:
                img = Variable(img, requires_grad=True)
            optimizer.zero_grad()
            logits = model(img)
            loss, loss_coord, loss_conf = criterion(logits, gt)
            loss.backward()
            optimizer.step()
            if img_nr % 100 == 0:
                print('Iteration {0:7d} loss: {1:8.4f}, \tcoord_loss: {2:8.4f}, \tconf_loss: {3:8.4f}'
                      .format(epoch*epoch_len + img_nr,
                              loss.detach().cpu().numpy(),
                              loss_coord.detach().cpu().numpy(),
                              loss_conf.detach().cpu().numpy()))

            writer.add_scalar('Train/Total_loss', loss, epoch*epoch_len + img_nr)
            writer.add_scalar('Train/Coordination_loss', loss_coord, epoch*epoch_len + img_nr)
            writer.add_scalar('Train/Confidence_loss', loss_conf, epoch*epoch_len + img_nr)

        # eval stuff
        # model, eval_loader, criterion, writer/out_losses
        model.eval()
        loss_ls = []
        loss_coord_ls = []
        loss_conf_ls = []
        for te_iter, te_batch in enumerate(eval_loader):
            te_image, te_label = te_batch
            num_sample = len(te_label)
            if torch.cuda.is_available() and useCuda:
                te_image = te_image.cuda()
            with torch.no_grad():
                te_logits = model(te_image)
                batch_loss, batch_loss_coord, batch_loss_conf = criterion(te_logits, te_label)
            loss_ls.append(batch_loss * num_sample)
            loss_coord_ls.append(batch_loss_coord * num_sample)
            loss_conf_ls.append(batch_loss_conf * num_sample)
        te_loss = sum(loss_ls) / eval_set.__len__()
        te_coord_loss = sum(loss_coord_ls) / eval_set.__len__()
        te_conf_loss = sum(loss_conf_ls) / eval_set.__len__()
        print('{}  {}   {}'.format(te_loss, te_coord_loss, te_conf_loss))
        writer.add_scalar('Test/Total_loss', te_loss, epoch)
        writer.add_scalar('Test/Coordination_loss', te_coord_loss, epoch)
        writer.add_scalar('Test/Confidence_loss', te_conf_loss, epoch)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   opt.log_path+'/snapshot{:04d}.tar'.format(epoch))

    #writer.export_scalars_to_json(log_path + os.sep + "all_logs.json")
    writer.close()
    #torch.save(model.state_dict(), '/home/marc/Documents/projects/ADL4CV_project/models/my_test_model/model_state_dict.pt')


if __name__ == "__main__":
    opt = get_args()
    train()
