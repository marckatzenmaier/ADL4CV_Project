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


def train(opt):
    useCuda = False
    # setup train and eval set
    if torch.cuda.is_available() and useCuda:
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    learning_rate_schedule = {"0": 1e-5, "5": 1e-4,
                              "80": 1e-5, "110": 1e-6}
    #training_params = {"batch_size": opt.batch_size,
    #                   "shuffle": True,
    #                   "drop_last": True,
    #                   "collate_fn": custom_collate_fn}
    #test_params = {"batch_size": opt.batch_size,
    #               "shuffle": False,
    #               "drop_last": False,
    #               "collate_fn": custom_collate_fn}

    #todo hardcoded path

    opt.image_size = 96  #todo hardcoded bad
    dataset_file = '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/Mot17_test_single.txt'
    trans = torchvision.transforms.Compose([torchvision.transforms.Resize((opt.image_size, opt.image_size))])
    training_set = MOT_bb_singleframe(dataset_file, transform=trans)
    training_loader = DataLoader(training_set)

    #test_set = COCODataset(opt.data_path, opt.year, opt.test_set, opt.image_size, is_training=False)
    #test_generator = DataLoader(test_set, **test_params)

    # load the model
    #todo hardcoded path
    opt.pre_trained_model_path = '/home/marc/Documents/projects/ADL4CV_project/models/coco2014_2017/trained_models/whole_model_trained_yolo_coco'
    opt.log_path = '/home/marc/Documents/projects/ADL4CV_project/log/test'
    opt.pre_trained_model_path = '/home/marc/Documents/projects/ADL4CV_project/models/my_test_model/model_state_dict.pt'
    opt.pre_trained_model_path = '/home/marc/Documents/projects/ADL4CV_project/models/baseWeights/yolo80.pt'
    # todo use convention like: https://pytorch.org/tutorials/beginner/saving_loading_models.html is much easyer to convert
    '''if torch.cuda.is_available() and useCuda:
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path)
        else:
            model = Yolo(training_set.num_classes)
            model.load_state_dict(torch.load(opt.pre_trained_model_path))
    else:
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage)
        else:
            model = Yolo(training_set.num_classes)
            model.load_state_dict(torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage))'''

    model = Yolo(training_set.num_classes)
    if torch.cuda.is_available() and useCuda:
        device = torch.device("cuda")
        model.load_state_dict(torch.load(opt.pre_trained_model_path, map_location="cuda:0"))
    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load(opt.pre_trained_model_path, map_location='cpu'))
    model.to(device)

    # The following line will re-initialize weight for the last layer, which is useful
    # when you want to retrain the model based on my trained weights. if you uncomment it,
    # you will see the loss is already very small at the beginning.
    nn.init.normal_(list(model.modules())[-1].weight, 0, 0.01)

    # log stuff
    log_path = os.path.join(opt.log_path, "{}".format(opt.year))
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
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=opt.momentum, weight_decay=opt.decay)
    model.train()

    for img, gt in training_loader:
        if torch.cuda.is_available() and useCuda:
            img = Variable(img.cuda(), requires_grad=True)
        else:
            img = Variable(img, requires_grad=True)
        break
    for i in range(20):
        optimizer.zero_grad()
        logits = model(img)
        loss, loss_coord, loss_conf = criterion(logits, gt)
        loss.backward()
        optimizer.step()
        print('Iteration {0:7d} loss: {1:8.4f}, \tcoord_loss: {2:8.4f}, \tconf_loss: {3:8.4f}'.format(i+1, loss.detach().cpu().numpy(), loss_coord.detach().cpu().numpy(),
                                                     loss_conf.detach().cpu().numpy()))

        writer.add_scalar('Train/Total_loss', loss, i)
        writer.add_scalar('Train/Coordination_loss', loss_coord, i)
        writer.add_scalar('Train/Confidence_loss', loss_conf, i)
    writer.export_scalars_to_json(log_path + os.sep + "all_logs.json")
    writer.close()
    torch.save(model.state_dict(), '/home/marc/Documents/projects/ADL4CV_project/models/my_test_model/model_state_dict.pt')

    '''# actual training
    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epoches):
        if str(epoch) in learning_rate_schedule.keys():
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_schedule[str(epoch)]
        for iter, batch in enumerate(training_generator):
            image, label = batch
            if torch.cuda.is_available() and useCuda:
                image = Variable(image.cuda(), requires_grad=True)
            else:
                image = Variable(image, requires_grad=True)
            optimizer.zero_grad()
            logits = model(image)
            loss, loss_coord, loss_conf, loss_cls = criterion(logits, label)
            loss.backward()
            optimizer.step()
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss:{:.2f} (Coord:{:.2f} Conf:{:.2f} Cls:{:.2f})".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss,
                loss_coord,
                loss_conf,
                loss_cls))
            writer.add_scalar('Train/Total_loss', loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Coordination_loss', loss_coord, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Confidence_loss', loss_conf, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Class_loss', loss_cls, epoch * num_iter_per_epoch + iter)
        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            loss_coord_ls = []
            loss_conf_ls = []
            loss_cls_ls = []
            for te_iter, te_batch in enumerate(test_generator):
                te_image, te_label = te_batch
                num_sample = len(te_label)
                if torch.cuda.is_available() and useCuda:
                    te_image = te_image.cuda()
                with torch.no_grad():
                    te_logits = model(te_image)
                    batch_loss, batch_loss_coord, batch_loss_conf, batch_loss_cls = criterion(te_logits, te_label)
                loss_ls.append(batch_loss * num_sample)
                loss_coord_ls.append(batch_loss_coord * num_sample)
                loss_conf_ls.append(batch_loss_conf * num_sample)
                loss_cls_ls.append(batch_loss_cls * num_sample)
            te_loss = sum(loss_ls) / test_set.__len__()
            te_coord_loss = sum(loss_coord_ls) / test_set.__len__()
            te_conf_loss = sum(loss_conf_ls) / test_set.__len__()
            te_cls_loss = sum(loss_cls_ls) / test_set.__len__()
            print("Epoch: {}/{}, Lr: {}, Loss:{:.2f} (Coord:{:.2f} Conf:{:.2f} Cls:{:.2f})".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss,
                te_coord_loss,
                te_conf_loss,
                te_cls_loss))
            writer.add_scalar('Test/Total_loss', te_loss, epoch)
            writer.add_scalar('Test/Coordination_loss', te_coord_loss, epoch)
            writer.add_scalar('Test/Confidence_loss', te_conf_loss, epoch)
            writer.add_scalar('Test/Class_loss', te_cls_loss, epoch)
            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                # torch.save(model, opt.saved_path + os.sep + "trained_yolo_coco")
                torch.save(model.state_dict(), opt.saved_path + os.sep + "only_params_trained_yolo_coco")
                torch.save(model, opt.saved_path + os.sep + "whole_model_trained_yolo_coco")

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break
    writer.export_scalars_to_json(log_path + os.sep + "all_logs.json")
    writer.close()'''


if __name__ == "__main__":
    opt = get_args()
    train(opt)
