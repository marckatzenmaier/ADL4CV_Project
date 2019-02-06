"""
@author Marc Katzenmaier
This script trains the YOLO network
"""
import torch.nn as nn
from yolo.yolo_utils import *
from yolo.yolo_net import Yolo
from yolo.loss import YoloLoss as yloss
from tensorboardX import SummaryWriter
import shutil
from dataset_utils.datasets.MotBBImageSingle import *
from torch.utils.data import DataLoader, Subset


class Opt(object):
    """
    class containing all parameter for training the network
    """
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

    def device(self):
        return torch.device('cuda') if self.useCuda else torch.device('cpu')


def writeLossToSummary(writer, prefix, loss, loss_coord, loss_conf, index):
    """
    addes all parts of the YOLO loss to the tensorboardX writer
    :param writer: the tensorboardX writer
    :param prefix: prefix of the losses eg 'train/' or 'val/'
    :param loss: the total YOLO loss
    :param loss_coord: the coordination loss part
    :param loss_conf: the confidence loss part
    :param index: which iteration
    """
    writer.add_scalar(prefix + '/Total_loss', loss, index)
    writer.add_scalar(prefix + '/Coordination_loss', loss_coord, index)
    writer.add_scalar(prefix + '/Confidence_loss', loss_conf, index)


def loadTrainEvalSet(opt):
    dataset = MotBBImageSingle('dataset_utils/Mot17_test_single.txt', use_only_first_video=False, seq_length=1,
                               new_height=opt.image_size, new_width=opt.image_size)
    training_set = Subset(dataset, range(0, dataset.valid_begin))
    eval_set = Subset(dataset, range(dataset.valid_begin, len(dataset)))
    training_loader = DataLoader(training_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              drop_last=False)
    eval_loader = DataLoader(eval_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                              drop_last=False)
    return training_set, training_loader, eval_set, eval_loader


def loadYoloBaseWeights(yolo_model, opt):
    """
    load the model basis weights using https://pytorch.org/tutorials/beginner/saving_loading_models.html
    as convention
    :param yolo_model: the model where the weights should be applyed
    :param opt: option for loading the weights include path of the snapshot and where it should be restored (cpu/cuda)
    """
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
    """
    performs the training
    :param opt: all parameters inclusive network and opimizer are here combined
    """
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
        training_set.dataset.is_training = True
        print('num epoch: {:4d}'.format(epoch))
        opt.model.train()
        for img_nr, (gt, img) in enumerate(training_loader):
            if torch.cuda.is_available() and opt.useCuda:
                img = Variable(img.cuda(), requires_grad=True)
            else:
                img = Variable(img, requires_grad=True)
            opt.optimizer.zero_grad()
            logits = opt.model(img)
            loss, loss_coord, loss_conf = opt.criterion(logits, gt)
            writeLossToSummary(writer, 'Train', loss.item(),
                               loss_coord.item(), loss_conf.item(), epoch * epoch_len + img_nr)
            loss.backward()
            opt.optimizer.step()

        # eval stuff
        opt.model.eval()
        eval_set.dataset.is_training = False
        loss_ls = []
        loss_coord_ls = []
        loss_conf_ls = []
        all_ap = []
        all_ap1 = []
        for te_iter, te_batch in enumerate(eval_loader):
            te_label, te_image = te_batch
            num_sample = len(te_label)
            if torch.cuda.is_available() and opt.useCuda:
                te_image = te_image.cuda()
            with torch.no_grad():
                te_logits = opt.model(te_image)
                batch_loss, batch_loss_coord, batch_loss_conf = opt.criterion(te_logits, te_label)
                for i in range(num_sample):
                    ap = get_ap(te_logits[i], filter_non_zero_gt_without_id(te_label[i]), opt.image_size,
                                opt.image_size, opt.model.anchors, .5)
                    ap1 = get_ap(te_logits[i], filter_non_zero_gt_without_id(te_label[i]), opt.image_size,
                                 opt.image_size, opt.model.anchors, .8)
                    if not np.isnan(ap):
                        all_ap.append(ap)
                    if not np.isnan(ap1):
                        all_ap1.append(ap1)
            loss_ls.append(batch_loss * num_sample)
            loss_coord_ls.append(batch_loss_coord * num_sample)
            loss_conf_ls.append(batch_loss_conf * num_sample)
        te_loss = sum(loss_ls) / eval_set.__len__()
        te_coord_loss = sum(loss_coord_ls) / eval_set.__len__()
        te_conf_loss = sum(loss_conf_ls) / eval_set.__len__()
        writer.add_scalar('Val/AP0.5', np.mean(np.array(all_ap)), epoch * epoch_len)
        writer.add_scalar('Val/AP0.8', np.mean(np.array(all_ap1)), epoch * epoch_len)
        writeLossToSummary(writer, 'Val', te_loss.item(),
                           te_coord_loss.item(), te_conf_loss.item(), epoch * epoch_len)

        torch.save({'epoch': epoch, 'model_state_dict': opt.model.state_dict(),
                    'optimizer_state_dict': opt.optimizer.state_dict()},
                   opt.log_path + '/snapshot{:04d}.tar'.format(epoch))
    writer.close()


if __name__ == "__main__":
    opt = Opt()
    opt.batch_size = 10
    opt.num_epoches = 100
    opt.useCuda = True
    opt.learning_rate = 1e-4
    opt.num_workers = 4
    anchors = [(0.43, 1.715), (0.745625, 3.645), (1.24375, 5.9325), (2.5, 12.24), (6.1225, 22.41375)]
    opt.model = Yolo(0, anchors=anchors)
    loadYoloBaseWeights(opt.model, opt)
    opt.criterion = yloss(opt.model.anchors, opt.reduction, filter_fkt=filter_non_zero_gt_without_id)
    opt.log_path = './log/yolo'
    opt.image_size = 832
    opt.model.image_size = opt.image_size
    loadYoloBaseWeights(opt.model, opt)
    opt.criterion = yloss(opt.model.anchors, opt.reduction, filter_fkt=filter_non_zero_gt_without_id)
    train(opt)
