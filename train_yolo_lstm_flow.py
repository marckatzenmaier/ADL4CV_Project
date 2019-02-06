"""
@author Nikita Kister
This script trains the yolo lstm flow network using the finetuned yolo detector.
"""
from dataset_utils.datasets.MotBBImageSequence import *
from torch.utils.data import DataLoader, Subset
from Flow import preprocess
from yolo.yolo_LSTM import BigYoloFlowLSTM
from yolo.loss import YoloLoss
from train_yolo import Opt
from tensorboardX import SummaryWriter
import shutil
from yolo.yolo_utils import filter_non_zero_gt, get_ap, load_Snapshot_to_yolo_LSTM
from train_yolo import writeLossToSummary


def filter_gt_batch(gt):
    # used to filter only existing bb for loss
    t = gt[gt[:, :, 0] != 0.0]

    return t[:, 1:]


def train(opt):
    # log stuff
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    dataset = MotBBImageSequence('dataset_utils/Mot17_test_single.txt', use_only_first_video=False,
                                 new_width=832,
                                 new_height=832)
    train_data = Subset(dataset, range(0, dataset.valid_begin))
    valid_data = Subset(dataset, range(dataset.valid_begin, len(dataset)))
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=opt.num_workers, drop_last=True)

    obs_length = 10

    epoch_len = len(train_loader)

    for epoch in range(opt.num_epoches):
        for img_num, (gt, b, img1, img2) in enumerate(train_loader):

            opt.model.train()

            opt.model.lstm_part.reinit_lstm(opt.batch_size)

            seq_loss = 0
            seq_loss_coord, seq_loss_conf = (0.0, 0.0)
            for i in range(obs_length):
                single_gt = b[:, i].to(opt.device())
                single_img2 = img2[:, i].to(opt.device())
                single_img2_normed = preprocess(img2[:, i]).to(opt.device())
                single_img1_normed = preprocess(img1[:, i]).to(opt.device())

                double_image = torch.cat((single_img1_normed, single_img2_normed), dim=1)

                logits = opt.model((double_image, single_img2 * 255.))

                loss, loss_coord, loss_conf = opt.yolo_loss(logits, single_gt)

                seq_loss += loss
                seq_loss_conf += loss_conf.item()
                seq_loss_coord += loss_coord.item()

            seq_loss.backward()

            opt.optimizer_encoder.step()
            opt.optimizer_encoder.zero_grad()

            seq_loss = seq_loss.item()

            writeLossToSummary(writer, 'Train', seq_loss / obs_length,
                               seq_loss_coord / obs_length,
                               seq_loss_conf / obs_length,
                               epoch * epoch_len + img_num)

            print(f"epoch:{epoch} it: {img_num}")
            print(f"loss_seq: {seq_loss/obs_length}, "
                  f"loss_coord: {seq_loss_coord / obs_length}, "
                  f"loss_conf: {seq_loss_conf / obs_length}")

        print("###############")
        print("# VALIDATION BEGIN")
        print("###############")
        opt.model.eval()
        valid_len = len(valid_loader)
        loss_val = 0
        loss_ap = 0
        loss_coord_val = 0
        loss_conf_val = 0
        for img_num, (gt, b, img1, img2) in enumerate(valid_loader):
            opt.model.lstm_part.reinit_lstm(opt.batch_size)
            opt.optimizer_encoder.zero_grad()

            seq_loss = 0
            seq_loss_coord, seq_loss_conf = (0.0, 0.0)
            seq_ap = 0
            for i in range(obs_length):
                single_gt = b[:, i].to(opt.device())
                single_img2 = img2[:, i].to(opt.device())
                single_img2_normed = preprocess(img2[:, i]).to(opt.device())
                single_img1_normed = preprocess(img1[:, i]).to(opt.device())

                double_image = torch.cat((single_img1_normed, single_img2_normed), dim=1)
                with torch.no_grad():
                    logits = opt.model((double_image, single_img2 * 255.))

                    loss, loss_coord, loss_conf = opt.yolo_loss(logits, single_gt)

                seq_ap += get_ap(logits.detach(), filter_gt_batch(single_gt), opt.image_size, opt.image_size,
                                 opt.model.anchors)
                seq_loss += loss
                seq_loss_conf += loss_conf.item()
                seq_loss_coord += loss_coord.item()

            loss_val += seq_loss.item() / valid_len / obs_length
            loss_coord_val += seq_loss_coord / valid_len / obs_length
            loss_conf_val += seq_loss_conf / valid_len / obs_length
            loss_ap += seq_ap / obs_length / valid_len

        writeLossToSummary(writer, 'Val', loss_val, loss_coord_val,
                           loss_conf_val, (epoch + 1) * epoch_len)

        print(f"epoch:{epoch}")
        print(f"loss_seq: {loss_val}, loss_coord: {loss_coord_val}, loss_conf: {loss_conf_val}, mAP: {loss_ap}")
        writer.add_scalar('Val/mAP', loss_ap, (epoch + 1) * epoch_len)

        print("###############")
        print("# VALIDATION END")
        print("###############")

        torch.save({'epoch': epoch, 'model_state_dict': opt.model.state_dict(),
                    'optimizer_state_dict': opt.optimizer_encoder.state_dict()},
                   opt.log_path + f'/snapshot_encoder{epoch}.tar')

    writer.close()


def filter_gt_batch(gt):

    # used to filter only existing bb for loss
    t = gt[gt[:, :, 0] != 0.0]

    return t[:, 1:]


if __name__ == "__main__":
    opt = Opt()
    opt.num_epoches = 8
    opt.image_size = 832
    opt.useCuda = False
    opt.learning_rate = 1e-4
    opt.decay = 1e-5
    opt.batch_size = 1
    opt.num_workers = 4
    anchors = [(0.43, 1.715), (0.745625, 3.645), (1.24375, 5.9325), (2.5, 12.24), (6.1225, 22.41375)]
    opt.model = BigYoloFlowLSTM(opt.batch_size, anchors=anchors)
    opt.pre_trained_yolo_path = './models/yolo_832.tar'
    load_Snapshot_to_yolo_LSTM(opt.model, opt)
    # all parameters are on the device the hidden states are not parameters
    opt.model.to(opt.device())
    opt.yolo_loss = YoloLoss(opt.model.anchors, filter_fkt=filter_non_zero_gt)
    opt.optimizer_encoder = torch.optim.Adam(opt.model.parameters(), lr=opt.learning_rate, betas=(opt.momentum, 0.999),
                                             weight_decay=opt.decay)

    train(opt)
