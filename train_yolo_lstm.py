"""
@author Marc Katzenmaier
This script trains the YOLO LSTM network
"""
from dataset_utils.datasets.MotBBImageSequence import *
from torch.utils.data import DataLoader, Subset
from yolo.yolo_LSTM import YoloLSTM
from yolo.loss import YoloLoss
from train_yolo import Opt
from tensorboardX import SummaryWriter
import shutil
from yolo.yolo_utils import *

from train_yolo import writeLossToSummary


def load_Snapshot_to_yolo_LSTM(model, opt):
    """
    loads a snapshot of the YOLO net and uses the weights in the YOLO LSTM net
    :param model: YOLO LSTM model
    :param opt: opt class containing yolo path and if use Cuda
    """
    load_strict = False
    if torch.cuda.is_available() and opt.useCuda:
        device = torch.device("cuda")
        model_state_dict = torch.load(opt.pre_trained_yolo_path, map_location="cuda:0")['model_state_dict']
    else:
        device = torch.device('cpu')
        model_state_dict = torch.load(opt.pre_trained_yolo_path, map_location='cpu')['model_state_dict']
    del model_state_dict["stage3_conv2.weight"]
    del model_state_dict["stage3_conv1.0.weight"]
    del model_state_dict["stage3_conv1.1.weight"]
    del model_state_dict["stage3_conv1.1.bias"]
    del model_state_dict["stage3_conv1.1.running_mean"]
    del model_state_dict["stage3_conv1.1.running_var"]
    del model_state_dict["stage3_conv1.1.num_batches_tracked"]
    model.load_state_dict(model_state_dict, strict=load_strict)
    model.to(device)


def train(opt):
    """
    performs the training
    :param opt: all parameters inclusive network and opimizer are here combined
    """
    if torch.cuda.is_available() and opt.useCuda:
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    dataset = MotBBImageSequence('dataset_utils/Mot17_test_single.txt', use_only_first_video=False, new_height=832,
                                 new_width=832)
    train_data = Subset(dataset, range(0, dataset.valid_begin))
    valid_data = Subset(dataset, range(dataset.valid_begin, len(dataset)))
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers)  # , drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=False,
                              num_workers=opt.num_workers)  # , drop_last=True)

    # log stuff
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    writer.add_text('Hyperparams',
                    'lr: {}, \nbatchsize: {}, \nimg_size:{}'.format(opt.learning_rate, opt.batch_size, opt.image_size))

    opt.optimizer = torch.optim.Adam(opt.model.parameters(), lr=opt.learning_rate, betas=(opt.momentum, 0.999),
                                     weight_decay=opt.decay)
    epoch_len = len(train_loader)
    print('epoch_len: {}'.format(epoch_len))
    for epoch in range(opt.num_epoches):
        print('num epoch: {:4d}'.format(epoch))
        opt.model.train()
        for img_nr, (gt, b, img, d) in enumerate(train_loader):
            sequence_len = gt.shape[1]
            opt.model.reinit_lstm(gt.shape[0])
            opt.optimizer.zero_grad()
            seq_loss, seq_loss_coord, seq_loss_conf = (0.0, 0.0, 0.0)
            for i in range(sequence_len):
                if torch.cuda.is_available() and opt.useCuda:
                    single_gt = gt[:, i].cuda()
                    single_img = img[:, i].cuda()
                else:
                    single_gt = gt[:, i]
                    single_img = img[:, i]
                logits = opt.model(single_img * 255.)
                loss, loss_coord, loss_conf = opt.criterion(logits, single_gt)
                seq_loss += loss
                seq_loss_conf += loss_conf.item()
                seq_loss_coord += loss_coord.item()
            seq_loss.backward()
            opt.optimizer.step()
            writeLossToSummary(writer, 'Train', seq_loss.item() / sequence_len, seq_loss_coord / sequence_len,
                               seq_loss_conf / sequence_len, epoch * epoch_len + img_nr)

        # eval stuff
        opt.model.eval()
        valid_len = len(valid_loader)
        loss_eval = 0
        loss_coord_eval = 0
        loss_conf_eval = 0
        all_ap = []
        all_ap1 = []
        for img_nr, (gt, b, img, d) in enumerate(valid_loader):
            sequence_len = gt.shape[1]
            opt.model.reinit_lstm(gt.shape[0])
            seq_loss, seq_loss_coord, seq_loss_conf = (0.0, 0.0, 0.0)
            for i in range(sequence_len):
                if torch.cuda.is_available() and opt.useCuda:
                    single_gt = gt[:, i].cuda()
                    single_img = img[:, i].cuda()
                else:
                    single_gt = gt[:, i]
                    single_img = img[:, i]
                with torch.no_grad():
                    logits = opt.model(single_img * 255.)
                    loss, loss_coord, loss_conf = opt.criterion(logits, single_gt)
                    for i in range(single_gt.shape[0]):
                        ap = get_ap(logits[i], filter_non_zero_gt(single_gt[i]), opt.image_size, opt.image_size,
                                    opt.model.anchors, .5)
                        ap1 = get_ap(logits[i], filter_non_zero_gt(single_gt[i]), opt.image_size, opt.image_size,
                                     opt.model.anchors, .8)
                        if not np.isnan(ap):
                            all_ap.append(ap)
                        if not np.isnan(ap1):
                            all_ap1.append(ap1)
                seq_loss += loss.item()
                seq_loss_conf += loss_conf.item()
                seq_loss_coord += loss_coord.item()
            loss_eval += seq_loss / sequence_len
            loss_coord_eval += seq_loss_coord / sequence_len
            loss_conf_eval += seq_loss_conf / sequence_len

        writeLossToSummary(writer, 'Val', loss_eval / valid_len, loss_coord_eval / valid_len,
                           loss_conf_eval / valid_len, (epoch + 1) * epoch_len)

        writer.add_scalar('Val/AP0.5', np.mean(np.array(all_ap)), epoch * epoch_len)
        writer.add_scalar('Val/AP0.8', np.mean(np.array(all_ap1)), epoch * epoch_len)
        torch.save({'epoch': epoch, 'model_state_dict': opt.model.state_dict(),
                    'optimizer_state_dict': opt.optimizer.state_dict()},
                   opt.log_path + '/snapshot{:04d}.tar'.format(epoch))

    writer.close()


if __name__ == "__main__":
    opt = Opt()
    opt.useCuda = True
    opt.learning_rate = 1e-4
    opt.batch_size = 4
    anchors = [(0.43, 1.715), (0.745625, 3.645), (1.24375, 5.9325), (2.5, 12.24), (6.1225, 22.41375)]
    opt.model = YoloLSTM(opt.batch_size, image_size=832, anchors=anchors)
    opt.pre_trained_yolo_path = './models/yolo_832.tar'
    opt.image_size = 832
    opt.model.load_pretrained_weights(opt.pre_trained_yolo_path)
    if opt.useCuda:
        device = torch.device("cuda")
        opt.model.to(device)
    opt.criterion = YoloLoss(opt.model.anchors, filter_fkt=filter_non_zero_gt)
    opt.num_workers = 0

    opt.log_path = './log/yolo_lstm'


    train(opt)
