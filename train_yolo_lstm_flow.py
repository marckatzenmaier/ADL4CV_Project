from dataset_utils.datasets.MotBBImageSequence import *
from torch.utils.data import DataLoader, Subset
from yolo.yolo_LSTM import YoloFlowLSTM
from yolo.loss import YoloLoss
from train_yolo_example import Opt
import torch
from Flow.FlowNet import preprocess
from tensorboardX import SummaryWriter
import shutil
from train_yolo_example import writeLossToSummary


def filter_gt(gt):
    # used to filter only existing bb for loss
    return gt[gt[:, 0] != 0.0]


def load_Snapshot_to_yolo_LSTM(model, opt):
    def remove_flownet_weights(dict):
        del dict["deconv4.0.weight"]
        del dict["deconv3.0.weight"]
        del dict["deconv2.0.weight"]

        del dict["predict_flow5.weight"]
        del dict["predict_flow4.weight"]
        del dict["predict_flow3.weight"]
        del dict["predict_flow2.weight"]

        del dict["upsampled_flow5_to_4.weight"]
        del dict["upsampled_flow4_to_3.weight"]
        del dict["upsampled_flow3_to_2.weight"]
        return dict

    if torch.cuda.is_available() and opt.useCuda:
        device = torch.device("cuda")
        model_state_dict_yolo = torch.load(opt.pre_trained_yolo_path, map_location="cuda:0")['model_state_dict']
        model_state_dict_flow = torch.load(opt.pre_trained_flownet_path, map_location="cuda:0")['state_dict']
    else:
        device = torch.device('cpu')
        model_state_dict_yolo = torch.load(opt.pre_trained_yolo_path, map_location='cpu')['model_state_dict']
        model_state_dict_flow = torch.load(opt.pre_trained_flownet_path, map_location='cpu')['state_dict']

    del model_state_dict_yolo["stage3_conv2.weight"]
    model_state_dict_flow = remove_flownet_weights(model_state_dict_flow)
    model.encoder.load_state_dict(model_state_dict_yolo, strict=True)
    model.flownet.load_state_dict(model_state_dict_flow, strict=True)
    model.to(device)


def train(opt):
    # setup train and eval set
    if torch.cuda.is_available() and opt.useCuda:
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    dataset = MotBBImageSequence('dataset_utils/Mot17_test_single.txt', use_only_first_video=False)
    train_data = Subset(dataset, range(0, dataset.valid_begin))
    valid_data = Subset(dataset, range(dataset.valid_begin, len(dataset)))
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=True)

    # log stuff
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    opt.optimizer = torch.optim.Adam(opt.model.parameters(), lr=opt.learning_rate, betas=(opt.momentum, 0.999), weight_decay=opt.decay)
    epoch_len = len(train_loader)
    print('epoch_len: {}'.format(epoch_len))

    for epoch in range(opt.num_epoches):
        print('num epoch: {:4d}'.format(epoch))
        opt.model.train()
        for img_num, (gt, b, img1, img2) in enumerate(train_loader):
            sequence_len = gt.shape[1]
            opt.model.reinit_lstm(opt.batch_size)
            opt.optimizer.zero_grad()
            seq_loss = 0
            seq_loss_coord, seq_loss_conf = (0.0, 0.0)
            for i in range(sequence_len): # todo geht es auf mit den sequenzen?
                if torch.cuda.is_available() and opt.useCuda:
                    single_gt = b[:, i].cuda()
                    single_img2 = img2[:, i].cuda()
                    single_img2_normed = preprocess(img2[:, i]).cuda()
                    single_img1_normed = preprocess(img1).cuda()
                else:
                    single_gt = b[:, i]
                    single_img2 = img2[:, i]
                    single_img2_normed = preprocess(img2[:, i])
                    single_img1_normed = preprocess(img1[:, i])

                double_image = torch.cat((single_img1_normed, single_img2_normed), dim=1)
                logits = opt.model((single_img2, double_image))
                loss, loss_coord, loss_conf = opt.criterion(logits, single_gt)
                seq_loss += loss
                seq_loss_conf += loss_conf.item()
                seq_loss_coord += loss_coord.item()
            seq_loss.backward()
            opt.optimizer.step()

            seq_loss = seq_loss.detach().numpy()
            print(f"epoch:{epoch} it: {img_num} loss: {seq_loss/sequence_len}")
            writeLossToSummary(writer, 'Train', seq_loss/sequence_len, seq_loss_coord/sequence_len,
                               seq_loss_conf/sequence_len, epoch * epoch_len + img_num)


        # eval stuff
        opt.model.eval()
        valid_len = len(valid_loader)
        loss_eval = 0
        loss_coord_eval = 0
        loss_conf_eval = 0
        for img_num, (gt, b, img1, img2) in enumerate(valid_loader):
            sequence_len = gt.shape[1]
            opt.model.reinit_lstm()
            seq_loss, seq_loss_coord, seq_loss_conf = (0.0, 0.0, 0.0)
            for i in range(sequence_len):
                if torch.cuda.is_available() and opt.useCuda:
                    single_gt = gt[:, i].cuda()
                    single_img = img1[:, i].cuda()
                else:
                    single_gt = gt[:, i]
                    single_img = img1[:, i]
                with torch.no_grad():
                    logits = opt.model(single_img)
                    loss, loss_coord, loss_conf = opt.criterion(logits, single_gt)
                seq_loss += loss.item()
                seq_loss_conf += loss_conf.item()
                seq_loss_coord += loss_coord.item()

            loss_eval += seq_loss/sequence_len
            loss_coord_eval += seq_loss_coord/sequence_len
            loss_conf_eval += seq_loss_conf/sequence_len


        writeLossToSummary(writer, 'Val', loss_eval/valid_len, loss_coord_eval/valid_len,
                           loss_conf_eval/valid_len, (epoch+1) * epoch_len)
        print('{}  {}   {}'.format(loss_eval/valid_len, loss_coord_eval/valid_len,
                                   loss_conf_eval/valid_len))

        torch.save({'epoch': epoch, 'model_state_dict': opt.model.state_dict(),
                    'optimizer_state_dict': opt.optimizer.state_dict()},
                   opt.log_path+'/snapshot{:04d}.tar'.format(epoch))

    writer.close()


if __name__ == "__main__":
    opt = Opt()
    opt.useCuda = False
    opt.learning_rate = 1e-5
    opt.batch_size = 1
    opt.model = YoloFlowLSTM(opt.batch_size)
    opt.pre_trained_yolo_path = 'models/snapshot0020.tar'

    load_Snapshot_to_yolo_LSTM(opt.model, opt)
    device = torch.device("cuda")
    if opt.useCuda and torch.cuda.is_available():
        opt.model.encoder.to(device)
        opt.model.lstm_part.to(device)
        opt.model
        opt.model.to(device)
    opt.criterion = YoloLoss(opt.model.anchors, filter_fkt=filter_gt)
    train(opt)