"""
@author Marc Katzenmaier
This script trains the YOLO LSTM network
"""
from dataset_utils.datasets.MotBBImageSequence import *
from torch.utils.data import DataLoader, Subset
from lstm.LSTMLosses import center_distance, displacement_error, prediction_to_box_list, draw_pred_sequence, NaiveLoss
from Flow import preprocess
from yolo.yolo_LSTM import BigYoloFlowLSTM, ConvLSTM
from yolo.loss import YoloLoss
from train_yolo import Opt
from tensorboardX import SummaryWriter
import shutil
from yolo.yolo_utils import filter_non_zero_gt, get_ap, logits_to_box_params
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

    dataset = MotBBImageSequence('dataset_utils/Mot17_test_single.txt', use_only_first_video=False, new_width=832,
                                 new_height=832)
    train_data = Subset(dataset, range(0, dataset.valid_begin))
    valid_data = Subset(dataset, range(dataset.valid_begin, len(dataset)))
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              drop_last=True)

    obs_length = 10
    pred_length = 9

    epoch_len = len(train_loader)

    for epoch in range(opt.num_epoches):
        for img_num, (gt, b, img1, img2) in enumerate(train_loader):
            opt.model.train()
            opt.decoder.train()
            opt.model.lstm_part.reinit_lstm(opt.batch_size)
            opt.decoder.reset_hidden(opt.batch_size)

            seq_loss = 0
            seq_ap = 0
            seq_loss_coord, seq_loss_conf, seq_loss_pred = (0.0, 0.0, 0.0)
            pred_sequence = []
            for i in range(obs_length):
                single_gt = b[:, i].to(opt.device())
                single_img2 = img2[:, i].to(opt.device())
                single_img2_normed = preprocess(img2[:, i]).to(opt.device())
                single_img1_normed = preprocess(img1[:, i]).to(opt.device())

                double_image = torch.cat((single_img1_normed, single_img2_normed), dim=1)

                logits = opt.model((double_image, single_img2 * 255.))

                loss, loss_coord, loss_conf = opt.yolo_loss(logits, single_gt)
                seq_ap += get_ap(logits, filter_gt_batch(single_gt), opt.image_size, opt.image_size, opt.model.anchors)
                seq_loss += loss
                seq_loss_conf += loss_conf.item()
                seq_loss_coord += loss_coord.item()

            prev_out = logits_to_box_params(logits.detach(), opt.model.anchors)
            # the box parameters are normalized to [0, 1]
            # at the moment [batch, anchors, ...., h* w]
            # rearrange tensor s.t [batch, h, w, anchors, ...]
            # ... == x, y, w, h, conf
            prev_out = prev_out.view(opt.batch_size, len(opt.model.anchors), -1, opt.encoding_size, opt.encoding_size) \
                .permute(0, 3, 4, 1, 2)
            # be carefull the position of the conf/id in the targets is 0 not 4
            # so we change it at this point to be consistent with the labels

            prev_out = torch.Tensor(np.roll(prev_out.cpu().numpy(), 1, axis=-1)).to(opt.device())

            opt.decoder.set_hidden(opt.model.lstm_part.hidden, opt.model.lstm_part.cell)

            for i in range(pred_length):
                _, yolo_target = opt.pred_loss.to_yolo(input=gt[:, obs_length].numpy(),
                                                       target=b[:, obs_length + i].numpy(), use_iou=True)
                yolo_target = torch.Tensor(yolo_target).to(opt.device())
                # target boxes are in [0, grid_h] -> normalize to 1
                # at this point i assume that the image is a square !!!
                yolo_target[:, :, :, :, 1:] = yolo_target[:, :, :, :, 1:] / opt.encoding_size

                input_tensor = prev_out[:, :, :, :, 1:].contiguous() \
                                                       .view(opt.batch_size, opt.encoding_size, opt.encoding_size,
                                                             len(opt.model.anchors) * 4) \
                                                       .permute(0, 3, 1, 2)
                pred = opt.decoder(input_tensor)
                pred = pred.view(opt.batch_size, len(opt.model.anchors), -1, opt.encoding_size, opt.encoding_size) \
                    .permute(0, 3, 4, 1, 2)

                seq_loss_pred += opt.pred_loss.forward(pred, prev_out, yolo_target)
                pred_sequence.append((prev_out.detach().cpu().numpy(),
                                      pred.detach().cpu().numpy(),
                                      yolo_target.detach().cpu().numpy()))

                prev_out[:, :, :, :, 1:] = yolo_target[:, :, :, :, 1:]

            seq_loss += seq_loss_pred
            loss = seq_loss
            loss.backward()

            opt.optimizer_encoder.step()
            opt.optimizer_decoder.step()

            opt.optimizer_encoder.zero_grad()
            opt.optimizer_decoder.zero_grad()

            seq_loss = seq_loss.item() - seq_loss_pred.item()

            writeLossToSummary(writer, 'Train', seq_loss / obs_length,
                               seq_loss_coord / obs_length,
                               seq_loss_conf / obs_length,
                               epoch * epoch_len + img_num)

            print(f"epoch:{epoch} it: {img_num}")
            print(f"loss_seq: {seq_loss/obs_length}, "
                  f"loss_coord: {seq_loss_coord / obs_length}, "
                  f"loss_conf: {seq_loss_conf / obs_length}, "
                  f"mAP: {seq_ap/obs_length}")

            seq_loss_pred = seq_loss_pred.item()
            box_list = prediction_to_box_list(pred_sequence)
            dis_error = displacement_error(box_list, center_distance, image_size=832.0)
            writer.add_scalar('Train/loss_pred', seq_loss_pred / pred_length, epoch * epoch_len + img_num)
            writer.add_scalar('Train/dis_err', dis_error, epoch * epoch_len + img_num)
            writer.add_scalar('Train/AP', seq_ap / obs_length, epoch * epoch_len + img_num)
            print(f"loss_pred: {seq_loss_pred / pred_length}, dis_error: {dis_error}")

        # draws last batch
        draw_pred_sequence(box_list, img2[0], pred_length, obs_length, name='train_img.png', image_size=832)

        ###############
        # VALIDATION
        ###############
        print("###############")
        print("# VALIDATION BEGIN")
        print("###############")
        opt.model.eval()
        opt.decoder.eval()
        valid_len = len(valid_loader)
        loss_val = 0
        loss_ap = 0
        loss_coord_val = 0
        loss_conf_val = 0
        loss_pred_val = 0
        dis_error_val = 0
        for img_num, (gt, b, img1, img2) in enumerate(valid_loader):
            opt.model.lstm_part.reinit_lstm(opt.batch_size)
            opt.decoder.reset_hidden(opt.batch_size)
            opt.optimizer_encoder.zero_grad()
            opt.optimizer_decoder.zero_grad()

            seq_loss = 0
            seq_loss_coord, seq_loss_conf, seq_loss_pred = (0.0, 0.0, 0.0)
            seq_ap = 0
            pred_sequence = []
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

            with torch.no_grad():
                prev_out = logits_to_box_params(logits.detach(), opt.model.anchors)
                # the box parameters are normalized to [0, 1]
                # at the moment [batch, anchors, ...., h* w]
                # rearrange tensor s.t [batch, h, w, anchors, ...]
                # ... == x, y, w, h, conf
                # print(f"origin mask {torch.sum(mask, dim=(0, 1))}")
                prev_out = prev_out.view(opt.batch_size, len(opt.model.anchors), -1, opt.encoding_size,
                                         opt.encoding_size) \
                    .permute(0, 3, 4, 1, 2)
                # be carefull the position of the conf/id in the targets is 0 not 4
                # so we change it at this point to be consistent with the labels

                prev_out = torch.Tensor(np.roll(prev_out.cpu().numpy(), 1, axis=-1)).to(opt.device())

                opt.decoder.set_hidden(opt.model.lstm_part.hidden, opt.model.lstm_part.cell)
                for i in range(pred_length):
                    _, yolo_target = opt.pred_loss.to_yolo(input=gt[:, obs_length].numpy(),
                                                           target=b[:, obs_length + i].numpy(), use_iou=True)
                    yolo_target = torch.Tensor(yolo_target).to(opt.device())
                    # target boxes are in [0, grid_h] -> normalize to 1
                    # at this point i assume that the image is a square !!!
                    yolo_target[:, :, :, :, 1:] = yolo_target[:, :, :, :, 1:] / opt.encoding_size
                    input_tensor = prev_out[:, :, :, :, 1:].contiguous() \
                                                           .view(opt.batch_size, opt.encoding_size, opt.encoding_size,
                                                                 len(opt.model.anchors) * 4) \
                                                           .permute(0, 3, 1, 2)
                    pred = opt.decoder(input_tensor)
                    pred = pred.view(opt.batch_size, len(opt.model.anchors), -1, opt.encoding_size, opt.encoding_size) \
                        .permute(0, 3, 4, 1, 2)

                    seq_loss_pred += opt.pred_loss.forward(pred, prev_out, yolo_target)
                    pred_sequence.append((prev_out.detach().cpu().numpy(),
                                          pred.detach().cpu().numpy(),
                                          yolo_target.detach().cpu().numpy()))
                    prev_out[:, :, :, :, 1:] += pred

                seq_loss_pred = seq_loss_pred.item()
                box_list = prediction_to_box_list(pred_sequence)
                dis_error = displacement_error(box_list, center_distance, image_size=832.0)

            loss_val += seq_loss.item() / valid_len / obs_length
            loss_coord_val += seq_loss_coord / valid_len / obs_length
            loss_conf_val += seq_loss_conf / valid_len / obs_length
            loss_ap += seq_ap / obs_length / valid_len

            loss_pred_val += seq_loss_pred / valid_len / pred_length
            dis_error_val += dis_error / valid_len
        draw_pred_sequence(box_list, img2[0], pred_length, obs_length, name=f'val_img.png', image_size=832)
        writeLossToSummary(writer, 'Val', loss_val, loss_coord_val,
                           loss_conf_val, (epoch + 1) * epoch_len)

        print(f"epoch:{epoch}")
        print(f"loss_seq: {loss_val}, loss_coord: {loss_coord_val}, loss_conf: {loss_conf_val}, mAP: {loss_ap}")

        writer.add_scalar('Val/loss_pred', loss_pred_val, (epoch + 1) * epoch_len)
        writer.add_scalar('Val/dis_err', dis_error_val, (epoch + 1) * epoch_len)
        writer.add_scalar('Val/mAP', loss_ap, (epoch + 1) * epoch_len)

        print(f"loss_pred: {loss_pred_val}, dis_error: {dis_error_val}")
        print("###############")
        print("# VALIDATION END")
        print("###############")

        torch.save({'epoch': epoch, 'model_state_dict': opt.model.state_dict(),
                    'optimizer_state_dict': opt.optimizer_encoder.state_dict()},
                   opt.log_path + f'/snapshot_encoder{epoch}.tar')
        torch.save({'epoch': epoch, 'model_state_dict': opt.decoder.state_dict(),
                    'optimizer_state_dict': opt.optimizer_decoder.state_dict()},
                   opt.log_path + f'/snapshot_decoder{epoch}.tar')

    writer.close()


if __name__ == "__main__":
    opt = Opt()
    opt.num_epoches = 10
    opt.image_size = 832
    opt.encoding_size = 26
    opt.useCuda = True
    opt.learning_rate = 1e-5
    opt.decay = 1e-5
    opt.batch_size = 1
    opt.num_workers = 4
    anchors = [(0.43, 1.715), (0.745625, 3.645), (1.24375, 5.9325), (2.5, 12.24), (6.1225, 22.41375)]
    opt.model = BigYoloFlowLSTM(opt.batch_size, anchors=anchors)
    opt.decoder = ConvLSTM((opt.encoding_size, opt.encoding_size), 5 * 4, 1024, opt.batch_size)

    check_point_encoder = torch.load('./models/yolo_lstm_flow_832.tar', map_location=opt.device())
    opt.model.load_state_dict(check_point_encoder['model_state_dict'], strict=True)

    # all parameters are on the device the hidden states are not parameters
    opt.model.to(opt.device())
    opt.decoder.to(opt.device())
    opt.yolo_loss = YoloLoss(opt.model.anchors, filter_fkt=filter_non_zero_gt)
    loss_params = {"grid_shape": (opt.encoding_size, opt.encoding_size),
                   "image_shape": (opt.image_size, opt.image_size),
                   "path_anchors": "dataset_utils/anchors/anchors5.txt"}
    opt.pred_loss = NaiveLoss(loss_params)
    opt.pred_loss.anchors *= 2.0  # the anchors from the txt are for 416 by 416 images

    opt.optimizer_encoder = torch.optim.Adam(opt.model.parameters(), lr=opt.learning_rate * 0.1,
                                             betas=(opt.momentum, 0.999),
                                             weight_decay=opt.decay)
    opt.optimizer_decoder = torch.optim.Adam(opt.decoder.parameters(), lr=1e-4, betas=(opt.momentum, 0.999),
                                             weight_decay=opt.decay)

    train(opt)
