"""
@author Nikita Kister
Takes the trained model for detection and prediction and evaluates it on the validation data.
In addition, for every sequence an image with the real and predicted path is written in the specified location.
"""
from dataset_utils.datasets.MotBBImageSequence import *
from torch.utils.data import DataLoader, Subset
from lstm.LSTMLosses import center_distance, displacement_error, prediction_to_box_list, draw_pred_sequence, NaiveLoss
from lstm.LSTMLosses import mean_squared_trajectory_error, mean_iou
from Flow import preprocess
from yolo.yolo_LSTM import BigYoloFlowLSTM, ConvLSTM
from yolo.loss import YoloLoss
from train_yolo import Opt
import shutil
from yolo.yolo_utils import get_ap, logits_to_box_params, filter_non_zero_gt


def filter_gt_batch(gt):
    # used to filter only existing bb for loss
    t = gt[gt[:, :, 0] != 0.0]

    return t[:, 1:]


def test_final(opt):
    dataset = MotBBImageSequence('dataset_utils/Mot17_test_single.txt', use_only_first_video=False, new_width=832,
                                 new_height=832)
    valid_data = Subset(dataset, range(dataset.valid_begin, len(dataset)))
    valid_loader = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              drop_last=True)

    obs_length = 10
    pred_length = 9

    ###############
    # VALIDATION
    ###############
    print("###############")
    print("# VALIDATION BEGIN")
    print("###############")
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    opt.model.eval()
    opt.decoder.eval()
    valid_len = len(valid_loader)
    loss_val = 0
    loss_ap = 0
    loss_coord_val = 0
    loss_conf_val = 0
    loss_pred_val = 0
    dis_error_val = 0
    traj_error_val = 0
    iou_val = 0
    for img_num, (gt, b, img1, img2) in enumerate(valid_loader):
        opt.model.lstm_part.reinit_lstm(opt.batch_size)
        opt.decoder.reset_hidden(opt.batch_size)

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
                prev_out[:, :, :, :, 1:] += pred

            seq_loss_pred = seq_loss_pred.item()
            box_list = prediction_to_box_list(pred_sequence)
            dis_error = displacement_error(box_list, center_distance, image_size=832.0)
            traj_error = mean_squared_trajectory_error(box_list, center_distance, image_size=832.0)
            iou = mean_iou(box_list, image_size=832.0)

        loss_val += seq_loss.item() / valid_len / obs_length
        loss_coord_val += seq_loss_coord / valid_len / obs_length
        loss_conf_val += seq_loss_conf / valid_len / obs_length
        loss_ap += seq_ap / obs_length / valid_len

        loss_pred_val += seq_loss_pred / valid_len / pred_length
        dis_error_val += dis_error / valid_len
        traj_error_val += traj_error / valid_len
        iou_val += iou / valid_len
        draw_pred_sequence(box_list, img2[0], pred_length, obs_length,
                           name=f'{opt.log_path}/val_img{img_num}.png', image_size=832)

    print(f"loss_seq: {loss_val}, loss_coord: {loss_coord_val}, loss_conf: {loss_conf_val}, mAP: {loss_ap}")
    print(f"loss_pred: {loss_pred_val}, dis_error: {dis_error_val}, traj_error: {traj_error_val}, iou_val: {iou_val}")
    print("###############")
    print("# VALIDATION END")
    print("###############")


if __name__ == "__main__":
    opt = Opt()
    opt.num_epoches = 100
    opt.image_size = 832
    opt.encoding_size = 26
    opt.useCuda = True
    opt.batch_size = 1
    opt.num_workers = 4
    opt.log_path = "./log/pred_imgs"
    anchors = [(0.43, 1.715), (0.745625, 3.645), (1.24375, 5.9325), (2.5, 12.24), (6.1225, 22.41375)]
    opt.model = BigYoloFlowLSTM(opt.batch_size, anchors=anchors)
    opt.decoder = ConvLSTM((opt.encoding_size, opt.encoding_size), 5 * 4, 1024, opt.batch_size)

    check_point_encoder = torch.load('./models/yolo_lstm_flow_832_encoder_final.tar', map_location=opt.device())
    check_point_decoder = torch.load('./models/yolo_lstm_flow_832_decoder_final.tar', map_location=opt.device())

    opt.model.load_state_dict(check_point_encoder['model_state_dict'], strict=True)
    opt.decoder.load_state_dict(check_point_decoder['model_state_dict'], strict=True)

    # all parameters are on the device the hidden states are not parameters
    opt.model.to(opt.device())
    opt.decoder.to(opt.device())
    opt.yolo_loss = YoloLoss(opt.model.anchors, filter_fkt=filter_non_zero_gt)
    loss_params = {"grid_shape": (opt.encoding_size, opt.encoding_size),
                   "image_shape": (opt.image_size, opt.image_size),
                   "path_anchors": "dataset_utils/anchors/anchors5.txt"}
    opt.pred_loss = NaiveLoss(loss_params)
    opt.pred_loss.anchors *= 2.0

    test_final(opt)
