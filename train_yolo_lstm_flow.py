from dataset_utils.datasets.MotBBImageSequence import *
from torch.utils.data import DataLoader, Subset
from yolo.yolo_LSTM import YoloFlowLSTM
from yolo.loss import YoloLoss
from yolo.yolo_utils import logits_to_box_params
from train_yolo_example import Opt
import torch
from flow import preprocess
from tensorboardX import SummaryWriter
import shutil
from train_yolo_example import writeLossToSummary
from lstm.LSTMLosses import NaiveLoss, displacement_error, prediction_to_box_list, center_distance
from lstm.LSTMModels import ConvLSTM


def filter_gt(gt):
    # used to filter only existing bb for loss
    return gt[gt[:, 0] != 0.0]


def draw_pred_sequence(box_list, images, seq_length, obs_length):

    # draw the predicted path
    # first goal is to get an actual path
    image = images[obs_length]
    image = misc.imresize(image, (416, 416))
    for i in range(seq_length):
        boxes_in, boxes_pred, boxes_tar = box_list[i]
        boxes_pred = np.squeeze(boxes_pred) * 416.0
        boxes_tar = np.squeeze(boxes_tar) * 416.0

        # draw predicted boxes
        for box in range(len(boxes_pred)):
            if np.sum(boxes_pred[box, 1:]) != 0:

                x = max(min(boxes_pred[box, 1], 415), 0)
                y = max(min(boxes_pred[box, 2], 415), 0)

                image[int(y), int(x)] = [255, 0, 0]

        # draw target boxes
        for box in range(len(boxes_tar)):
            if np.sum(boxes_tar[box, 1:]) != 0:

                x = boxes_tar[box, 1]
                y = boxes_tar[box, 2]

                image[int(y), int(x)] = [0, 255, 0]

    misc.imsave('train_img.png', image)


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



    model_state_dict_yolo = torch.load(opt.pre_trained_yolo_path, map_location=opt.device())['model_state_dict']
    model_state_dict_flow = torch.load(opt.pre_trained_flownet_path, map_location=opt.device())['state_dict']

    del model_state_dict_yolo["stage3_conv2.weight"]
    model_state_dict_flow = remove_flownet_weights(model_state_dict_flow)
    model.encoder.load_state_dict(model_state_dict_yolo, strict=True)
    model.flownet.load_state_dict(model_state_dict_flow, strict=True)
    model.to(opt.device())


def train(opt):
    # setup train and eval set
    torch.cuda.manual_seed(123)
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

    decoder_model = ConvLSTM((opt.encoding_size, opt.encoding_size), 5 * 4, 1024, batch_size_init=1)
    decoder_model.to(opt.device())
    # todo two optimizers
    opt.optimizer = torch.optim.Adam(opt.model.parameters(), lr=opt.learning_rate, betas=(opt.momentum, 0.999),
                                     weight_decay=opt.decay)
    optimizer_decoder = torch.optim.Adam(decoder_model.parameters(), lr=opt.learning_rate, betas=(opt.momentum, 0.999),
                                     weight_decay=opt.decay)
    epoch_len = len(train_loader)
    print('epoch_len: {}'.format(epoch_len))

    obs_length = 10
    pred_length = 9
    loss_params = {"grid_shape": (13, 13),
                   "image_shape": (416, 416),
                   "path_anchors": "dataset_utils/anchors/anchors5.txt"}
    prediction_loss = NaiveLoss(loss_params)

    # todo include decoder model in opt
    for epoch in range(opt.num_epoches):
        print('num epoch: {:4d}'.format(epoch))
        opt.model.train()
        decoder_model.train()
        for img_num, (gt, b, img1, img2) in enumerate(train_loader):
            opt.model.reinit_lstm(opt.batch_size)
            decoder_model.reset_hidden(opt.batch_size)
            opt.optimizer.zero_grad()
            decoder_model.zero_grad()
            seq_loss = 0
            seq_loss_coord, seq_loss_conf = (0.0, 0.0)
            pred_sequence = []

            for i in range(obs_length):  # todo geht es auf mit den sequenzen?
                single_gt = b[:, i].to(opt.device())
                single_img2 = img2[:, i].to(opt.device())
                single_img2_normed = preprocess(img2[:, i]).to(opt.device())
                single_img1_normed = preprocess(img1[:, i]).to(opt.device())

                double_image = torch.cat((single_img1_normed, single_img2_normed), dim=1)
                logits = opt.model((single_img2, double_image))
                loss, loss_coord, loss_conf = opt.criterion(logits, single_gt)
                seq_loss += loss
                seq_loss_conf += loss_conf.item()
                seq_loss_coord += loss_coord.item()

            prev_out = logits_to_box_params(logits.detach(), opt.model.anchors)
            # the box parameters are normalized to [0, 1]
            # at the moment [batch, anchors, ...., h* w]
            # rearrange tensor s.t [batch, h, w, anchors, ...]
            # ... == x, y, w, h, conf

            prev_out = prev_out.view(opt.batch_size, len(opt.model.anchors), -1, opt.encoding_size, opt.encoding_size)\
                               .permute(0, 3, 4, 1, 2)
            # be carefull the position of the conf/id in the targets is 0 not 4
            # so we change it at this point to be consistent with the labels
            prev_out = torch.Tensor(np.roll(prev_out.cpu().numpy(), 1, axis=-1)).to(opt.device())
            mask = (prev_out[:, :, :, :, 0] > opt.conf_threshold).float().unsqueeze(-1)
            pred_loss = 0
            decoder_model.set_hidden(opt.model.lstm_part.hidden, opt.model.lstm_part.cell)
            for i in range(pred_length):
                _, yolo_target = prediction_loss.to_yolo(b[:, obs_length].numpy(), b[:, obs_length + i].numpy())
                yolo_target = torch.Tensor(yolo_target).to(opt.device())
                # target boxes are in [0, grid_h] -> normalize to 1
                # at this point i assume that the image is a square !!!
                yolo_target[:, :, :, :, 1:] = yolo_target[:, :, :, :, 1:] / opt.encoding_size
                input = prev_out[:, :, :, :, 1:].contiguous()\
                                                .view(opt.batch_size, len(opt.model.anchors)*4,
                                                      opt.encoding_size, opt.encoding_size)
                pred = decoder_model(input)
                pred = pred.view(opt.batch_size, len(opt.model.anchors), -1, opt.encoding_size, opt.encoding_size)\
                               .permute(0, 3, 4, 1, 2)
                pred_loss += prediction_loss.forward(pred, prev_out, mask*yolo_target)
                pred_sequence.append((prev_out.detach().cpu().numpy(),
                                      pred.detach().cpu().numpy(),
                                      (mask*yolo_target).detach().cpu().numpy()))
                prev_out[:, :, :, :, 1:] += pred

            # todo google collab
            seq_loss += pred_loss
            seq_loss.backward()
            opt.optimizer.step()
            optimizer_decoder.step()

            box_list = prediction_to_box_list(pred_sequence)
            dis_error = displacement_error(box_list, center_distance)
            if (epoch * epoch_len + img_num) % 1 == 0:
                draw_pred_sequence(box_list, img2[0], pred_length, obs_length)

            seq_loss = seq_loss.detach().numpy()
            pred_loss = pred_loss.detach().numpy()
            print(f"epoch:{epoch} it: {img_num}"
                  f" loss_seq: {seq_loss/obs_length}, loss_pred: {pred_loss}, dis_error{dis_error}")
            writeLossToSummary(writer, 'Train', seq_loss/obs_length, seq_loss_coord/obs_length,
                               seq_loss_conf/obs_length, epoch * epoch_len + img_num)
            writer.add_scalar('Train/pred_loss', pred_loss)


        # eval stuff
        opt.model.eval()
        decoder_model.eval()
        valid_len = len(valid_loader)
        loss_eval = 0
        loss_coord_eval = 0
        loss_conf_eval = 0
        loss_pred = 0
        dis_error = 0
        for img_num, (gt, b, img1, img2) in enumerate(valid_loader):
            sequence_len = gt.shape[1]
            opt.model.reinit_lstm(opt.batch_size)
            decoder_model.reset_hidden(opt.batch_size)
            seq_loss, seq_loss_coord, seq_loss_conf, seq_loss_pred = (0.0, 0.0, 0.0, 0.0)
            for i in range(obs_length):
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
                with torch.no_grad():
                    logits = opt.model((single_img2, double_image))
                    loss, loss_coord, loss_conf = opt.criterion(logits, single_gt)
                seq_loss += loss.item()
                seq_loss_conf += loss_conf.item()
                seq_loss_coord += loss_coord.item()

            loss_eval += seq_loss/obs_length
            loss_coord_eval += seq_loss_coord/obs_length
            loss_conf_eval += seq_loss_conf/obs_length

            prev_out = logits_to_box_params(logits.detach(), opt.model.anchors)
            prev_out = prev_out.view(opt.batch_size, len(opt.model.anchors), -1, opt.encoding_size, opt.encoding_size)\
                               .permute(0, 3, 4, 1, 2)
            prev_out = torch.Tensor(np.roll(prev_out.cpu().numpy(), 1, axis=-1)).to(opt.device())

            mask = (prev_out[:, :, :, :, 0] > opt.conf_threshold).float().unsqueeze(-1)
            decoder_model.set_hidden(opt.model.lstm_part.hidden, opt.model.lstm_part.cell)

            pred_sequence = []
            for i in range(pred_length):  # todo cuda
                _, yolo_target = prediction_loss.to_yolo(b[:, obs_length].numpy(), b[:, obs_length + i].numpy())
                yolo_target = torch.Tensor(yolo_target).to(opt.device())
                yolo_target[:, :, :, :, 1:] = yolo_target[:, :, :, :, 1:] / opt.encoding_size

                input = prev_out[:, :, :, :, 1:].contiguous()\
                                                .view(opt.batch_size, len(opt.model.anchors)*4,
                                                      opt.encoding_size, opt.encoding_size)
                pred = decoder_model(input)
                pred = pred.view(opt.batch_size, len(opt.model.anchors), -1, opt.encoding_size, opt.encoding_size)\
                               .permute(0, 3, 4, 1, 2)
                seq_loss_pred += prediction_loss.forward(pred, prev_out, mask*yolo_target)
                pred_sequence.append((prev_out.detach().cpu().numpy(),
                                      pred.detach().cpu().numpy(),
                                      (mask*yolo_target).detach().cpu().numpy()))
                prev_out[:, :, :, :, 1:] += pred

            loss_eval += seq_loss_pred / pred_length
            loss_pred += seq_loss_pred / pred_length
            box_list = prediction_to_box_list(pred_sequence)
            dis_error += displacement_error(box_list, center_distance)

        writeLossToSummary(writer, 'Val', loss_eval/valid_len, loss_coord_eval/valid_len,
                           loss_conf_eval/valid_len, (epoch+1) * epoch_len)
        writer.add_scalar('Val/loss_pred', loss_pred / valid_len)
        print(f"epoch:{epoch} it: {img_num}"
              f" loss_seq: {loss_eval/valid_len}, "
              f"loss_coord: {loss_coord_eval/valid_len}, loss_conf: {loss_conf_eval/valid_len}"
              f" loss_pred: {loss_pred/valid_len}, dis_error{dis_error/valid_len}")

        torch.save({'epoch': epoch, 'model_state_dict': opt.model.state_dict(),
                    'optimizer_state_dict': opt.optimizer.state_dict()},
                   opt.log_path+'/snapshot_encoder{:04d}.tar'.format(epoch))
        torch.save({'epoch': epoch, 'model_state_dict': decoder_model.state_dict(),
                    'optimizer_state_dict': optimizer_decoder.state_dict()},
                   opt.log_path+'/snapshot_encoder{:04d}.tar'.format(epoch))


    writer.close()


if __name__ == "__main__":
    opt = Opt()
    opt.useCuda = False
    opt.learning_rate = 1e-5
    opt.batch_size = 1
    opt.model = YoloFlowLSTM(opt.batch_size)
    opt.pre_trained_yolo_path = 'models/snapshot0020.tar'

    load_Snapshot_to_yolo_LSTM(opt.model, opt)
    opt.model.encoder.to(opt.device())
    opt.model.lstm_part.to(opt.device())
    opt.model.to(opt.device())
    opt.criterion = YoloLoss(opt.model.anchors, filter_fkt=filter_gt)
    train(opt)