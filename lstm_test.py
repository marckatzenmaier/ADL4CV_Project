import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tensorboardX import SummaryWriter

from lstm.LSTMModels import *
from dataset_utils.datasets.MotBBSequence import MotBBSequence
from lstm.LSTMLosses import *
import matplotlib.pyplot as plt
import scipy.misc as misc


# todo format of yolov2 label
# todo network
# structure: input-> feature trans.-> lstm->output
# todo train loop
# todo load data

def draw_pred_sequence(box_list, image_paths, seq_length, obs_length):

    frame_paths = image_paths
    for i in range(seq_length):
        boxes_in, boxes_pred, boxes_tar = box_list[i]
        boxes_pred = np.squeeze(boxes_pred) * 416.0 / 16
        boxes_tar = np.squeeze(boxes_tar) * 416.0 / 16

        image = misc.imread(frame_paths[i + obs_length])
        image = misc.imresize(image, (416, 416))

        # draw predicted boxes
        for box in range(len(boxes_pred)):
            if np.sum(boxes_pred[box, 1:]) != 0:
                width = int(boxes_pred[box, 3])
                height = int(boxes_pred[box, 4])

                top_left_x = max(min(boxes_pred[box, 1] - width / 2, 415), 0)
                top_left_y = max(min(boxes_pred[box, 2] - height / 2, 415), 0)

                image[int(top_left_y), int(top_left_x): min(int(top_left_x + width), 415)] = [255, 0, 0]
                image[min(int(top_left_y + height), 415), int(top_left_x): min(int(top_left_x + width), 415)] = [255, 0,
                                                                                                                 0]
                image[int(top_left_y):int(top_left_y + height), int(top_left_x)] = [255, 0, 0]
                image[int(top_left_y):min(int(top_left_y + height), 415), min(int(top_left_x + width), 415)] = [255, 0,
                                                                                                                0]
        # draw target boxes
        for box in range(len(boxes_tar)):
            if np.sum(boxes_tar[box, 1:]) != 0:
                width = int(boxes_tar[box, 3])
                height = int(boxes_tar[box, 4])

                top_left_x = boxes_tar[box, 1] - width / 2
                top_left_y = boxes_tar[box, 2] - height / 2

                image[int(top_left_y), int(top_left_x): int(top_left_x + width)] = [0, 255, 0]
                image[int(top_left_y + height), int(top_left_x): int(top_left_x + width)] = [0, 255, 0]
                image[int(top_left_y):int(top_left_y + height), int(top_left_x)] = [0, 255, 0]
                image[int(top_left_y):int(top_left_y + height), int(top_left_x + width)] = [0, 255, 0]

        plt.imshow(image)
        plt.show()


if __name__ == "__main__":
    writer = SummaryWriter()
    name = "aölsdjfalök"
    saving_path = "models/" + name

    num_epochs = 100
    learn_rate = 3e-5

    dataset = MotBBSequence('dataset_utils/Mot17_test_single.txt', use_only_first_video=False)
    train_data = Subset(dataset, range(0, dataset.valid_begin))
    valid_data = Subset(dataset, range(dataset.valid_begin, len(dataset)))

    obs_length = 10
    pred_length = 9

    loss_params = {"grid_shape": (16, 16),
                   "image_shape": (416, 416),
                   "path_anchors": "dataset_utils/anchors/anchors5.txt"}
    loss_function = NaiveLoss(loss_params)
    model = SequenceClassifier([16, 16, loss_function.num_anchors, 4], [16, 16, loss_function.num_anchors, 4], 16)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=0, )

    niter = 0
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=1)

    for epoch in range(num_epochs):

        model.train()
        for batch in train_loader:
            model.zero_grad()
            optimizer.zero_grad()

            model.reset_hidden(batch[0].shape[0])

            bb_sequence = batch[0].detach()
            bb_target_sequence = batch[1].detach()

            loss = 0
            bb_sequence_old = bb_sequence[:, 0].numpy()
            for i in range(obs_length):
                _, yolo_in = loss_function.to_yolo(bb_sequence_old, bb_sequence[:, i].numpy())
                _, yolo_target = loss_function.to_yolo(bb_sequence_old, bb_target_sequence[:, i].numpy())
                pred = model(Variable(torch.from_numpy(yolo_in[:, :, :, :, 1:]).float()))
                #loss += loss_function.forward(pred, yolo_in, yolo_target)

            prev_out = loss_function.shift(torch.from_numpy(yolo_in).float(), pred)  # todo without teacher forcing
            pred_sequence = []
            #loss_weights = np.square(np.linspace(-pred_length//2, pred_length//2)) / (pred_length**2/4)
            for i in range(pred_length):

                if niter >= 0:  # use your own prediction to calculate the next input from your own prediction
                    _, yolo_target = loss_function.to_yolo(bb_sequence_old, bb_target_sequence[:, obs_length + i].numpy())
                    prev_out_in = prev_out[:, :, :, :, 1:].contiguous()
                    pred = model(Variable(prev_out_in))
                    loss += loss_function.forward(pred, prev_out, yolo_target)  # * loss_weights[i]
                    pred_sequence.append((prev_out.detach().numpy(), pred.detach().numpy(), yolo_target))
                    prev_out = loss_function.shift(prev_out, pred)
                else:
                    _, yolo_target = loss_function.to_yolo(bb_sequence_old,
                                                           bb_target_sequence[:, obs_length + i].numpy())
                    _, yolo_in = loss_function.to_yolo(bb_sequence_old,
                                                           bb_sequence[:, obs_length + i].numpy())
                    pred = model(torch.from_numpy(yolo_in[:, :, :, :, 1:]).float())
                    loss += loss_function.forward(pred, yolo_in, yolo_target)
                    pred_sequence.append((yolo_in, pred.detach().numpy(), yolo_target))

            loss.backward()
            optimizer.step()

            box_list = prediction_to_box_list(pred_sequence)
            dis_error = displacement_error(box_list, center_distance)

            # logging and stats
            writer.add_scalar('train_loss', loss, niter)
            writer.add_scalar('train_dis_error', dis_error, niter)
            print("{}: loss: {}, diss_error: {}".format(niter, loss.detach(), dis_error))
            niter += 1

        # validation
        model.eval()
        print("---------- ")
        print("VALIDATION epoch {}".format(epoch))
        print("---------- ")
        loss = []
        dis_error = []
        for batch in valid_loader:

            model.reset_hidden(batch[0].shape[0])

            bb_sequence = batch[0]
            bb_target_sequence = batch[1]
            target_frames = batch[3]

            bb_sequence_old = bb_sequence[:, 0].numpy()
            local_loss = 0
            for i in range(obs_length):
                _, yolo_in = loss_function.to_yolo(bb_sequence_old, bb_sequence[:, i].numpy())
                _, yolo_target = loss_function.to_yolo(bb_sequence_old, bb_target_sequence[:, i].numpy())
                pred = model(torch.from_numpy(yolo_in[:, :, :, :, 1:]).float())
                #local_loss += loss_function.forward(pred, yolo_in, yolo_target)

            prev_out = loss_function.shift(torch.from_numpy(yolo_in).float(), pred)  # todo without teacher forcing
            pred_sequence = []

            for i in range(pred_length):
                _, yolo_target = loss_function.to_yolo(bb_sequence_old, bb_target_sequence[:, obs_length + i].numpy())
                prev_out_in = prev_out[:, :, :, :, 1:].contiguous()
                pred = model(prev_out_in)
                local_loss += loss_function.forward(pred, prev_out, yolo_target)
                pred_sequence.append((prev_out.detach().numpy(), pred.detach().numpy(), yolo_target))
                prev_out = loss_function.shift(prev_out, pred)
            loss.append(local_loss.detach().numpy())
            box_list = prediction_to_box_list(pred_sequence, False)
            dis_error.append(displacement_error(box_list, center_distance))
            if epoch % 80 == 0 and epoch != 0:
                draw_pred_sequence(box_list, list(map(lambda x: x[0], target_frames)), pred_length, obs_length)

            # logging and stats
        loss = np.mean(loss)
        dis_error = np.mean(dis_error)
        writer.add_scalar('valid_loss', loss, niter)
        writer.add_scalar('valid_dis_error', dis_error, niter)
        print("{}: loss: {}, dis_error: {}".format(niter, loss, dis_error))
        print("---------- ")

    torch.save(model.state_dict(), saving_path)
    writer.export_scalars_to_json("test.json")
    writer.close()

