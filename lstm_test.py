import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tensorboardX import SummaryWriter

from lstm.LSTMModels import SequenceClassifier
from dataset_utils.datasets.MotBBSequence import MotBBSequence
from yolo.loss import YoloLoss
from lstm.LSTMLosses import *


# todo format of yolov2 label
# todo network
# structure: input-> feature trans.-> lstm->output
# todo train loop
# todo load data



# a sequence is the outputs of the yolo network
# yolov2 loss image split in gridcells predict for each gridcell 5 boxes
# each box has 4 parameter plus 20 classes plus 1 confidence score

writer = SummaryWriter()
name = "aölsdjfalök"
saving_path = "models/" + name

num_epochs = 100
learn_rate = 1e-4

dataset = MotBBSequence('dataset_utils/Mot17_test_single.txt')
train_data = Subset(dataset, range(0, dataset.valid_begin))
valid_data = Subset(dataset, range(dataset.valid_begin, len(dataset)))

obs_length = 10
pred_length = 9


loss_params = {"grid_shape": (16, 16),
               "image_shape": (416, 416),
               "path_anchors": "dataset_utils/anchors/anchors5.txt"}
loss_function = NaiveLoss(loss_params)
model = SequenceClassifier([16, 16, loss_function.num_anchors, 4], [16, 16, loss_function.num_anchors, 4], 16)
optimizer = optim.Adam(model.parameters(), lr=learn_rate)

"""
##################################################################
batch_in = np.expand_dims(train_data[0][0][0], axis=0)
batch_out = np.expand_dims(train_data[0][1][0], axis=0)
yolo_in, yolo_target = loss_function.to_yolo(batch_in, batch_out)
pred = yolo_target[:, :, :, :, 1:] - yolo_in[:, :, :, :, 1:]

pred_sequence = [(yolo_in, pred, yolo_target)]
box_list = prediction_to_box_list(pred_sequence)
for frame in box_list:
    input, pred, target = frame
    for batch_id in range(len(input)):
        for box, box_id in enumerate(batch_out[batch_id, :, 0]):
            target_box = np.where(pred[batch_id, :, 0] == box_id)
            target_box = pred[batch_id, target_box, 1:] * 416.0 / 16.0
            print(np.sum(target_box - batch_out[batch_id, box, 1:]))
###################################################################
"""


niter = 0
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=True, num_workers=1)

for epoch in range(100):

    model.train()
    for batch in train_loader:
        model.zero_grad()
        optimizer.zero_grad()

        model.hidden = model.init_hidden(batch[0].shape[0])
        model.cell   = model.init_hidden(batch[0].shape[0])

        bb_sequence = batch[0]
        bb_target_sequence = batch[1]

        loss = 0
        for i in range(obs_length):
            yolo_in, yolo_target = loss_function.to_yolo(bb_sequence[:, i].numpy(), bb_target_sequence[:, i].numpy())
            pred = model(torch.from_numpy(yolo_in[:, :, :, :, 1:]).float())
            loss += loss_function.forward(pred, yolo_in, yolo_target)

        #prev_out, _ = loss_function.shift(yolo_in, pred.detach())  # todo without teacher forcing
        pred_sequence = []
        for i in range(pred_length):
            #pred = model(prev_out)
            #prev_out = prev_out.detach() + pred.detach()
            yolo_in, yolo_target = loss_function.to_yolo(bb_sequence[:, obs_length + i].numpy(), bb_target_sequence[:, obs_length + i].numpy())
            pred = model(torch.from_numpy(yolo_in[:, :, :, :, 1:]).float())
            loss += loss_function.forward(pred, yolo_in, yolo_target)
            pred_sequence.append((yolo_in, pred.detach().numpy(), yolo_target))

        loss.backward()
        optimizer.step()

        box_list = prediction_to_box_list(pred_sequence)
        dis_error = displacement_error(box_list, center_distance)

        # logging and stats
        writer.add_scalar('train_loss', loss, niter)
        print("{}: loss: {}, diss_error: {}".format(niter, loss.detach(), dis_error))
        niter += 1

    # validation
    model.eval()
    print("---------- ")
    print("VALIDATION")
    print("---------- ")
    loss = []
    dis_error = []
    for batch in valid_loader:

        model.hidden = model.init_hidden(batch[0].shape[0])
        model.cell   = model.init_hidden(batch[0].shape[0])

        bb_sequence = batch[0]
        bb_target_sequence = batch[1]

        for i in range(obs_length):
            yolo_in, yolo_target = loss_function.to_yolo(bb_sequence[:, i].numpy(), bb_target_sequence[:, i].numpy())
            pred = model(torch.from_numpy(yolo_in[:, :, :, :, 1:]).float())
            # loss += loss_function.forward(pred, yolo_in, yolo_target)

        #prev_out, _ = loss_function.shift(yolo_in, pred.detach())  # todo without teacher forcing
        pred_sequence = []
        local_loss = 0
        for i in range(pred_length):
            #pred = model(prev_out)
            #prev_out = prev_out.detach() + pred.detach()
            yolo_in, yolo_target = loss_function.to_yolo(bb_sequence[:, obs_length + i].numpy(), bb_target_sequence[:, obs_length + i].numpy())
            pred = model(torch.from_numpy(yolo_in[:, :, :, :, 1:]).float())
            local_loss += loss_function.forward(pred, yolo_in, yolo_target).detach()
            pred_sequence.append((yolo_in, pred.detach().numpy(), yolo_target))
        loss.append(local_loss)
        box_list = prediction_to_box_list(pred_sequence)
        dis_error.append(displacement_error(box_list, center_distance))

        # logging and stats
    loss = np.mean(loss)
    dis_error = np.mean(dis_error)
    writer.add_scalar('valid_loss', loss, niter)
    writer.add_scalar('dis_error', dis_error, niter)
    print("{}: loss: {}, dis_error: {}".format(niter, loss, dis_error))
    print("---------- ")

torch.save(model.state_dict(), saving_path)
writer.export_scalars_to_json("test.json")
writer.close()

