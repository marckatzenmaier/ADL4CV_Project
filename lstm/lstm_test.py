import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tensorboardX import SummaryWriter

from lstm.LSTMModels import SequenceClassifier
from dataset_utils.datasets.MotBBSequence import MotBBSequence
from yolo.loss import YoloLoss
from lstm.LSTMLosses import NaiveLoss


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
saving_path = "SavedModels/" + name

num_epochs = 100
learn_rate = 1e-4

dataset = MotBBSequence('../dataset_utils/Mot17_test_single.txt')
train_data = Subset(dataset, range(0, dataset.valid_begin))
valid_data = Subset(dataset, range(dataset.valid_begin, len(dataset)))

obs_length = 10
pred_length = 9


loss_params = {"grid_shape": (16, 16),
               "image_shape": (416, 416),
               "path_anchors": "../dataset_utils/anchors/anchors5.txt"}
loss_function = NaiveLoss(loss_params)
model = SequenceClassifier([16, 16, loss_function.num_anchors, 4], [16, 16, loss_function.num_anchors, 4], 16)
optimizer = optim.Adam(model.parameters(), lr=learn_rate)

# test to yolo
#batch_in = np.array([dataset[0][0][0], dataset[0][0][0]])
#batch_tar = np.array([dataset[0][1][0], dataset[0][1][0]])
#loss_function.to_yolo(batch_in, batch_tar)




# possible use of lstm and dataset (probably not)
niter = 0
for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=1)
    for batch in train_loader:
        model.zero_grad()
        optimizer.zero_grad()

        model.hidden = model.init_hidden(batch[0].shape[0])
        model.cell   = model.init_hidden(batch[0].shape[0])

        # bb_sequence: [batch_size, seq_index, grid_h, grid_w, anchors, (w,h,c1,c2,conf)]
        # bb_sequence: first try [batch_size, seq_index, (id bb)] start with 100 possible bb per image (rest are zeros)
        bb_sequence = batch[0]
        bb_target_sequence = batch[1]

        loss = 0
        for i in range(obs_length):
            yolo_in, yolo_target = loss_function.to_yolo(bb_sequence[:, i].numpy(), bb_target_sequence[:, i].numpy())
            pred = model(torch.from_numpy(yolo_in[:, :, :, :, 1:]).float())
            loss += loss_function.forward(pred, yolo_in, yolo_target)

        #prev_out, _ = loss_function.shift(yolo_in, pred.detach())  # todo without teacher forcing
        for i in range(pred_length):
            #pred = model(prev_out)
            #prev_out = prev_out.detach() + pred.detach()
            yolo_in, yolo_target = loss_function.to_yolo(bb_sequence[:, obs_length + i].numpy(), bb_target_sequence[:, obs_length + i].numpy())
            pred = model(torch.from_numpy(yolo_in[:, :, :, :, 1:]).float())
            loss += loss_function.forward(pred, yolo_in, yolo_target)

        #source_sequence = torch.Tensor(np.arange(1, 21)).reshape(-1, 1)
        #target_sequence = torch.Tensor(np.arange(2, 22))

        #prev_out = source_sequence[0]
        #generated_seq = []
        #for i in range(19):
        #    output = model(source_sequence[i])
        #    prev_out = output.detach()
        #    generated_seq.append(prev_out.data.numpy().squeeze())
        #    loss += loss_function(output.view(-1), target_sequence[i])

        loss.backward()
        optimizer.step()

        # logging and stats
        writer.add_scalar('train_loss', loss, niter)
        print("{}: loss: {}".format(niter, loss.detach()))
        niter += 1
#torch.save(model.state_dict(), saving_path)
#writer.export_scalars_to_json("test.json")
writer.close()
