from flow import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import  imread, imresize


def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)

# taken from https://github.com/ClementPinard/FlowNetPytorch (gifs, weights ...)
# used example gifs from this github and split them to generate the images
net = FlowNetSBigEncoder(False)
model_path = "../models/flownets_EPE1.951.pth.tar"
net.load_state_dict(torch.load(model_path)['state_dict'], strict=True)

# load images
img_1 = imresize(imread("000007.jpg"), [832, 832]).astype(np.float32)
# flownet expects bgr -> roll over the last axis (channel) and rearrange for pytorch
img_1 = np.roll(img_1, 1, axis=-1).transpose((2, 0, 1))
img_1 /= 255.0
img_1 += -0.5

img_2 = imresize(imread("000008.jpg"), [832, 832]).astype(np.float32)
img_2 = np.roll(img_2, 1, axis=-1).transpose((2, 0, 1))
img_2 /= 255.0
img_2 += -0.5

batch = torch.from_numpy(np.vstack([img_1, img_2])).unsqueeze(0)
flow = net(batch)
flow = flow2rgb(flow[0], None)
plt.imshow(flow.transpose(1, 2, 0))
plt.show()
# load weights