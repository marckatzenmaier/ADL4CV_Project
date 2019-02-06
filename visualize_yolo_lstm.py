"""
@author Marc Katzenmaier
Takes an snapshot specified with snapshot_path and a image folder with sorted images and then produces an .avi video
with the results of the model
"""
from dataset_utils import MOT_utils as motu
from yolo.yolo_LSTM import YoloLSTM
from yolo.yolo_utils import *
from yolo.yolo_utils import draw_boxes_opencv
from Object_Detection_Metrics.lib_s.Evaluator import *
from Object_Detection_Metrics.lib_s.utils import *

path = 'dataset_utils/datasets/MOT17/MOT17-10-SDP/'
snapshot_path = './models/yolo_lstm_832.tar'

log_path = './log/yolo_lstm_visualize/'
if not os.path.isdir(log_path):
    os.makedirs(log_path)

batch_size = 1   # only works with batchsize 1
cuda = False  # careful can take up much gpu memory
save_every_image = False  # if true all images are additionaly saved as .png
anchors = [(0.43, 1.715), (0.745625, 3.645), (1.24375, 5.9325), (2.5, 12.24), (6.1225, 22.41375)]
net = YoloLSTM(batch_size, image_size=832, anchors=anchors)
net.load_snapshot(snapshot_path)
net.eval()
if cuda:
    net.cuda()
net.reinit_lstm(batch_size)
imgs = []
_, img_paths, _ = motu.get_gt_img_inf(path)
a = 0
with torch.no_grad():
    for img_path in img_paths[int(len(img_paths) * .8):]:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (net.image_size, net.image_size))
        img = image.copy()
        image = torch.from_numpy(np.transpose(image.astype(np.float32), (2, 0, 1))).unsqueeze(0)
        if cuda:
            image = image.cuda()
        logits = net(image)
        if cuda:
            logits = logits.cpu()
        boxes = post_processing(logits, net.image_size, net.anchors, 0.25, 0.5)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if len(boxes) != 0:
            img = draw_boxes_opencv(img, boxes[0])
        img = cv2.resize(img, (net.image_size, net.image_size))
        imgs.append(img)
        if save_every_image:
            cv2.imwrite(log_path+f'{a}.png', img)
            a += 1

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(log_path+'video.avi', fourcc, 20.0, (net.image_size, net.image_size))
for i in imgs:
    out.write(i)
out.release()
