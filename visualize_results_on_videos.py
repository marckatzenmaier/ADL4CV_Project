"""
@author Marc Katzenmaier
Takes an snapshot specified with snapshot_path and a image folder with sorted images and then produces an .avi video
with the results of the model
"""
from dataset_utils import MOT_utils as motu
# from yolo.yolo_LSTM import YoloLSTM
from yolo.yolo_net import Yolo
from yolo.yolo_utils import *
from yolo.yolo_utils import draw_boxes_opencv
from Object_Detection_Metrics.lib_s.Evaluator import *
from Object_Detection_Metrics.lib_s.utils import *

path = 'dataset_utils/datasets/MOT17/MOT17-13-SDP/'
out_path = '/home/marc/Downloads/13.avi'
snapshot_path = '/home/marc/Downloads/snapshots/yolo_832/snapshot0025.tar'

batch_size = 1
anchors = [(0.215 * 2, 0.8575 * 2), (0.3728125 * 2, 1.8225 * 2), (0.621875 * 2, 2.96625 * 2), (1.25 * 2, 6.12 * 2),
           (3.06125 * 2, 11.206875 * 2)]
net = Yolo(image_size=832, anchors=anchors)
net.load_snapshot(snapshot_path)
net.eval()
net.cuda()
imgs = []
_, img_paths, _ = motu.get_gt_img_inf(path)
for img_path in img_paths[int(len(img_paths) * .8):]:
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (net.image_size, net.image_size))
    img = image.copy()
    image = torch.from_numpy(np.transpose(image.astype(np.float32), (2, 0, 1))).unsqueeze(0).cuda()
    boxes = post_processing(net(image), net.image_size, net.anchors, 0.25, 0.5)
    img = draw_boxes_opencv(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), boxes[0])
    imgs.append(img)
    # cv2.imshow('img', img)
    # cv2.waitKey(20)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_path, fourcc, 20.0, (832, 832))
for i in imgs:
    out.write(i)
out.release()
