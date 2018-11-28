import dataset_utils.MOT_utils as motu
from os.path import isfile, join
import argparse
import numpy as np
import sys
import os
import random

width_in_cfg_file = 416.
height_in_cfg_file = 416.


def iou(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape
    return np.array(similarities)


def avg_iou(X, centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum += max(iou(X[i], centroids))
    return sum / n


def write_anchors_to_file(centroids, X, anchor_file):
    f = open(anchor_file, 'w')

    anchors = centroids.copy()
    print(anchors.shape)

    #for i in range(anchors.shape[0]):
    #    anchors[i][0] *= width_in_cfg_file / 32.
    #    anchors[i][1] *= height_in_cfg_file / 32.

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, ' % (anchors[i, 0], anchors[i, 1]))

    # there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n' % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1]))

    f.write('%f\n' % (avg_iou(X, centroids)))
    print()


def kmeans(X, centroids, eps, anchor_file):

    N = X.shape[0]
    iterations = 0
    k, dim = centroids.shape
    prev_assignments = np.ones(N) * (-1)
    iter = 0
    old_D = np.zeros((N, k))

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - iou(X[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)

        print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_D - D))))

        # assign samples to centroids
        assignments = np.argmin(D, axis=1)

        if (assignments == prev_assignments).all():
            print("Centroids = ", centroids)
            write_anchors_to_file(centroids, X, anchor_file)
            return

        # calculate new centroids
        centroid_sums = np.zeros((k, dim), np.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j))

        prev_assignments = assignments.copy()
        old_D = D.copy()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-filelist', default='Mot17_test_single.txt',
                        help='path to filelist\n')
    parser.add_argument('-output_dir', default='generated_anchors/anchors', type=str,
                        help='Output anchor directory\n')
    parser.add_argument('-num_clusters', default=0, type=int,
                        help='number of clusters\n')
    filelist = 'Mot17_test_single.txt'
    output_dir = 'anchors'
    arg_num_clusters = 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    paths_file = filelist
    paths = motu.parse_videos_file(paths_file)  # todo
    all_boxes = np.zeros([0, 2])

    for i, path in enumerate(paths):
        # gt format: [frame id 4 box parameter]
        # tmp debug until i download the dataset
        path = '../' + path
        # tmp
        # get ground truth (gt)
        gt, info = motu.get_gt_info(path)
        num_frames = info.seqLength

        # remove everything that is not a pedestrian
        cond = np.vstack([(gt[:, 7] == 1).reshape(1, -1), (gt[:, 7] == 7).reshape(1, -1)]).T
        gt = gt[np.where(np.any(cond, axis=1))]

        # change negative boxes (dont know if it is really important)(remove it if not necessary anymore)
        # ( i really hope there are no to big boxes -> there are to big boxes (face palm))
        neg_ids_x = np.where(gt[:, 2] < 0)
        neg_ids_y = np.where(gt[:, 3] < 0)
        pos_ids_x = np.where(gt[:, 2] + gt[:, 4] >= info.imWidth)
        pos_ids_y = np.where(gt[:, 3] + gt[:, 5] >= info.imHeight)
        gt[neg_ids_x, 4] += gt[
            neg_ids_x, 2]  # if we move the top left corner into the image we must adapt the height
        gt[neg_ids_x, 2] = 0
        gt[neg_ids_y, 5] += gt[neg_ids_y, 3]  # same here (dont know if y<0 exists)
        gt[neg_ids_y, 3] = 0
        gt[pos_ids_x, 4] = info.imWidth - gt[pos_ids_x, 2] - 1  # equal would also be bad
        gt[pos_ids_y, 5] = info.imHeight - gt[pos_ids_y, 3] - 1

        all_boxes = np.concatenate([all_boxes, gt[:, [4, 5]]])

    annotation_dims = all_boxes

    eps = 0.005

    if arg_num_clusters == 0:
        for num_clusters in range(1, 11):  # we make 1 through 10 clusters
            anchor_file = join(output_dir, 'anchors%d.txt' % (num_clusters))

            indices = [random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
            centroids = annotation_dims[indices]
            kmeans(annotation_dims, centroids, eps, anchor_file)
            print('centroids.shape', centroids.shape)
    else:
        anchor_file = join(output_dir, 'anchors%d.txt' % (arg_num_clusters))
        indices = [random.randrange(annotation_dims.shape[0]) for i in range(arg_num_clusters)]
        centroids = annotation_dims[indices]
        kmeans(annotation_dims, centroids, eps, anchor_file)
        print('centroids.shape', centroids.shape)

if __name__ == "__main__":
    main(sys.argv)