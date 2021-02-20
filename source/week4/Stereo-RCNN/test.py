# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick

# Modified by Peiliang Li for Stereo RCNN test
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import shutil
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision 
import math as m
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg
from model.rpn.bbox_transform import clip_boxes
# from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, kpts_transform_inv, border_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.stereo_rcnn.resnet import resnet
from model.utils import kitti_utils
from model.utils import vis_3d_utils as vis_utils
from model.utils import box_estimator as box_estimator
from model.dense_align import dense_align

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test the Stereo R-CNN network')

    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models_stereo",
                        type=str)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=12, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=6477, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    np.random.seed(cfg.RNG_SEED)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb('kitti_val', False)
    print('{:d} roidb entries'.format(len(roidb)))

    # input_dir = args.load_dir + "/"
    # if not os.path.exists(input_dir):
    #     raise Exception('There is no input directory for loading network from ' + input_dir)
    #  load_name = os.path.join(input_dir, 'stereo_rcnn_{}_{}.pth'.format(args.checkepoch, args.checkpoint))
    load_name = '/home/kkb/users/yqh/Stereo-RCNN/models_stereo/stereo_rcnn_12_6477.pth'

    result_dir = args.load_dir + '/result/'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)

    # initilize the network here.
    stereoRCNN = resnet(imdb.classes, 101, pretrained=False)
    stereoRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    stereoRCNN.load_state_dict(checkpoint['model'])
    print('load model successfully!')

    with torch.no_grad():
        # initilize the tensor holder here.
        im_left_data = Variable(torch.FloatTensor(1).cuda())
        im_right_data = Variable(torch.FloatTensor(1).cuda())
        im_info = Variable(torch.FloatTensor(1).cuda())
        num_boxes = Variable(torch.LongTensor(1).cuda())
        gt_boxes_left = Variable(torch.FloatTensor(1).cuda())
        gt_boxes_right = Variable(torch.FloatTensor(1).cuda())
        gt_boxes_merge = Variable(torch.FloatTensor(1).cuda())
        gt_dim_orien = Variable(torch.FloatTensor(1).cuda())
        gt_kpts = Variable(torch.FloatTensor(1).cuda())

        stereoRCNN.cuda()

        eval_thresh = 0.05
        vis_thresh = 0.7

        num_images = len(imdb.image_index)

        dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                            imdb.num_classes, training=False, normalize = False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0,
                                pin_memory=True)

        data_iter = iter(dataloader)
        data = next(data_iter)
        data = next(data_iter)
        for i in data:
            print(i)