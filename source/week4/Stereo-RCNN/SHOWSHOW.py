import cv2 
import torch 
import numpy as np 
from model.utils import kitti_utils
from model.utils import vis_3d_utils as vis_utils 
from model.utils import box_estimator as box_estimator
from model.stereo_rcnn.resnet import resnet 

# 定义文件路径
img_l_path = 'data/kitti/object/training/image_2/000104.png' # 左眼图片文件路径
img_r_path = 'data/kitti/object/training/image_3/000104.png' # 右眼图片文件路径
calib_path = 'data/kitti/object/training/calib/000104.txt' # 左右相机校订文件路径
velodyne_path = 'data/kitti/object/training/velodyne/000104.bin' # 点云数据文件路径
predict_path = 'models_stereo/result/data/000104.txt' # 模型预测解雇文件路径
GT_path = 'data/kitti/object/training/label_2/000104.txt' # label文件路径

# 载入文件
img_left = cv2.imread(img_l_path) # 左眼图片
img_right = cv2.imread(img_r_path) # 由眼图片
calib = kitti_utils.read_obj_calibration(calib_path) # 相机校订
pointcloud = kitti_utils.get_point_cloud('demo/lidar.bin', calib) # 点云数据

# 获取所有label(只关注Car)
objects = kitti_utils.read_obj_data(GT_path, calib, img_left.shape) # 读取、生成完整label数据
# 遍历所有Car（one_object就代表一个Car的信息）
for one_object in objects:
    # 主要label
    CLS = one_object.cls
    TRUNCATE = one_object.truncate
    ALPHA = one_object.alpha
    BOXES = one_object.boxes
    POS = one_object.pos
    DIM = one_object.dim
    ORIENTATION = one_object.orientation
    R = one_object.R
    # 左、右、合并 三个box和keypoint
    BOX_L = BOXES[0].box
    BOS_R = BOXES[1].box
    BOX_M = BOXES[2].box
    KEYPOINT_L = BOXES[0].keypoints
    KEYPOINT_R = BOXES[1].keypoints
    KEYPOINT_M = BOXES[2].keypoints
    print('KEYPOINT_L:',KEYPOINT_L)
    print('KEYPOINT_R:',KEYPOINT_R)
    print('KEYPOINT_M:',KEYPOINT_M)

#     status, state = box_estimator.solve_x_y_z_theta_from_kpt(img_left.shape,calib,ALPHA,\
#         DIM,BOX_L,BOS_R)



# status, state = box_estimator.solve_x_y_z_theta_from_kpt(im2show_left.shape, calib, alpha, \
#                                 dim, box_left, box_right, cls_kpts[detect_idx].cpu().numpy())








