#coding:utf-8
import numpy as np
 
 
####################仅仅是一个示例###################################
 
 
# 双目相机参数
class stereoCamera1(object):
#class stereoCameral(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([[1499.64168081943, 0, 1097.61651199043],
                                         [0., 1497.98941910377, 772.371510027325],
                                         [0., 0., 1.]])
        # 右相机内参
        self.cam_matrix_right = np.array([[1494.85561041115, 0, 1067.32184876563],
                                          [0., 1491.89013795616, 777.983913223449],
                                          [0., 0., 1.]])
 
        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-0.110331619900584, 0.0789239541458329, -0.000417147132750895, 0.00171210128855920, -0.00959533143245654]])
        self.distortion_r = np.array([[-0.106539730103100, 0.0793246026401067, -0.000288067586478778, -8.92638488356863e-06, -0.0161669384831612]])
 
        # 旋转矩阵
        self.R = np.array([[0.993995723217419, 0.0165647819554691, 0.108157802419652],
                           [-0.0157381345263306, 0.999840084288358, -0.00849217121126161],
                           [-0.108281177252152, 0.00673897982027135, 0.994097466450785]])
 
        # 平移矩阵
        self.T = np.array([[-423.716923177417], [2.56178287450396], [21.9734621041330]])
 
        # 焦距
        self.focal_length = 1602.46406  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]
 
        # 基线距离
        self.baseline = 423.716923177417  # 单位：mm， 为平移向量的第一个参数（取绝对值）
