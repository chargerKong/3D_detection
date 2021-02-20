# AutoDrivingStereoVision20210124

#### 介绍
- 课后作业同步
- 项目地址：https://gitee.com/anjiang2020_admin/auto_driving_stereo_vision20210124.git

#### week4

```
week4 复杂场景下无人车如何检测物体-三维目标检测
项目代码地址：https://gitee.com/anjiang2020_admin/auto_driving_stereo_vision20210124.git

Pipeline:
1. 一个三维重建代码样例
2. 三维重建所依赖的相机内参与外参
3. 双目图片可恢复出三维图，双目bbox是否可恢复出三维bbox
4. 三维目标检测网络:stereoRCNN的设计思路

作业：
   [必做]1. 完成代码week4/3DReconstruction/main_pptk.py的202行，203行，226行，256行填空，并能成功运行，查看到三维重建整个过程。
    作业步骤：
         1 202行需要你从middlebury数据集：https://vision.middlebury.edu/stereo/data/scenes2014/zip/ 下载一对双目图，并把你下载到的图片路径填入。
         2 226行的填空，是将视差图转换为点云的过程，这里需要调用opencv的函数cv2.reprojectImageTo3D,请查阅其使用方法，并使用
         3 256行的填空，主要练习用旋转矩阵对点云进行旋转。所以这里需要根据要求计算出旋转角度，然后对点云数据进行计算，得到旋转后的点云。
    作业要求：
         1 完成填空后，运行作业代码。需要提交内容：代码和 4张图片。4张图片分别为左图，右图，视差图，以及最后显示出的点云模型的截图。
         
   [选做]2. 运行出老师提供的stereoRCNN cpu版本的代码。
    建议步骤：
          1. 从链接:https://pan.baidu.com/s/1LzgooYdgm0vMziQh3RdX2Q  密码:7ayx 下载预训练的StereoRCNN模型，将其放入week4/Stereo-RCNN/models_stereo文件夹内
          2. 切换到目录Stereo-RCNN下
          3. 用命令：python demo.py 运行demo，20秒左右，会得到demo_result.jpg
          4. 本代码是StereoRCNN的代码，最好是先运行出模型的demo.py代码，也就是完整的运行出模型的前向计算过程，然后断点调试可以产看模型整个前向计算过程，查看如何使用模型输入的2d 的bbox信息的2d转3d的求解方程组的得到3d 的bbox的过程。 
          5. 建议预习一下本代码，下节课会讲，讲完后要求大家去下载数据集训练。
   目录文件说明：
    stereoRCNN论文：week4/Stereo R-CNN based 3D Object Detection for Autonomous Driving.pdf
    week4/3DReconstruction:待填空作业代码,三维重建代码
    张正友标定法：张正友标定发.pdf
   资料整理：
      1 使用opencv 进行双目测距c++ 版本，含相机标定，不含点云显示:https://www.cnblogs.com/zhiyishou/p/5767592.html
      2 使用opencv进行双目测距python版本[课堂代码，含点云显示]:https://blog.csdn.net/dulingwen/article/details/98071584
      3 stereoRCNN 代码-gpu版本：https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN
      4 stereoRCNN 代码-cpu版本demo: week4/Stereo-RCNN代码
      5 用matplotlib来显示3d图：https://blog.csdn.net/groundwalker/article/details/84786773
      6 middlebury数据集：https://vision.middlebury.edu/stereo/data/scenes2014/zip/
      7 双目视觉算法的排名表:https://vision.middlebury.edu/stereo/eval3/

```



```
week5 复杂场景下无人车如何检测物体-三维目标检测实战
项目代码地址：https://gitee.com/anjiang2020_admin/auto_driving_stereo_vision20210124.git

Pipeline:
1. week4作业以及坐标系转换方法
2. StereoRCNN前向计算得到3d-box思路梳理
3. 左box与右box计算得到3D-bbox实现过程
4. StereoRCNN训练代码以及训练数据的生成

作业：
   [必做]1. 将stereoRCNN的输出结果显示成点云的形式。
         作业说明：
             在stereoRCNN工程代码中，python demo.py是将结果显示在2d图片上，
             为了加深2d->3d转换的认识，
             这里需要大家把stereoRCNN检测到的3d-box显示到点云里.
    作业步骤：
         1 首先，读出雷达的点云数据，用pptk显示到3d空间里。
         2 把demo.py最后计算得到的x,y,z,theta,w,h,l,表示的3d矩形框先换算成8个顶点。
         3 利用8个顶点，把12条边上的点的(x，y,z )坐标求出来即可。
         4 把12条边的点云数据与雷达的点云数据合并，显示到3d空间里，并给12条边上色，与雷达的点云做区分。
         5 调整雷达点云 与 3d-bbox点云的相互尺度，角度等，使显示正常。
         6  参考代码：week4/Stereo-RCNN/demo_pptk.py 第256,257,361行。
    作业要求：
         1 需要提交内容：修改后的代码demo_week5.py以及运行截图。
         
   [选做]2. 下来kitti数据集，用GPU 训练stereoRCNN。
    建议步骤：
          1. 准备好kitti数据集，讲数据集传到ai studio上。
          1. 将week4/Stereo-RCNN 复制到ai studio上。
          2. 调整代码中的数据集加载目录，然后运行trainval.py 

   目录文件说明：
    homework_answer/3DReconstruction:week4作业答案
   资料整理：
      1 kitti 数据集下载链接以及使用办法：[待添加]
```
图1 点云显示图
![输入图片说明](https://images.gitee.com/uploads/images/2021/0201/200909_578e1d51_7401441.png "屏幕截图.png")

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
