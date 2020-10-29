![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

Yolo v4 paper: https://arxiv.org/abs/2004.10934

Yolo v4 source code: https://github.com/AlexeyAB/darknet

Yolov v4 tiny discussion: https://www.reddit.com/r/MachineLearning/comments/hu7lyt/p_yolov4tiny_speed_1770_fps_tensorrtbatch4/

Useful links: https://medium.com/@alexeyab84/yolov4-the-most-accurate-real-time-neural-network-on-ms-coco-dataset-73adfd3602fe?source=friends_link&sk=6039748846bbcf1d960c3061542591d7

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).


# 一、编译：

	[以/trainDarknet-yolov3为根目录]
	
	1.1 根据电脑配置设置Makefile：这里默认使用gpu，cudnn，不使用opencv
	
	1.2 make

# 二、测试：

	[以/trainDarknet-yolo3为根目录]
	
	2.0 下载好权重，yolov3.weights放置到./;darknet53.conv.74放置到./myData/weights/preWeights/
	
	2.1 测试命令：./darknet detect myData/cfg/myYolov3.cfg myData/weights/my_yolov3_30000.weights myData/JPEGImages/01_000038.png
	
	2.2 测试结果将保存在./results/中
	
	2.3 如果想使用voc或者coco真实标签(根据自己权重放置位置调整命令)：
	
		./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg
		
	    或者修改examples/darkent.c中的447行，使用如下命令：
		
		./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg

# 三、训练：

	[以/myData为运行根目录]
	
	3.1 voc数据格式：xml文件放到myData/Annotations，图片放到myData/JPEGImages（文件夹不存在自己新建就行）
	
	3.2 修改cfg中的：myData.data主要是路径和种类数
	
			myData.names根据自己的类别填写，顺序需要和dataPrecess.py一样
			
			myYolov3.cfg根据自己的需求需要更改，搜索yolo，有三个地方：classes = n，filters=3 × (5 + n)
			
			根据自己电脑的性能修改batch和subdivisions值，一般来讲电脑性能越好batch值越大，subdivisions值越小，最好是2的整数指数值
			
	3.3 根据里面的注释运行dataPrecess.py，将voc数据转成darknet可以训练的格式；
	
	[以/trainDarknet-yolov3为根目录]
	
	3.4 训练执行命令：./darknet detector train myData/cfg/myData.data myData/cfg/myYolov3.cfg myData/weights/preWeights/darknet53.conv.74

# 四、权重：

	链接：https://pan.baidu.com/s/1Dw3-T9fxcPSbmrXH09u6tQ 
	
	提取码：vgib
	
# 五、更新

	[添加了修改yolo.cfg参数的函数，使用更加方便]

	3.1 voc数据格式：xml文件放到myData/Annotations，图片放到myData/JPEGImages（文件夹不存在自己新建就行）
	
	[以/myData为运行根目录]
	
	3.2 myData.names根据自己的类别填写，要写全
	
	3.3 执行命令：python dataPrecess.py
	
	[以/trainDarknet-yolov3为根目录]
	
	3.4 执行命令：./darknet detector train myData/cfg/myData.data myData/cfg/myYolov3.cfg myData/weights/preWeights/darknet53.conv.74
	