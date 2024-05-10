# FacialExpressionRecognition

## 简介

基于ResNet50在RAF-DB数据集上的人脸情绪识别

## 环境部署

推荐使用conda虚拟环境
```
git clone https://github.com/Qingyyx/FacialExpressionRecognition.git
cd FacialExpressionRecognition
conda create -n FER python=3.8 -y
conda activate FER
pip install torch==1.12.0 gradio==4.29.0 torchvision==0.13.0 opencv-python==3.4.15.55 h5py==3.11.0
```
## 演示

预训练模型已经传到百度网盘，[连接](https://pan.baidu.com/s/1-0HyQoiX9Bmz7IsHid4-Cg )给出，提取码：7i8j。下载后将模型放入`RAF_`文件夹下。然后`python visualize_pro.py`

## FRA-DB数据集

来自[这里](https://paperswithcode.com/sota/facial-expression-recognition-on-raf-db)，RAF-DB数据集由100*100的rgb图像组成，它包含 29672 张面部图像.每个图像包含7个不同情绪的图像。

下载完成后，运行`python preprocess_RAF+.py`进行图片预处理，生成h5文件

## 训练和评估

`python mainpre_RAF.py --bs 128 --lr 0.003`  
`python visualize_pro.py`
## 精确度

PublicTrain_acc： 99.999%   
PublicTest_acc： 82.226%  

## 参考

[1] [Facial Expression Recognition with Deep Convolutional Neural Networks](https://arxiv.org/abs/1310.5401)

[2]https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
