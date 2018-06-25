# Mache Learning Demos(Mac OS X, Python2.7)
Demo代码均源自如下资源，并增加适当注释和调整：
- https://tensorflow.google.cn/
- https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mnist/

## Prepare
```
更新pip，以便支持TLSv1.0和TLSv1.1
$ curl https://bootstrap.pypa.io/get-pip.py | sudo python
Install Numpy, nose and tornado for matplotlib.
$ python -m pip install --user numpy nose tornado
Install TensorFlow, 当前版本只支持 CPU
$ sudo pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow  --ignore-installed numpy
用以解决import tensorflow错误
$ pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall --upgrade protobuf
```
注意：如下方式可以强制安装最新版本，避免如下错误“Cannot uninstall ‘***’. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.”
$ sudo pip install six --ignore-installed six

## Tensor's mnist demo
ref: https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/examples/tutorials/mnist

### Softmax Regression(mnist_softmax.py)
阅读代码并增加注释

### Softmax Regression(mnist_cnn.py)
卷积神经网络+Softmax regression, 阅读代码并增加注释

### 简易前馈神经网络(fully_connected_feed.py)
展示如何利用TensorFlow使用（经典）MNIST数据集训练并评估一个用于识别手写数字的简易前馈神经网络（feed-forward neural network）。

### (BTW)Python3.3 up内置了pip包管理器
```
python3 -m pip install --upgrade pip
sudo pip3 install numpy
sudo pip3 install scipy
sudo pip3 install matplotlib
sudo pip3 install tensorflow-gpu #安装gpu版tensorflow
```
注：目前用Python2.7，所以这部分暂时用不到～

## Ref
- http://www.tensorfly.cn/
- https://tensorflow.google.cn/
- https://tensorflow.google.cn/tutorials/
- TF基本用法：http://www.tensorfly.cn/tfdoc/get_started/basic_usage.html

### 图像通道数
- 单通道图：俗称灰度图，每个像素点只能有有一个值表示颜色，它的像素值在0到255之间，0是黑色，255是白色，中间值是一些不同等级的灰色。（也有3通道的灰度图，3通道灰度图只有一个通道有值，其他两个通道的值都是零）。
- 三通道图：每个像素点都有3个值表示，所以就是3通道。也有4通道的图。例如RGB图片即为三通道图片，RGB色彩模式是工业界的一种颜色标准，是通过对红(R)、绿(G)、蓝(B)三个颜色通道的变化以及它们相互之间的叠加来得到各式各样的颜色的，RGB即是代表红、绿、蓝三个通道的颜色，这个标准几乎包括了人类视力所能感知的所有颜色，是目前运用最广的颜色系统之一。总之，每一个点由三个值表示。
- 四通道：在RGB基础上加上alpha通道，表示透明度，alpha=0表示全透明
- 其他 略

## Tips
- 模型普遍是套用那些经典模型结构AlexNet、VGG、inception、resnet，在这些基础上综合考虑自己gpu的存储、性能确定网络结构和复杂度。