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
ref: 
- https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/examples/tutorials/mnist
- http://www.tensorfly.cn/tfdoc/tutorials/mnist_tf.html

### Softmax Regression(mnist_softmax.py)
阅读代码并增加注释

### Softmax Regression(mnist_cnn.py)
卷积神经网络+Softmax regression, 阅读代码并增加注释

### 简易前馈神经网络(fully_connected_feed.py)
展示如何利用TensorFlow使用（经典）MNIST数据集训练并评估一个用于识别手写数字的简易前馈神经网络（feed-forward neural network）。

#### mnist.py
构建一个完全连接（fully connected）的MINST模型所需的代码
经过三阶段的模式函数操作：推理inference()， 损失loss()，和training()。图表就构建完成了。
- mnist.inference() —— 尽可能地构建好图表，满足促使神经网络向前反馈并做出预测的要求。
- mnist.loss() —— 往inference图表中添加生成损失（loss）所需要的操作（ops）。
- mnist.training() —— 往损失图表中添加计算并应用梯度（gradients）所需的操作。
- mnist.evaluation(logits, labels) —— 对比labels做模型评估

#### fully_connected_feed.py
利用下载的数据集训练构建好的MNIST模型的主要代码，以数据反馈字典（feed dictionary）的形式作为输入模型。
- placeholder_inputs(batch_size):将生成两个tf.placeholder操作，定义传入图表中的shape参数
- fill_feed_dict(data_set, images_pl, labels_pl)
- do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_set)
- run_training(): 程序的实际入口module!
- main(_) python标准入口

#### mnist_with_summaries.py
TensorBoard:可视化学习
tensorboard --logdir=/path/to/log-directory。其中，logdir是SummaryWriter函数指定的日志存储目录
BTW：fully_connected_feed.py也有日志哈～

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
- pyc只有在对应文件被引用时才会生成（如A.py引用B.py, 执行A时，会生成B.pyc）。如果有pyc文件，则对应的py文件可以没有，以达到一定保密性。

## Glossary
|Name|Content|
|:- | :- | 
|dw Conv|DWConv表示depthwise separable convolution|
|pw Conv|？？？|
|K-fold cross-validation|K层交叉检验就是把原始的数据随机分成K个部分。在这K个部分中，选择一个作为测试数据，剩下的K-1个作为训练数据。<br/>交叉检验的过程实际上是把实验重复做K次，每次实验都从K个部分中选择一个不同的部分作为测试数据（保证K个部分的数据都分别做过测试数据），剩下的K-1个当作训练数据进行实验，最后把得到的K个实验结果平均。| 
|LMTCNN|Lightweight Multi-task CNN|
|ReLu|Rectified Linear Units激活函数|
|softmax 回归|多项式逻辑回归|
|tf.nn.softmax|Multiplies matrix|
|BP神经网络|BP(back propagation)神经网络是1986年由Rumelhart和McClelland为首的科学家提出的概念，是一种按照误差逆向传播算法训练的多层前馈神经网络，是目前应用最广泛的神经网络。|
|LSTM|Long Short-Term Memory，是长短期记忆网络，是一种时间递归神经网络，适合于处理和预测时间序列中间隔和延迟相对较长的重要事件。|
|MSE|Mean Square Error均方误差|
|RNN|Recurrent neural network，递归神经网络（RNN），是两种人工神经网络的总称，一种是时间递归神经网络（recurrent neural network），另一种是结构递归神经网络（recursive neural network）|

时间序列预测分析就是利用过去一段时间内某事件时间的特征来预测未来一段时间内该事件的特征。这是一类相对比较复杂的预测建模问题，和回归分析模型的预测不同，时间序列模型是依赖于事件发生的先后顺序的，同样大小的值改变顺序后输入模型产生的结果是不同的。
举个栗子：根据过去两年某股票的每天的股价数据推测之后一周的股价变化；根据过去2年某店铺每周想消费人数预测下周来店消费的人数等等
