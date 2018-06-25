# -*- coding: utf-8 -*-

"""A very simple MNIST classifier. 
用卷积神经网络改进后的Softmax（Softmax regression + CNN）
See extensive documentation at
http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html
Softmax回归(softmax regression)基本概念：MNIST的每一张图片都表示一个数字，从0到9。我们希望得到给定图片代表每个数字的概率。
比如说，我们的模型可能推测一张包含9的图片代表数字9的概率是80%但是判断它是8的概率是5%（因为8和9都有上半部分的小圆），然后给予它代表其他数字的概率更小的值。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
import input_data
import mnist_cnn

import tensorflow as tf

def weight_variable(shape):
  """此模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。"""
  # 从截断的正态分布中输出随机值(随机数Matrix)，均值和标准差自己设定.shape表示生成张量的维度，mean是均值，stddev是标准差
  # 如shape=[5,3]，则生成5行*3列的二维数组；如shape=[5, 5, 1, 32]
  # 截断：产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。即从正态分布的“中间部分”取值
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """由于使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题（dead neurons）。"""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  """卷积使用1步长（stride size），0边距（padding size）的模板"""
  # conv2d: 卷积
  # x 指需要做卷积的输入图像，格式为[batch, height, width, channels], 分别为[输入的批次数量、图像的高（行数）、宽（列数）、图像通道数]
  # W 卷积矩阵，二维、分别为[高，宽]
  # strides 为滑动窗口尺寸，分别为[1, height, width, 1]， 通常 strides[0]=strdes[3]=1
  # padding 为扩展方式，有两种 vaild 和 same
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  """池化用简单传统的2x2大小的模板做max pooling"""
  # tf.nn.max_pool(value, ksize, strides, padding, name=None)
  # 第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
  # 第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
  # 第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
  # 第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
  # 返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

sess = tf.InteractiveSession()

# Create the model.
# y=softmax(wx+b), w-权重值, b-偏置量
# 每张图片28x28=784; None表示此张量的第一个维度可以是任何长度的
x = tf.placeholder(tf.float32, [None, 784])
# W and b就是通过训练确定的参数；一个Variable代表一个可修改的张量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# matmul:矩阵相乘. y也是[None, 10]的二维数组
#y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer. y_是标准答案的onehot表达
y_ = tf.placeholder(tf.float32, [None, 10])

# NOTE 步骤与图片维度变化：
# Raw[28,28] -卷积1-> Conv1[28,28] -池化1-> Pool1[14, 14] --卷积2-> Conv2[14,14]
# -ReLU-> ReLU[14,14] -池化2-> Pool2[7, 7]...
# 这里卷积步长是1，所以不会发生降维。池化将导致维度减半

## 第一层卷积（conv2d中只能指定卷积核个数，不能设置卷积核初始值）
# 卷积在每个5x5的patch中算出32个特征。卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目（灰度图，通道数=1），最后是输出的通道数目/卷积核个数。 
# 而对于每一个输出通道都有一个对应的偏置量bias。
# NOTE 输出的32通道什么意思？32=2^5，即要求用32个[5,5]卷积核生成32个feature map。这个参数是“猜”出来的，可以理解为卷积核数越大，挖掘分析的越深。
# NOTE 卷积的维度一般都是奇数，如[3,3],[5,5]等，卷积核越大，分析出的特征越抽象/宏观。卷积核数目设置，按照16的倍数倍增，结合了gpu硬件的配置。
# BTW：深度学习工程师50%的时间在调参数，49%的时间在对抗过/欠拟合，剩下1%时间在修改网上down下来的程序
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
# -1什么意思？参数为-1时，那么reshape函数会根据另一个参数的维度计算出数组的另外一个shape属性值。
# NOTE 这里-1代表样本图片的数量！
x_image = tf.reshape(x, [-1,28,28,1])

# ReLU函数其实是分段线性函数，把所有的负值都变为0，而正值不变，这种操作被成为单侧抑制（即抑制部分神经元，只让部分神经元起作用，并对其进行分析）。
# 人的大脑在进行运算或者欣赏时，都会有一部分神经元处于激活或是抑制状态，可以说是各司其职。
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

## 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

## 密集连接层
# 此时图片尺寸减小到7x7
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

## Dropout
keep_prob = tf.placeholder("float")
# dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。注意是暂时，对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络。
# dropout是CNN中防止过拟合提高效果的一个大杀器
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

## 训练和评估模型
# 损失函数是目标类别和预测类别之间的交叉熵cross entropy
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# 用更加复杂的"ADAM优化器"来做梯度最速下降
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

# 在feed_dict中加入额外的参数keep_prob来控制dropout（丢弃）比例
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print ('Have a rest. ', end = '')
    print('step %d, training accuracy %g' % (i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  print('h_conv1:', h_conv1)
  print('h_pool1', h_pool1)
  print('h_conv2:', h_conv2)
  print('h_pool2', h_pool2)

# NOTE 如下方式写(end = '')则不会有换行！
print ('Test Result: ', end = '')
print('test accuracy %g' % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))