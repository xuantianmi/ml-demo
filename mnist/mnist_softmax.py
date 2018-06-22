# -*- coding: utf-8 -*-
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
Softmax回归(softmax regression)基本概念：MNIST的每一张图片都表示一个数字，从0到9。我们希望得到给定图片代表每个数字的概率。
比如说，我们的模型可能推测一张包含9的图片代表数字9的概率是80%但是判断它是8的概率是5%（因为8和9都有上半部分的小圆），然后给予它代表其他数字的概率更小的值。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
import input_data

import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

sess = tf.InteractiveSession()

# Create the model.
# y=softmax(wx+b), w-权重值, b-偏置量
# 每张图片28x28=784; None表示此张量的第一个维度可以是任何长度的
x = tf.placeholder(tf.float32, [None, 784])
# W and b就是通过训练确定的参数；一个Variable代表一个可修改的张量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
print('b:', b)
# matmul:矩阵相乘. y也是[None, 10]的二维数组
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer. y_是标准答案的onehot表达
y_ = tf.placeholder(tf.float32, [None, 10])
# tf.log: 对张量所有元素进行对数运算
# 在机器学习，我们通常定义指标来表示一个模型是坏的，这个指标称为成本（cost）或损失（loss），然后尽量最小化这个指标。
# 一个常见的成本函数是“交叉熵”（cross-entropy）,下面就是标准公式
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Train
tf.initialize_all_variables().run()
for i in range(1000):
  # NOTE 随机抓取训练数据中的100个批处理数据点，确认下是否是随机的
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # 通过Feed机制，在运行时替换两个tensor(用placeholder声明，类似与行參与实参的关系)。
  # 以此达到替换不同批次（每次100张图片）的训练数据的目的
  # NOTE 每执行一次，W和b都会更新以求最优
  train_step.run({x: batch_xs, y_: batch_ys})

# Test trained model
# tf.argmax：给出某个tensor对象在某一维上的其数据最大值所在的索引值
# 如果两个索引值相同，说明预测正确！返回一组布尔值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 把布尔值转换成浮点数，然后取平均值。例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 用测试数据验证模型的准确率～
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
