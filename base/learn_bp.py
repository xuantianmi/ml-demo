#-*- coding: UTF-8 -*-

"""
实现BP神经网络
一般步骤总结
√ 0 导入模块，生成模拟数据集
    import
    常量定义
    生成数据集
√1 前向传播：定义输入、参数和输出
    x= y_=
    w1= w2=
    a= y=
√2 反向传播：定义损失函数、反向传播方法
    loss=
    train_step=
√3 生成会话，训练STEPS轮
"""

import tensorflow as tf
import numpy as np
NUM_CORES = 4  # Choose how many cores to use.
BATCH_SIZE = 8
SEED = 23455

#基于seed产生随机数
rdm = np.random.RandomState(SEED)
#随机数返回32行2列的矩阵 表示32组 体积和重量 作为输入数据集
X = rdm.rand(32,2)
#从X这个32行2列的矩阵中 取出一行 判断如果和小于1 给Y赋值1 如果和不小于1 给Y赋值0 
#作为输入数据集的标签（正确答案） 
Y_ = [[int(x0 + x1 < 1)] for (x0, x1) in X]

#1定义神经网络的输入、参数和输出,定义前向传播过程。
x = tf.placeholder(tf.float32, shape=(None, 2))
y_= tf.placeholder(tf.float32, shape=(None, 1))

w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#2定义损失函数及反向传播方法。
loss_mse = tf.reduce_mean(tf.square(y-y_)) 
#均方误差MSE损失函数
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
#随机梯度下降算法训练参数

#3生成会话，训练STEPS轮
#with tf.Session() as sess:
# NOTE 指定CPU数量
with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                   intra_op_parallelism_threads=NUM_CORES)) as sess:
    #init_op = tf.global_variables_initializer()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # 输出目前（未经训练）的参数取值。

    # 训练模型。
    STEPS = 3000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            #每训练500个steps打印训练误差
            total_loss = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print("After %d training step(s), loss_mse on all data is %g" % (i, total_loss))