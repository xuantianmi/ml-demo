#-*- coding: UTF-8 -*-

"""下面都是一些涉及基础概念的例子，便于学习和理解基础知识"""

import tensorflow as tf

def full_connect_param():
    """实现了前向传输，但是使用了占位符placeholder，开始没有 给x具体的值，而是在运行会话时通过feed_dict传入了多组值。也就产生了多组结果。"""
    #定义输入和参数
    #用placeholder定义输入（sess.run喂多组数据）
    x = tf.placeholder(tf.float32, shape=(None, 2))
    w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    #定义前向传播过程
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    #调用会话计算结果
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()  
        sess.run(init_op)
        print(sess.run(y, feed_dict={x: [[0.7,0.5],[0.2,0.3],[0.3,0.4],[0.4,0.5]]}))

def full_connect():
    """两层简单神经网络（全连接）,实现了前向传输"""
    #定义输入和参数
    x = tf.constant([[0.7, 0.5]])
    # 随机生成二维矩阵2*3
    w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    #定义前向传播过程
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    #用会话计算结果
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print(sess.run(y))

def tf_matmul():
    """实现两个矩阵乘法"""
    x = tf.constant([[1.0, 2.0]])
    w = tf.constant([[3.0], [4.0]])
    y=tf.matmul(x,w)
    # 此时y只是一个张量运算符表达式
    print(y)

    with tf.Session() as sess:
        print(sess.run(y))

def tf_flags():
    """调用flags内部的DEFINE_string函数来制定解析规则. 也可以接收命令行参数！"""
    tf.flags.DEFINE_string("para_name_1","default_val", "description")
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 10)")

    #FLAGS是一个对象，保存了解析后的命令行参数
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()#进行解析，加上这一句可以把FLAGS.__flags变成一个字典
    print(FLAGS.batch_size)#运行结果输出64
    print(FLAGS.__flags)#运行结果见下图

    # FLAGS可以从命令行接受参数。
    # python3 test.py --batch_size=100

def tf_onehot():
    """此处sparse_to_dense等价于tf.one_hot(labels,NUM_CLASSES,1,0)，但sparse_to_dense还有更灵活用法"""
    sess = tf.InteractiveSession()
    NUM_CLASSES = 10
    labels = tf.Variable([1,2,3,4])
    batch_size = tf.size(labels)
    labels1 = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    #concated0 = tf.concat(0, [indices, labels1])
    concated = tf.concat(1, [indices, labels1])
    tf_pack = tf.pack([batch_size, NUM_CLASSES])
    onehot_labels = tf.sparse_to_dense(concated, tf_pack, 1.0, 0.0)
    
    tf.initialize_all_variables().run()
    print('labels:', sess.run(labels))
    print('batch_size:', sess.run(batch_size))
    print('indices:', sess.run(indices))
    print('labels1:', sess.run(labels1))
    print('concated:', sess.run(concated))
    # print('concated-0:', sess.run(concated0))
    print('tf_pack:', sess.run(tf_pack))
    print('onehot_labels:', sess.run(onehot_labels))

    return onehot_labels

if __name__ == '__main__':
    #tf_flags()
    #tf_onehot()
    #tf_matmul()