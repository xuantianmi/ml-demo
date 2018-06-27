#-*- coding: UTF-8 -*-

import tensorflow as tf

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
    sess = tf.InteractiveSession()
    #tf_flags()
    tf_onehot()