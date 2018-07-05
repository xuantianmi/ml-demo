'''
通过TensorFlow实现一个LSTM(Recurrent Neural Network).
通过训练样本学习预测：通过三个已知的开奖球预测下一开奖球

Author: Merlin
>python3 digital3_lstm.py
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import os
import collections
import time

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = './logs'
writer = tf.summary.FileWriter(logs_path)

# Text file containing words for training
training_file = './data/cp.txt'

def read_data(path):
    """Load Dataset from File"""
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data

#training_data = read_data(training_file)
text = read_data(training_file)
# 将文件字符串转成一维向量，每个元素是三个连续的开奖号码，如123
list1 = np.array(text.split('\n'))
print(list1)
list2 = [int(item) for item in list1]
# 将一维的开奖号码转成二维，即将开奖号码的三个数字split成数组
arr1 = [[item[0:1], item[1:2], item[2:3]] for item in list1]
arr2 = np.array(arr1, dtype=np.int32)
print(arr2)
# 将上面的二维数组转成一维向量，表示历次的开奖球号的列表（不再体现明确显示奖期）
training_data = arr2.reshape((arr2.size))
print(training_data)
print("Loaded training data...")

# 0-9
digital_size = 10

# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 3

# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, digital_size])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, digital_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([digital_size]))
}

def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)

        symbols_in_keys = [ [ training_data[i]] for i in range(offset, offset+n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        symbols_out_onehot = np.zeros([digital_size], dtype=float)
        symbols_out_onehot[training_data[offset+n_input]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc

        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = int(tf.argmax(onehot_pred, 1).eval())
            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        step += 1
        offset += (n_input+1)
    
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    
    while True:
        prompt = "%s digitals: " % n_input
        three_digitals = input(prompt)
        three_digitals = three_digitals.strip()
        digitals = three_digitals.split(' ')
        if len(digitals) != n_input:
            continue
        try:
            symbols_in_keys = [digitals[i] for i in range(len(digitals))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                four_digitals = "%s %s" % (three_digitals, onehot_pred_index)
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(four_digitals)
        except:
            print("Digitals not in dictionary")

    
# Tensorboard
#print("\ttensorboard --logdir=%s" % (logs_path))
#print("Point your web browser to: http://localhost:6006/")