# -*- coding: utf-8 -*-
#!/usr/bin/env python

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

"""Functions for downloading and reading MNIST data."""
# __future__: 运行Python3.X功能 on Python 2.7
# absolute_import: 区分出绝对导入和相对导入
# division: 支持浮点除/整除(地板除或截断) 10 / 3=3.3333333333333335, 10 // 3=3
# print_function: print("Hello world"), not print "Hello world"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import tensorflow.python.platform

# NumPy是Python语言的一个扩充程序库。支持高级大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
import numpy
## six是用来兼容python 2 和 3的, six.moves 是用来处理那些在2和3里面函数的位置有变化的，直接用six.moves就可以屏蔽掉这些变化
# Urllib是python内置的HTTP请求库
from six.moves import urllib
# xrange 函数用法与 range 完全相同，所不同的是生成的不是一个数组，而是一个生成器。
from six.moves import xrange  # pylint: disable=redefined-builtin
# TensorFlow™ 是一个采用数据流图（data flow graphs），用于数值计算的开源软件库。
# 节点（Nodes）在图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组，即张量（tensor）。
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath

def _read32(bytestream):
  """定义dtype：numpy.uint32设定将高序字节存储在起始地址（高位编址）(‘>’/big endian)"""
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  # 按照制定dtype从字节流读取一个整数. frombuffer: Interpret a buffer as a 1-dimensional array.
  # print("!!!", numpy.frombuffer(bytestream.read(4), dtype=dt))
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    #print("_read32: ", magic)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    # 记录图片的数量=train-60000/test-10000
    num_images = _read32(bytestream)
    #print("num_images:", num_images)
    # 图片的rows/cols都是28，手写数字的图片
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    #print("rows:", rows)
    #print("cols:", cols)
    # 从流中获取所有图片的字节
    buf = bytestream.read(rows * cols * num_images)
    # 将字节流转成数组
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    # 一维数组转换成4维数组：图片索引/图片行/图片列/图片元素“灰度”。reshape：用于改变数组的形状/维度
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors. 将标签一维数组转成二位数组[imagesindex, onehot], onehot即lable对应位为1，其他为0"""
  # onehot标签: 一个长度为n的数组，只有一个元素是1.0，其他元素是0.0
  # 获取标签数组的长度. shape: 查看矩阵或者数组的维数，返回一个数组(比如维度是2，则数组长度是2)
  num_labels = labels_dense.shape[0]
  #print('dense_to_one_hot num_labels:', num_labels)
  # 生成一个一维等差数组(0-num_labels)，长度为num_labels, 并为每位*num_classes. arange函数用于创建等差数组
  index_offset = numpy.arange(num_labels) * num_classes
  #print('numpy.arange(num_labels):', numpy.arange(num_labels))
  #print('index_offset:', index_offset)
  # Return a new array of given shape and type, filled with zeros.
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  # flat: A 1-D iterator over the array. 多维数组按照1维做迭代
  # numpy.ravel() vs numpy.flatten(): 两者所要实现的功能是一致的（将多维数组降位一维）
  # 两者的区别在于返回拷贝（copy）还是返回视图（view），numpy.flatten()返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，而numpy.ravel()返回的是视图（view，也颇有几分C/C++引用reference的意味），会影响（reflects）原始矩阵。
  # index_offset代表image索引（即数组行），labels_dense则是标签值。使得labels_one_hot每行对应label的位置，值为1
  # 注：由于labels_dense本身就是一维的，所以可以不必ravel
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  #print('labels_one_hot', labels_one_hot)
  return labels_one_hot


def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    # 标签总数量，对应images数量
    num_items = _read32(bytestream)
    #print('num_items:', num_items)
    buf = bytestream.read(num_items)
    # 从流中获得标签数组
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels


class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):
    """Construct a DataSet.

    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      print('self._num_examples', self._num_examples)

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      print('images.shape[3]', images.shape[3])
      assert images.shape[3] == 1
      # NOTE 疑问：降维的目的还要再看看！
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      if dtype == tf.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    # NOTE 再看一遍！
    if fake_data:
      fake_image = [1] * 784
      print('fake_image:', fake_image)
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      # NOTE 再看一遍！
      print('next_batch return:', [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)])
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      # numpy.random.shuffle洗牌，用随机数填充数组
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    #print('self._images[start:end], self._labels[start:end]:', self._images[start:end], self._labels[start:end])
    return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=tf.float32):
  # 注意:不是class DataSet(object):
  class DataSets(object):
    pass
  data_sets = DataSets()

  if fake_data:
    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
    data_sets.train = fake()
    data_sets.validation = fake()
    data_sets.test = fake()
    return data_sets

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 5000

  local_file = maybe_download(TRAIN_IMAGES, train_dir)
  train_images = extract_images(local_file)

  local_file = maybe_download(TRAIN_LABELS, train_dir)
  train_labels = extract_labels(local_file, one_hot=one_hot)

  local_file = maybe_download(TEST_IMAGES, train_dir)
  test_images = extract_images(local_file)

  local_file = maybe_download(TEST_LABELS, train_dir)
  test_labels = extract_labels(local_file, one_hot=one_hot)

  # train_images中前一部分用于validation，后一部分用于train
  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]

  data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
  data_sets.validation = DataSet(validation_images, validation_labels,
                                 dtype=dtype)
  data_sets.test = DataSet(test_images, test_labels, dtype=dtype)

  return data_sets
