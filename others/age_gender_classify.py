import keras as k
import keras.applications.imagenet_utils
import keras.optimizers
import keras.preprocessing.image
import numpy as np
from keras.callbacks import TensorBoard
from keras import Sequential, Model, Input
from keras.applications import InceptionV3, VGG16, ResNet50, VGG19
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, InputLayer, Dense, Reshape, \
    Conv3D, Conv1D, Flatten, DepthwiseConv2D, SeparableConvolution2D, SeparableConv2D, GlobalAveragePooling2D, \
    concatenate, GlobalAveragePooling1D, AveragePooling1D
from keras.preprocessing.image import  ImageDataGenerator
from keras.utils import to_categorical
from matplotlib import  pyplot as plt
import tensorflow as tf

def lrn(x):
    return tf.nn.lrn(x,name='first')
def cnn_network():
    i=Input((227,227,3))
    x=Conv2D(96, (7, 7),activation='relu')(i)
    x=MaxPooling2D(pool_size=(3, 3),strides=2)(x)
    x=Activation(activation=lrn)(x)
    x=SeparableConv2D(filters=256,kernel_size=(3,3),depth_multiplier=2)(x)
    x=SeparableConv2D(filters=384,kernel_size=(3,3),depth_multiplier=1)(x)
    x=MaxPooling2D(pool_size=(3,3))(x)
    # x=GlobalAveragePooling2D()(x)
    x=Flatten()(x)
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.3)(x)
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.3)(x)
    sex=Dense(2,activation='softmax')(x)
    age=Dense(8,activation='softmax')(x)
    model =Model(inputs=i,outputs=[sex,age])
    model.summary()
    return model

cnn_network()