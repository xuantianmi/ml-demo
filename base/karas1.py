import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation

# https://www.cnblogs.com/LittleHann/p/6442161.html
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
    ])
"""
Same as:
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
"""