from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
import tensorflow as tf

import get_data

tf.python.control_flow_ops = tf

n_epochs = 4
n_samples_per_epoch = 20000
n_validation_samples = 6400
learning_rate = 1e-4
relu = 'relu'

# This model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

# 5 convolution and max-pooling layers
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

# 5 fully connected layers
model.add(Dense(1164))
model.add(Activation(relu))

model.add(Dense(100))
model.add(Activation(relu))

model.add(Dense(50))
model.add(Activation(relu))

model.add(Dense(10))
model.add(Activation(relu))

model.add(Dense(1))

model.summary()
model.compile(loss="mse", optimizer=Adam(learning_rate))

# Generators for training and validation
train_gen = get_data.generate_next_batch()
validation_gen = get_data.generate_next_batch()

model.fit_generator(
    train_gen, samples_per_epoch=n_samples_per_epoch, nb_epoch=n_epochs,
    validation_data=validation_gen, nb_val_samples=n_validation_samples, verbose=1
)

model.save('model.h5')
