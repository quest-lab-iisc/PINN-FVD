# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.initializers import GlorotNormal
from keras.utils import plot_model

from scipy.io import loadmat

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from custom_func_LID_NS_AD_Re import PdeModel, get_ibc_and_inner_data

np.random.seed(1234)
tf.random.set_seed(1234)
tf.keras.backend.set_floatx('float32')

# Data PreProcessing
grid_size = 100

xb, yb, xd, yd, u_ob, v_ob = get_ibc_and_inner_data(grid_size=grid_size)

ivals = {'xin': xd, 'yin': yd, 'xb': xb, 'yb': yb}
ovals = {'ub': u_ob, 'vb': v_ob}
parameters = {'Re': 1000,'grid_size': grid_size}

# Initializer
initializer = GlorotNormal(1234)

# Input layers
input_x = keras.Input(shape=(1,))
input_y = keras.Input(shape=(1,))
x = tf.keras.layers.Concatenate(name='concatenate_layer')([input_x, input_y])
x = tf.keras.layers.Rescaling(scale=2, offset=-1)(x)

# Hidden layers
x = tf.keras.layers.Dense(units = 300,activation='swish',kernel_initializer=initializer,name = "layer_1")(x)
x = tf.keras.layers.Dense(units = 300,activation='swish',kernel_initializer=initializer,name = "layer_2")(x)
x = tf.keras.layers.Dense(units = 300,activation='swish',kernel_initializer=initializer,name = "layer_3")(x)
x = tf.keras.layers.Dense(units = 300,activation='swish',kernel_initializer=initializer,name = "layer_4")(x)
x = tf.keras.layers.Dense(units = 300,activation='swish',kernel_initializer=initializer,name = "layer_5")(x)

# Output layers
output_p      = tf.keras.layers.Dense(units = 1,use_bias=False,kernel_initializer=initializer)(x)
output_u      = tf.keras.layers.Dense(units = 1,use_bias=False,kernel_initializer=initializer)(x)
output_v      = tf.keras.layers.Dense(units = 1,use_bias=False,kernel_initializer=initializer)(x)

# Build model
model = keras.Model([input_x , input_y ],[output_u,output_v,output_p])

model.summary()

initial_learning_rate = 1e-3
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
     initial_learning_rate=initial_learning_rate,
     decay_steps=10000,
     decay_rate=0.9,
     staircase=False)
loss_fn = keras.losses.MeanSquaredError()
optimizer = {"optimizer":keras.optimizers.Adam(learning_rate = lr_schedule)}
metrics = {"std_loss": keras.metrics.Mean(name='std_loss'),
           "residual_loss": keras.metrics.Mean(name='residual_loss'),
           "loss": keras.metrics.Mean(name='loss')}

cm = PdeModel(inputs=ivals, outputs=ovals, nn_model=model, 
              loss_fn=loss_fn, optimizer=optimizer, metrics=metrics, parameters=parameters)

# Training the model
history = cm.run(epochs=100000, proj_name="ICCS_AD",
                 verbose_freq=1000, error_freq=5000)


# Save trained model
model.save('saved_model/ICCS_my_model_Re_1000_layers_5_AD_lrs.keras')
