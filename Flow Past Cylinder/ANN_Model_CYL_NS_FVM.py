import tensorflow as tf
from tensorflow import keras
from keras import layers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.initializers import GlorotNormal, GlorotUniform
from keras.utils import plot_model
from scipy.io import loadmat

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from custom_func_CYL_NS_FVM import PdeModel, get_ibc_and_inner_data

np.random.seed(1234)
tf.random.set_seed(1234)
tf.keras.backend.set_floatx('float32')

# Data PreProcessing

x_lim = 20.
y_lim = 3.
grid_size_x = 256
grid_size_y = 256

xd, yd, xp, yp, uc, vc,  Xint, Yint, Xout, Yout, xb_vel, yb_vel, u_ob, v_ob, xb_pre, yb_pre, p_right, indices, xb_top, xb_top_1, yb_top, yb_top_1, xb_bottom, xb_bottom_1, yb_bottom, yb_bottom_1, xb_right, xb_right_1, yb_right, yb_right_1, xb_vel_tb, yb_vel_tb, v_ob_tb, ub_dir, sign = get_ibc_and_inner_data(x_lim=x_lim, y_lim=y_lim, grid_size_x=grid_size_x, grid_size_y=grid_size_y)

ivals = {'xin': xd, 'yin': yd, 'xp': xp, 'yp': yp, 'x_inter': Xint, 
         'y_inter': Yint, 'x_exter': Xout, 'y_exter': Yout, 
         'xb_vel': xb_vel, 'yb_vel': yb_vel, 'xb_pre':xb_pre, 'yb_pre':yb_pre, 
         'xb_top': xb_top, 'xb_top_1': xb_top_1, 'yb_top': yb_top, 'yb_top_1': yb_top_1,
         'xb_bottom': xb_bottom, 'xb_bottom_1': xb_bottom_1, 'yb_bottom': yb_bottom, 'yb_bottom_1': yb_bottom_1,
         'xb_right': xb_bottom, 'xb_right_1': xb_bottom_1, 'yb_right': yb_bottom, 'yb_right_1': yb_bottom_1,
         'xb_vel_tb': xb_vel_tb, 'yb_vel_tb': yb_vel_tb}
ovals = {'uc': uc, 'vc': vc, 'u_ob': u_ob, 'v_ob': v_ob, 'p_right': p_right, 'v_ob_tb': v_ob_tb, 'ub_dir': ub_dir, 'sign': sign}
parameters = {'Re': 50, 'grid_size_x': grid_size_x, 'grid_size_y': grid_size_y, 'indices': indices}

# Building Model
initializer = GlorotNormal(1234)

input_x = keras.Input(shape=(1,))
input_x_rescale = layers.Rescaling(scale=2/x_lim, offset=-1)(input_x)
input_y = keras.Input(shape=(1,))
input_y_rescale = layers.Rescaling(scale=2/y_lim, offset=-1)(input_y)
# Initially 50 - 5
x = tf.keras.layers.Concatenate(name='concatenate_layer')([input_x_rescale, input_y_rescale])
x = tf.keras.layers.Dense(units = 32,activation='swish',kernel_initializer=initializer,name = "layer_1")(x)
x = tf.keras.layers.Dense(units = 32,activation='swish',kernel_initializer=initializer,name = "layer_2")(x)
x = tf.keras.layers.Dense(units = 32,activation='swish',kernel_initializer=initializer,name = "layer_3")(x)
x = tf.keras.layers.Dense(units = 32,activation='swish',kernel_initializer=initializer,name = "layer_4")(x)
x = tf.keras.layers.Dense(units = 32,activation='swish',kernel_initializer=initializer,name = "layer_5")(x)


output_p      = tf.keras.layers.Dense(units = 1,use_bias=False,kernel_initializer=initializer)(x)
output_u      = tf.keras.layers.Dense(units = 1,use_bias=False,kernel_initializer=initializer)(x)
output_v      = tf.keras.layers.Dense(units = 1,use_bias=False,kernel_initializer=initializer)(x)

model = keras.Model([input_x , input_y ],[output_u,output_v,output_p])

model.summary()
 
initial_learning_rate = 0.5e-3
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
     initial_learning_rate=initial_learning_rate,
     decay_steps=10000,
     decay_rate=0.8,
     staircase=False)

#step = ops.array(0)
step = tf.Variable(0, trainable=False, dtype=tf.int64)
boundaries = [10000, 20000, 30000, 40000, 50000]
values = [0.5e-3, 0.5e-4, 0.5e-5, 0.5e-6, 0.5e-7, 0.5e-8]
#values = [1e-5, 1e-5, 1e-5, 1e-6]
learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)

# Later, whenever we perform an optimization step, we pass in the step.
learning_rate = learning_rate_fn(step)
learning_rate_geo = 1e-4

# Training the model
loss_fn = keras.losses.MeanSquaredError()

optimizer = {"optimizer":keras.optimizers.Adam(learning_rate = initial_learning_rate)}

metrics = {"std_loss": keras.metrics.Mean(name='std_loss'),
           "residual_loss": keras.metrics.Mean(name='residual_loss'),
           "std_loss_inside": keras.metrics.Mean(name='std_loss_inside')}

cm = PdeModel(inputs=ivals, outputs=ovals, nn_model=model, 
              loss_fn=loss_fn, optimizer=optimizer, metrics=metrics, parameters=parameters)


history = cm.run(epochs=50000, proj_name="PINNS_CYL_Re_50_ICCS",
                 verbose_freq=1000, plot_freq=5000)


# Save trained model
model.save('saved_model_ICCS/model_ICCS_Final.keras')
