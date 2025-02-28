import tensorflow as tf
from tensorflow import keras
from keras import layers


import numpy as np

from keras.initializers import GlorotUniform
from KF_VPde_FVM import PdeModel
from utils import get_ibc_and_inner_data

import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

np.random.seed(1234)
tf.random.set_seed(1234)

# Data PreProcessing
grid_size = 64

xb, yb, xd, yd, u, v, p = get_ibc_and_inner_data(start=[-0.5, -0.5], stop=[1., 1.5], grid_size=grid_size,
                                                 nue=1/40)

ivals = {'xin': xd, 'yin': yd, 'xb': xb, 'yb': yb}
ovals = {'ub': u, 'vb': v, 'pb': p}
parameters = {'nue': 0.025}

initializer = GlorotUniform(1234)


# Define the input layers
input1 = keras.Input(shape=(1,), name='X_layer')
rescale_input1 = layers.Rescaling(scale=2/1.5, offset=-0.5/1.5)(input1)
input2 = keras.Input(shape=(1,), name='Y_layer')
rescale_input2 = layers.Rescaling(scale=1, offset=-0.5)(input2)
x = layers.Concatenate()([rescale_input1, rescale_input2])
x = layers.Dense(units=50, kernel_initializer=initializer, activation='swish')(x)
x = layers.Dense(units=50, kernel_initializer=initializer, activation='swish')(x)
x = layers.Dense(units=50, kernel_initializer=initializer, activation='swish')(x)
x = layers.Dense(units=50, kernel_initializer=initializer, activation='swish')(x)
x = layers.Dense(units=50, kernel_initializer=initializer, activation='swish')(x)
ou = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer)(x)
ov = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer)(x)
op = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer)(x)

model = keras.Model([input1, input2], [ou, ov, op])
model.summary()


initial_learning_rate = 1e-3

# # Training the model
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
model_dict = {"nn_model": model}

metrics = {"loss": keras.metrics.Mean(name='loss'),
           "boundary_loss": keras.metrics.Mean(name='std_loss'),
           "residual_loss": keras.metrics.Mean(name='residual_loss'),
           "u_loss": keras.metrics.Mean(name='u_loss'),
           "v_loss": keras.metrics.Mean(name='v_loss'),
           "p_loss": keras.metrics.Mean(name='p_loss')
           }

cm = PdeModel(inputs=ivals, outputs=ovals, get_models=model_dict, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics,
              parameters=parameters, batches=1)

epochs = 2000
vf = 100
pf = 500

log_dir = 'output_FVD/'
history = cm.run(epochs=epochs, log_dir=log_dir,
                 wb=False, verbose_freq=vf, plot_freq=pf)


# Model Saving
cm.nn_model.save('output_model/FVD_new.keras')