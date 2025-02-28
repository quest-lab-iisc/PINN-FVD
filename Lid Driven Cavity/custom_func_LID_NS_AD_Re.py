import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import scipy.io
from scipy.io import loadmat
from tensorflow import keras
from keras import layers
from keras import initializers

import time
import math
from utils import get_boundary_data

np.random.seed(1234)
tf.random.set_seed(1234)
tf.keras.backend.set_floatx('float32')

def get_ibc_and_inner_data(grid_size):

    # Boundary Points
    xdisc = np.linspace(start=0, stop=1., num=grid_size)
    ydisc = np.linspace(start=1., stop=0, num=grid_size)

    X, Y = np.meshgrid(xdisc, ydisc)
    grid_loc = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    # boundary conditions
    x_top = np.hstack((X[0, :][:, None], Y[0, :][:, None]))
    x_bottom = np.hstack((X[0, 1:-1][:, None], Y[-1, 1:-1][:, None]))
    x_left = np.hstack((X[1:, 0][:, None], Y[1:, 0][:, None]))
    x_right = np.hstack((X[1:, -1][:, None], Y[1:, 0][:, None]))

    u_top = np.ones(shape=(len(x_top), 1))
    v_top = np.zeros_like(u_top)

    u_bottom = np.zeros(shape=(len(x_bottom), 1))
    v_bottom = np.zeros_like(u_bottom)

    u_left = np.zeros(shape=(len(x_left), 1))
    v_left = np.zeros_like(u_left)

    u_right = np.zeros(shape=(len(x_right), 1))
    v_right = np.zeros_like(u_right)

    xb = np.vstack((x_top[:, 0:1], x_bottom[:, 0:1], x_left[:, 0:1], x_right[:, 0:1]))
    yb = np.vstack((x_top[:, 1:2], x_bottom[:, 1:2], x_left[:, 1:2], x_right[:, 1:2]))
    u_ob = np.vstack((u_top[:], u_bottom[:], u_left[:], u_right[:]))
    v_ob = np.vstack((v_top[:], v_bottom[:], v_left[:], v_right[:]))

    xd = grid_loc[:, 0:1]
    yd = grid_loc[:, 1:2]

    return xb, yb, xd, yd, u_ob, v_ob

class PdeModel:

    def __init__(self, inputs, outputs, nn_model, loss_fn,
                 optimizer, metrics, parameters):
        
        self.nn_model = nn_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer['optimizer']

        self.xin = tf.constant(inputs['xin'], dtype=tf.float32)
        self.yin = tf.constant(inputs['yin'], dtype=tf.float32)
        self.xb = tf.constant(inputs['xb'], dtype=tf.float32)
        self.yb = tf.constant(inputs['yb'], dtype=tf.float32)

        self.ub = tf.constant(outputs['ub'], dtype=tf.float32)
        self.vb = tf.constant(outputs['vb'], dtype=tf.float32)
        self.Re = tf.constant(parameters['Re'], dtype=tf.float32)
        self.g_s = tf.constant(parameters['grid_size'],dtype=tf.int32)
        
        # Loss tracker
        self.std_loss_tracker = metrics['std_loss']
        self.residual_loss_tracker = metrics['residual_loss']
        self.loss_tracker = metrics['loss']    
    
    @tf.function
    def pde_residual(self, training=True):

        with tf.GradientTape(persistent=True) as outer_tape:
            outer_tape.watch([self.xin, self.yin])
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch([self.xin, self.yin])
                u, v, P = self.nn_model([self.xin, self.yin], training=True)

            # first order derivatives wrt x
            ux = inner_tape.gradient(u, self.xin)
            vx = inner_tape.gradient(v, self.xin)
            px = inner_tape.gradient(P, self.xin)

            # first order derivatives wrt y
            uy = inner_tape.gradient(u, self.yin)
            vy = inner_tape.gradient(v, self.yin)
            py = inner_tape.gradient(P, self.yin)

        # second order derivatives wrt x
        uxx = outer_tape.gradient(ux, self.xin)
        vxx = outer_tape.gradient(vx, self.xin)
        pxx = outer_tape.gradient(px, self.xin)

        # second order derivatives wrt y
        uyy = outer_tape.gradient(uy, self.yin)
        vyy = outer_tape.gradient(vy, self.yin)
        pyy = outer_tape.gradient(py, self.yin)

        # Momentum equation calculation
        fx = u * ux + v * uy + px - (1 / self.Re) * (uxx + uyy)
        fy = u * vx + v * vy + py - (1 / self.Re) * (vxx + vyy)

        # Continuity equation
        div_u = ux + vy

        return fx, fy, div_u#, Pe
    
        
    @tf.function
    def train_step(self):

        # Loss 
        with tf.GradientTape(persistent=True) as tape:

            u_pred, v_pred, _ = self.nn_model([self.xb, self.yb], training=True)
            u_loss = self.loss_fn(self.ub, u_pred)
            v_loss = self.loss_fn(self.vb, v_pred)
            std_loss = u_loss + v_loss 
            fx, fy, div_u = self.pde_residual()
            fx_loss = tf.reduce_mean(tf.square(fx))
            fy_loss = tf.reduce_mean(tf.square(fy))
            div_loss = tf.reduce_mean(tf.square(div_u))
            residual_loss = fx_loss + fy_loss + div_loss 
            loss = std_loss + residual_loss

        grads = tape.gradient(loss, self.nn_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn_model.trainable_weights))

        self.std_loss_tracker.update_state(std_loss)
        self.residual_loss_tracker.update_state(residual_loss)
        self.loss_tracker.update_state(loss)
    
        return {"std_loss": self.std_loss_tracker.result(), "residual_loss": self.residual_loss_tracker.result(),"loss": self.loss_tracker.result()}#, grads_residual, grads_boundary

    def reset_metrics(self):
        self.std_loss_tracker.reset_state()
        self.residual_loss_tracker.reset_state()
        self.loss_tracker.reset_state()

    def run(self, epochs, proj_name, verbose_freq=1000, error_freq=5000):

        self.reset_metrics()
        history = {"std_loss": [], "residual_loss": [], "loss": []}
       
        start_time = time.time()

        for epoch in range(epochs):

            logs = self.train_step()
            tae = time.time() - start_time
            if (epoch + 1) % verbose_freq == 0:
                print(f'''Epoch:{epoch + 1}/{epochs}''')
                for key, value in logs.items():
                    history[key].append(value.numpy())
                    print(f"{key}: {value:.4f} {epoch}", end="")
                print(f"Time: {tae / 60:.4f}min")
    
            if (epoch + 1) % error_freq == 0:
                self.get_error(epoch + 1, proj_name)
        
        odata = pd.DataFrame(history)
        lo_pl = np.log(odata['loss'])
        DF = pd.DataFrame(lo_pl)
        
 
        # save the dataframe as a csv file
        DF.to_csv("Results/Re_1000_AD_loss.csv")
        
       
        return history

    def predictions(self, inputs):
        return self.nn_model(inputs, training=False)
    
    def get_error(self, step, name):

        # Interpolation to get same grid location for U,V,P
        test_data = loadmat('grid_data.mat')
        xp_test = test_data['XP'].reshape((-1, 1))
        xu_test = test_data['XU'].reshape((-1, 1))
        xv_test = test_data['XV'].reshape((-1, 1))
        yp_test = test_data['YP'].reshape((-1, 1))
        yu_test = test_data['YU'].reshape((-1, 1))
        yv_test = test_data['YV'].reshape((-1, 1))

        u_test_data = [xu_test, yu_test]
        v_test_data = [xv_test, yv_test]
        p_test_data = [xp_test, yp_test]

        u_test, _, _ = self.predictions(u_test_data)
        _, v_test, _ = self.predictions(v_test_data)
        _, _, p_test = self.predictions(p_test_data)

        # Actual Data
        true_data = loadmat('Re_1000.mat')
        u_data = true_data['u'][4:].reshape(test_data['XU'].shape, order='F')
        v_data = true_data['v'][4:].reshape(test_data['XV'].shape, order='F')
        p_data = true_data['P'][4:].reshape(test_data['XP'].shape, order='F')

        # Relative error 
        u_er = (np.abs(u_data - u_test.numpy().reshape(test_data['XU'].shape)))/(1+np.abs(u_data))
        v_er = (np.abs(v_data - v_test.numpy().reshape(test_data['XV'].shape)))/(1+np.abs(v_data))
        p_er = (np.abs(p_data - p_test.numpy().reshape(test_data['XP'].shape)))/(1+np.abs(p_data))
        
        # Mean and Variance 
        mean_u = np.mean(u_er)
        var_u  = np.var(u_er)
        mean_v = np.mean(v_er)
        var_v  = np.var(v_er)
        mean_p = np.mean(p_er)
        var_p  = np.var(p_er)

        print(f"Mean_u:{mean_u},Var_u:{var_u}")
        print(f"Mean_v:{mean_v},Var_v{var_v}")
        print(f"Mean_p:{mean_p},Var_p:{var_p}")
        

  
