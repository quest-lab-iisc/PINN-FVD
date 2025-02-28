
import tensorflow as tf
import scipy.io
from scipy.io import loadmat
from scipy.interpolate import griddata
from tensorflow import keras
from keras import layers
from keras import initializers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import math
from utils import get_boundary_data

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

tf.random.set_seed(1234)
tf.keras.backend.set_floatx('float32')

def get_ibc_and_inner_data(x_lim, y_lim, grid_size_x, grid_size_y):

    x_center = 1.5
    y_center = 1.5
    radius   = 0.5

    # Coarser mesh
    xcdisc   = np.linspace(start=0., stop=x_lim, num=grid_size_x)
    ycdisc   = np.linspace(start=y_lim, stop=0., num=grid_size_y)

    X, Y     = np.meshgrid(xcdisc, ycdisc)

    grid_loc = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    xd       = grid_loc[:,0:1]
    yd       = grid_loc[:,1:2]

    # # points on the circular perimeter
    theta    = np.linspace(start=0., stop=2*np.pi, num=25)
    xp       = (x_center + radius * np.cos(theta))
    yp       = (y_center + radius * np.sin(theta))
    xp_1     = xp[:, None]

    up       = np.zeros_like(xp)
    vp       = np.zeros_like(xp)

    # Interior points
 
    radius = 0.5 - min(x_lim/(float(grid_size_x)),y_lim/(float(grid_size_y)))
    radius_1 = 0.5 + min(x_lim/(float(grid_size_x)),y_lim/(float(grid_size_y)))

    R        = np.linalg.norm((X-x_center,Y-y_center),axis=0)
 
    indices = np.argwhere(R<=radius)
    Xint     = X.copy()
    Yint     = Y.copy()
    Xint[R>radius] = np.nan
    Yint[R>radius] = np.nan
    Xint     = Xint[np.logical_not(np.isnan(Xint))]
    Yint     = Yint[np.logical_not(np.isnan(Yint))]


    # Exterior points
    Xout      = X.copy()
    Yout      = Y.copy()
    Xout[R<=radius] = 0.0

    condition = (radius_1>=R)&(R>radius)
    Xout[condition] = 1.0
    Xout[R>radius_1] = 1.0


    # Signed distance function

    sign = 1/(np.linalg.norm((X-x_center,Y-y_center), axis=0)**2 - radius**2)
    sign = np.maximum(sign,0.0)


    min_value = np.min(sign)
    max_value = np.max(sign)

    sign = (sign - min_value) / (max_value - min_value)
    

    # Boundary conditions
    x_top     = np.hstack((X[0, :][:, None], Y[0, :][:, None]))
    x_top_1   = np.hstack((X[1, :][:, None], Y[1, :][:, None]))
    x_bottom  = np.hstack((X[0, 1:-1][:, None], Y[-1, 1:-1][:, None]))
    x_bottom_1= np.hstack((X[grid_size_y-2, 1:-1][:, None], Y[grid_size_y-2, 1:-1][:, None]))
    x_left    = np.hstack((X[1:, 0][:, None], Y[1:, 0][:, None]))
    x_right   = np.hstack((X[1:, -1][:, None], Y[1:, 0][:, None]))
    x_right_1 = np.hstack((X[1:, grid_size_x-2][:, None], Y[1:, 0][:, None]))

    u_left    = np.ones(shape=(len(x_left), 1))
    #u_left.fill(2)
    v_left    = np.zeros_like(u_left)

    v_top     = np.zeros_like(x_top)

    v_bottom  = np.zeros_like(x_bottom)

    #p_right  = np.ones(shape=(len(x_right), 1))
    p_right  = np.zeros_like(x_right)

    xb_vel   = x_left[:, 0:1]
    yb_vel   = x_left[:, 1:2]
    u_ob     = u_left[:]
    v_ob     = v_left[:]

    xb_vel_tb = np.vstack((x_top[:, 0:1], x_bottom[:, 0:1]))
    yb_vel_tb = np.vstack((x_top[:, 1:2], x_bottom[:, 1:2]))
    v_ob_tb   = np.vstack((v_top[:], v_bottom[:]))

    xb_pre   = x_right[:, 0:1]
    yb_pre   = x_right[:, 1:2]

    # Top 
    xb_top   = x_top[:, 0:1]
    xb_top_1 = x_top_1[:, 0:1]
    yb_top   = x_top[:, 1:2]
    yb_top_1 = x_top_1[:, 1:2]

    # Bottom
    xb_bottom = x_bottom[:, 0:1]
    xb_bottom_1 = x_bottom_1[:, 0:1] 
    yb_bottom = x_bottom[:, 1:2]
    yb_bottom_1 = x_bottom_1[:, 1:2]

    # Right
    xb_right = x_right[:, 0:1]
    xb_right_1 = x_right_1[:, 0:1]
    yb_right = x_right[:, 1:2]
    yb_right_1 = x_right_1[:, 1:2]

    # Plotting
    u_bound_dir = np.vstack((np.ones_like(x_left[:, 0:1]) * 1.0, np.zeros_like(xp_1)))

    Xout = Xout[:254,:254]
   
    return  xd, yd, xp, yp, up, vp,  Xint, Yint, Xout, Yout, xb_vel, yb_vel, u_ob, v_ob, xb_pre, yb_pre, p_right, indices, xb_top, xb_top_1, yb_top, yb_top_1, xb_bottom, xb_bottom_1, yb_bottom, yb_bottom_1, xb_right, xb_right_1, yb_right, yb_right_1, xb_vel_tb, yb_vel_tb, v_ob_tb, u_bound_dir, sign


class PdeModel:

    def __init__(self, inputs, outputs, nn_model, loss_fn,
                 optimizer, metrics, parameters):
        
        self.nn_model = nn_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer['optimizer']
        # self.optimizer_1 = optimizer['optimizer_1']
        # self.optimizer_2 = optimizer['optimizer_2']

        # Domain points
        self.xin = tf.Variable(inputs['xin'], dtype=tf.float32)
        self.yin = tf.Variable(inputs['yin'], dtype=tf.float32)

        # Points on the cylinder
        self.xp = tf.constant(inputs['xp'], dtype=tf.float32)
        self.yp = tf.constant(inputs['yp'], dtype=tf.float32)

        # Points inside the circle
        self.x_inter = tf.constant(inputs['x_inter'], dtype=tf.float32)
        self.y_inter = tf.constant(inputs['y_inter'], dtype=tf.float32)

        # Interior points velocity
        self.u_int = np.zeros_like(self.x_inter)
        self.v_int = np.zeros_like(self.x_inter)

        # Points outside the circle
        self.x_exter = tf.constant(inputs['x_exter'], dtype=tf.float32)
        self.y_exter = tf.constant(inputs['y_exter'], dtype=tf.float32)

        # Boundary points
        self.xb_vel  = tf.constant(inputs['xb_vel'], dtype=tf.float32)
        self.yb_vel  = tf.constant(inputs['yb_vel'], dtype=tf.float32)
        self.xb_vel_tb  = tf.constant(inputs['xb_vel_tb'], dtype=tf.float32)
        self.yb_vel_tb  = tf.constant(inputs['yb_vel_tb'], dtype=tf.float32)
        self.xb_pre  = tf.constant(inputs['xb_pre'], dtype=tf.float32)
        self.yb_pre  = tf.constant(inputs['yb_pre'], dtype=tf.float32)

        # Top and Bottom Boundary points
        self.xb_top = tf.constant(inputs['xb_top'], dtype=tf.float32)
        self.yb_top = tf.constant(inputs['yb_top'], dtype=tf.float32)
        self.xb_top_1 = tf.constant(inputs['xb_top_1'], dtype=tf.float32)
        self.yb_top_1 = tf.constant(inputs['yb_top_1'], dtype=tf.float32)

        self.xb_bottom = tf.constant(inputs['xb_bottom'], dtype=tf.float32)
        self.yb_bottom = tf.constant(inputs['yb_bottom'], dtype=tf.float32)
        self.xb_bottom_1 = tf.constant(inputs['xb_bottom_1'], dtype=tf.float32)
        self.yb_bottom_1 = tf.constant(inputs['yb_bottom_1'], dtype=tf.float32)

        self.xb_right = tf.constant(inputs['xb_right'], dtype=tf.float32)
        self.yb_right = tf.constant(inputs['yb_right'], dtype=tf.float32)
        self.xb_right_1 = tf.constant(inputs['xb_right_1'], dtype=tf.float32)
        self.yb_right_1 = tf.constant(inputs['yb_right_1'], dtype=tf.float32)

        # Points on the cylinder
        self.uc = tf.constant(outputs['uc'], dtype=tf.float32)
        self.vc = tf.constant(outputs['vc'], dtype=tf.float32)

        self.outputs = outputs
        # Points on the  boundary
        self.u_ob = tf.constant(outputs['u_ob'], dtype=tf.float32)
        self.v_ob = tf.constant(outputs['v_ob'], dtype=tf.float32)
        self.v_ob_tb = tf.constant(outputs['v_ob_tb'], dtype=tf.float32)
        self.p_right = tf.constant(outputs['p_right'], dtype=tf.float32)
        self.sign = tf.constant(outputs['sign'], dtype=tf.float32)

        self.u_ob_t = tf.zeros_like(self.xb_top, dtype=tf.float32)
        self.v_ob_t = tf.zeros_like(self.xb_top, dtype=tf.float32)

        self.u_ob_b = tf.zeros_like(self.xb_bottom, dtype=tf.float32)
        self.v_ob_b = tf.zeros_like(self.xb_bottom, dtype=tf.float32)

        self.u_ob_r = tf.zeros_like(self.xb_right, dtype=tf.float32)

        self.Re = tf.constant(parameters['Re'], dtype=tf.float32)
        self.g_s_x = tf.constant(parameters['grid_size_x'],dtype=tf.int32)
        self.g_s_y = tf.constant(parameters['grid_size_y'],dtype=tf.int32)
        self.indices = tf.Variable(parameters['indices'],dtype=tf.int64)

        # Loss tracker
        self.std_loss_tracker = metrics['std_loss']
        self.residual_loss_tracker = metrics['residual_loss']
        self.std_loss_inside_tracker = metrics['std_loss_inside']

        self.del_x = 20.0/(float(self.g_s_x))
        self.del_y = 3.0/(float(self.g_s_y))
        
        
    @tf.function
    def pde_residual(self):

        u, v, P = self.nn_model([self.xin, self.yin], training=True)
        u = tf.reshape(u,(self.g_s_y, self.g_s_x))
        v = tf.reshape(v,(self.g_s_y, self.g_s_x))
        P = tf.reshape(P,(self.g_s_y, self.g_s_x))
        u = tf.cast(u,dtype='float32')
        v = tf.cast(v,dtype='float32')
        P = tf.cast(P,dtype='float32')

        fe   =  (0.5*((u[1:self.g_s_y-1,1:self.g_s_x-1]+u[1:self.g_s_y-1,2:self.g_s_x])/self.del_x))
        fw   =  (0.5*((u[1:self.g_s_y-1,1:self.g_s_x-1]+u[1:self.g_s_y-1,0:self.g_s_x-2])/self.del_x)) 
        fn   =  (0.5*((v[1:self.g_s_y-1,1:self.g_s_x-1]+v[0:self.g_s_y-2,1:self.g_s_x-1])/self.del_y)) 
        fs   =  (0.5*((v[1:self.g_s_y-1,1:self.g_s_x-1]+v[2:self.g_s_y,1:self.g_s_x-1])/self.del_y))    

        # X-Momentum Equation 

        fx_x =  ((tf.multiply(u[1:self.g_s_y-1,1:self.g_s_x-1],tf.nn.relu(fe))-tf.multiply(u[1:self.g_s_y-1,2:self.g_s_x],tf.nn.relu(-fe)))
                   -(tf.multiply(u[1:self.g_s_y-1,0:self.g_s_x-2],tf.nn.relu(fw))-tf.multiply(u[1:self.g_s_y-1,1:self.g_s_x-1],tf.nn.relu(-fw)))
                   +(tf.multiply(u[1:self.g_s_y-1,1:self.g_s_x-1],tf.nn.relu(fn))-tf.multiply(u[0:self.g_s_y-2,1:self.g_s_x-1],tf.nn.relu(-fn)))
                   -(tf.multiply(u[2:self.g_s_y,1:self.g_s_x-1],tf.nn.relu(fs))-tf.multiply(u[1:self.g_s_y-1,1:self.g_s_x-1],tf.nn.relu(-fs)))
                   +(1/(2*self.del_x))*(P[1:self.g_s_y-1,2:self.g_s_x]-P[1:self.g_s_y-1,0:self.g_s_x-2]))
        
        fx_y = (1/self.Re)*((1/(self.del_x*self.del_x))*(u[1:self.g_s_y-1,2:self.g_s_x]-2*u[1:self.g_s_y-1,1:self.g_s_x-1]+u[1:self.g_s_y-1,0:self.g_s_x-2])
                                  +(1/(self.del_y*self.del_y))*(u[2:self.g_s_y,1:self.g_s_x-1]-2*u[1:self.g_s_y-1,1:self.g_s_x-1]+u[0:self.g_s_y-2,1:self.g_s_x-1]))
        
        fx = fx_x - (1.0)*fx_y
        
        # Y-Momentum Equation

        fy_x = ((tf.multiply(v[1:self.g_s_y-1,1:self.g_s_x-1],tf.nn.relu(fe))-tf.multiply(v[1:self.g_s_y-1,2:self.g_s_x],tf.nn.relu(-fe)))
                   -(tf.multiply(v[1:self.g_s_y-1,0:self.g_s_x-2],tf.nn.relu(fw))-tf.multiply(v[1:self.g_s_y-1,1:self.g_s_x-1],tf.nn.relu(-fw)))
                   +(tf.multiply(v[1:self.g_s_y-1,1:self.g_s_x-1],tf.nn.relu(fn))-tf.multiply(v[0:self.g_s_y-2,1:self.g_s_x-1],tf.nn.relu(-fn)))
                   -(tf.multiply(v[2:self.g_s_y,1:self.g_s_x-1],tf.nn.relu(fs))-tf.multiply(v[1:self.g_s_y-1,1:self.g_s_x-1],tf.nn.relu(-fs)))
                   +(1/(2*self.del_y))*(P[0:self.g_s_y-2,1:self.g_s_x-1]-P[2:self.g_s_y,1:self.g_s_x-1]))
        
        fy_y = (1/self.Re)*((1/(self.del_x*self.del_x))*(v[1:self.g_s_y-1,2:self.g_s_x]-2*v[1:self.g_s_y-1,1:self.g_s_x-1]+v[1:self.g_s_y-1,0:self.g_s_x-2])
                                 +(1/(self.del_y*self.del_y))*(v[2:self.g_s_y,1:self.g_s_x-1]-2*v[1:self.g_s_y-1,1:self.g_s_x-1]+v[0:self.g_s_y-2,1:self.g_s_x-1]))
        
        fy = fy_x - (1.0)*fy_y

        # Continuity

        div_u = ((1/(2*self.del_x))*(u[1:self.g_s_y-1,2:self.g_s_x]-u[1:self.g_s_y-1,0:self.g_s_x-2]) + (1/(2*self.del_y))*(v[0:self.g_s_y-2,1:self.g_s_x-1]-v[2:self.g_s_y,1:self.g_s_x-1]))

        fx = tf.multiply(self.x_exter,fx)
        fy = tf.multiply(self.x_exter,fy)
        div_u = tf.multiply(self.x_exter,div_u)

        return fx, fy, div_u
        
    @tf.function
    def train_step(self):

        ## Predictor Step
        with tf.GradientTape(persistent=True) as tape:

            # Velocity Boundary
            u_pred, v_pred, _ = self.nn_model([self.xb_vel, self.yb_vel], training=True)
            u_loss = self.loss_fn(self.u_ob, u_pred)
            v_loss = self.loss_fn(self.v_ob, v_pred)

            _ , v_pred_tb, _ = self.nn_model([self.xb_vel_tb, self.yb_vel_tb], training=True)
            v_loss_tb = self.loss_fn(self.v_ob_tb, v_pred_tb)

            #Top,Bottom and Right boundary
            u_pred_t, _ , _ = self.nn_model([self.xb_top, self.yb_top], training=True)
            u_pred_t_1, _ , _ = self.nn_model([self.xb_top_1, self.yb_top_1], training=True)
            u_loss_t = self.loss_fn(self.u_ob_t, (u_pred_t-u_pred_t_1)/(self.del_y))
            
            u_pred_b, _ , _ = self.nn_model([self.xb_bottom, self.yb_bottom], training=True)
            u_pred_b_1, _ , _ = self.nn_model([self.xb_bottom_1, self.yb_bottom_1], training=True)
            u_loss_b = self.loss_fn(self.u_ob_b, (u_pred_b-u_pred_b_1)/(self.del_y))


            _ , v_pred_r , _ = self.nn_model([self.xb_right, self.yb_right], training=True)
  
            v_loss_r = self.loss_fn(self.u_ob_r, v_pred_r)


            # Cylinder boundary 
            u_c, v_c, _ = self.nn_model([self.xp, self.yp], training=True)
            u_loss_cylinder = self.loss_fn(self.uc, u_c)
            v_loss_cylinder = self.loss_fn(self.vc, v_c) 

            # Total boundary loss
            std_loss = (u_loss + v_loss + v_loss_tb + v_loss_r) + (1)*(u_loss_t + u_loss_b) + (1)*(u_loss_cylinder + v_loss_cylinder)   #+ u_loss_r #+ v_loss_r 
            # Residual loss
            fx, fy, div_u = self.pde_residual()
            fx_loss = tf.reduce_mean(tf.square(fx))
            fy_loss = tf.reduce_mean(tf.square(fy))
            div_loss = tf.reduce_mean(tf.square(div_u))
            residual_loss = fx_loss + fy_loss + (100)*div_loss 

            # Inside the cylinder
            u_i, v_i, _ = self.nn_model([self.x_inter, self.y_inter], training=True)
            u_loss_inside = self.loss_fn(self.u_int, u_i)
            v_loss_inside = self.loss_fn(self.v_int, v_i)
            std_loss_inside = (u_loss_inside + v_loss_inside)

            # Total loss
            loss = (100)*std_loss + (10)*residual_loss + (10)*std_loss_inside

        grads = tape.gradient(loss, self.nn_model.trainable_weights)
  
        self.optimizer.apply_gradients(zip(grads, self.nn_model.trainable_weights))


        self.std_loss_tracker.update_state(std_loss)
        self.residual_loss_tracker.update_state(residual_loss)
        self.std_loss_inside_tracker.update_state(std_loss_inside)
    
        return {"std_loss": self.std_loss_tracker.result(), "residual_loss": self.residual_loss_tracker.result(), "std_loss_inside":self.std_loss_inside_tracker.result()}#, res_grads, bound_grads, geo_grads

    def reset_metrics(self):
        self.std_loss_tracker.reset_state()
        self.residual_loss_tracker.reset_state()
        self.std_loss_inside_tracker.reset_state()
        
    def run(self, epochs, proj_name, verbose_freq=1, plot_freq=1):

        log_dir = 'output/' + proj_name
        self.reset_metrics()
        history = {"std_loss": [], "residual_loss": [], "std_loss_inside": []}
 
        name = f"sign_Re : {self.Re},Grid_size_x : {self.g_s_x},Grid_size_y : {self.g_s_y}"
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
            if (epoch + 1) % plot_freq == 0:
                self.get_plots(epoch + 1, proj_name)

        odata = pd.DataFrame(history)

        return history


    def predictions(self, inputs):
        u_pred, v_pred, p_pred = self.nn_model.predict(inputs, batch_size=32)

        return u_pred, v_pred, p_pred
   
  

    def get_plots(self, step, name):

        param_data = loadmat('param.mat')
        true_data = loadmat('300.mat')

        x_grid, y_grid = param_data['XP'].T.shape
        xmesh = np.linspace(start=0., stop=param_data['app']['a'].item()[0, 0], num=x_grid)
        ymesh = np.linspace(start=0., stop=param_data['app']['b'].item()[0, 0], num=y_grid)
        X, Y = np.meshgrid(xmesh, ymesh)

        x_center = param_data['xcdisk']
        y_center = param_data['ycdisk']
        radius = param_data['rdisk']

        lx = X < (x_center - radius)
        ux = X > (x_center + radius)
        ly = Y < (y_center - radius)
        uy = Y > (y_center + radius)

        ind_sq = (lx + ux + ly + uy).astype('uint8')

        xc = X[1:-1, 1:-1][np.nonzero(ind_sq[1:-1, 1:-1])][:, None]
        yc = Y[1:-1, 1:-1][np.nonzero(ind_sq[1:-1, 1:-1])][:, None]

        u_interpolated_data = griddata(
            np.hstack((param_data['XU'].T[param_data['Nodeu'].T[1:-1, 1:-1] > 5].reshape((-1, 1)),
                       param_data['YU'].T[param_data['Nodeu'].T[1:-1, 1:-1] > 5].reshape((-1, 1)))),
            true_data['u'][5:],
            np.hstack((xc, yc)))
        v_interpolated_data = griddata(
            np.hstack((param_data['XV'].T[param_data['Nodev'].T[1:-1, 1:-1] > 5].reshape((-1, 1)),
                       param_data['YV'].T[param_data['Nodev'].T[1:-1, 1:-1] > 5].reshape((-1, 1)))),
            true_data['v'][5:],
            np.hstack((xc, yc)))

        test_data = [X.reshape((-1, 1)), Y.reshape((-1, 1))]

        u_test, v_test, p_test = self.predictions(test_data)
        u_true = np.zeros_like(X)
        v_true = np.zeros_like(X)

        u_true[np.nonzero(ind_sq - 1)] = np.nan
        v_true[np.nonzero(ind_sq - 1)] = np.nan
        u_true[1:-1, 1:-1][np.nonzero(ind_sq[1:-1, 1:-1])] = u_interpolated_data.reshape((-1))
        v_true[1:-1, 1:-1][np.nonzero(ind_sq[1:-1, 1:-1])] = v_interpolated_data.reshape((-1))
        u_true[1:-1, 0] = self.outputs['ub_dir'].max()
        v_true[1:-1, 0] = 0

        u_test = u_test.reshape(X.shape)
 
        u_test[np.nonzero(ind_sq-1)] = np.nan
        v_test = v_test.reshape(X.shape)
       
        v_test[np.nonzero(ind_sq-1)] = np.nan

        u_er = u_true - u_test
        v_er = v_true - v_test
       
        true_mag = (u_true ** 2 + v_true ** 2) ** 0.5
        pred_mag = (u_test ** 2 + v_test ** 2) ** 0.5
        err_mag = (true_mag - pred_mag)#/(true_mag)

        fig, ax = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(20, 3),
                               gridspec_kw={'wspace': 0.2, 'width_ratios': [1, 1, 1]})
        # fig.tight_layout()
        level = np.linspace(start=true_mag[1:-1, :-1][np.nonzero(ind_sq[1:-1, :-1])].min(),
                stop=true_mag[1:-1, :-1][np.nonzero(ind_sq[1:-1, :-1])].max(), num=7)

        pres = ax[0].streamplot(X, Y, u_test, v_test,
                                color='k', linewidth=0.5)
        pre = ax[0].contourf(X, Y, pred_mag, level,
                             cmap=plt.cm.cool, extend='both')
        fig.colorbar(pre, ax=ax[0])
        pre.cmap.set_under('yellow')
        pre.cmap.set_over('red')
        refs = ax[1].streamplot(X[1:-1, :-1], Y[1:-1, :-1], u_true[1:-1, :-1], v_true[1:-1, :-1],
                                color='k', linewidth=0.5)
        ref = ax[1].contourf(X[1:-1, :-1], Y[1:-1, :-1], true_mag[1:-1, :-1], level,
                             cmap=plt.cm.cool, extend='both')
        fig.colorbar(ref, ax=ax[1])
        ers = ax[2].streamplot(X[1:-1, :-1], Y[1:-1, :-1], u_er[1:-1, :-1], v_er[1:-1, :-1],
                               color='k', linewidth=0.5)
        er = ax[2].contourf(X[1:-1, :-1], Y[1:-1, :-1], err_mag[1:-1, :-1], cmap=plt.cm.cool)
        fig.colorbar(er, ax=ax[2])
        ax[0].title.set_text("Pred")
        ax[0].set_ylabel("Y")
        ax[1].title.set_text("True")
        ax[2].title.set_text("Error")
        ax[0].set_xlabel("X")
        ax[1].set_xlabel("X")
        ax[2].set_xlabel("X")
        plt.savefig('output/' + 'at_' + str(step) + '.png', dpi=300)
        plt.close()
     