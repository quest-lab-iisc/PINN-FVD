import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib as mpl
from matplotlib import lines

from scipy.io import loadmat
from scipy.interpolate import griddata


from matplotlib.ticker import EngFormatter, ScalarFormatter, FormatStrFormatter
np.random.seed(1234)
tf.random.set_seed(1234)

x_lim = 20
y_lim = 3
grid_size_x =256
grid_size_y = 256

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
theta    = np.linspace(start=0., stop=2*np.pi, num=100)
# Left
#theta = np.linspace(start=(0.5)*np.pi, stop=(1.5)*np.pi, num=100)
# Right
# theta = np.linspace(start=-(0.5)*np.pi, stop=(0.5)*np.pi, num=25)
xp       = (x_center + radius * np.cos(theta))
yp       = (y_center + radius * np.sin(theta))
xp_1     = xp[:, None]

up       = np.zeros_like(xp)
vp       = np.zeros_like(xp)

# Interior points
#radius = 0.5 + np.sqrt((x_lim/(float(grid_size_x)))*(x_lim/(float(grid_size_x))) + (y_lim/(float(grid_size_y)))*(y_lim/(float(grid_size_y))))
radius = 0.5 #- min(x_lim/(float(grid_size_x)),y_lim/(float(grid_size_y)))
radius_1 = 0.5 + min(x_lim/(float(grid_size_x)),y_lim/(float(grid_size_y)))

R        = np.linalg.norm((X-x_center,Y-y_center),axis=0)
#R_1      = np.linalg.norm((X-x_center,Y-y_center),axis=0)
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
#Yout[R<=radius] = 0.0
condition = (radius_1>=R)&(R>radius)
Xout[condition] = 10
Xout[R>radius_1] = 1.0
#Xout[R>radius_1] = 1.0
#Xout      = Xout[np.logical_not(np.isnan(Xout))]
#Yout      = Yout[np.logical_not(np.isnan(Yout))]

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
print(u_bound_dir.max())
Xout = Xout[:254,:254]


nn_model = tf.keras.models.load_model('saved_model_ICCS/model_ICCS_Final.keras')
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

u_test, v_test, p_test = nn_model.predict(test_data, batch_size=32)
u_true = np.zeros_like(X)
v_true = np.zeros_like(X)

u_true[np.nonzero(ind_sq - 1)] = np.nan
v_true[np.nonzero(ind_sq - 1)] = np.nan
u_true[1:-1, 1:-1][np.nonzero(ind_sq[1:-1, 1:-1])] = u_interpolated_data.reshape((-1))
v_true[1:-1, 1:-1][np.nonzero(ind_sq[1:-1, 1:-1])] = v_interpolated_data.reshape((-1))
u_true[1:-1, 0] = 1.0
v_true[1:-1, 0] = 0

u_test = u_test.reshape(X.shape)
u_test[np.nonzero(ind_sq-1)] = np.nan
v_test = v_test.reshape(X.shape)
v_test[np.nonzero(ind_sq-1)] = np.nan

u_er = u_true - u_test
v_er = v_true - v_test

# u_er = (u_true - u_test)/(u_true)
# v_er = (v_true - v_test)/(v_true)
true_mag = (u_true ** 2 + v_true ** 2) ** 0.5
pred_mag = (u_test ** 2 + v_test ** 2) ** 0.5
err_mag = (true_mag - pred_mag)#/(true_mag)

fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(9, 3),
                        gridspec_kw={'wspace': 0.1, 'width_ratios': [1, 1]})
# fig.tight_layout()
level = np.linspace(start=true_mag[1:-1, :-1][np.nonzero(ind_sq[1:-1, :-1])].min(),
        stop=true_mag[1:-1, :-1][np.nonzero(ind_sq[1:-1, :-1])].max(), num=7)

pres = ax[0].streamplot(X, Y, u_test, v_test,
                        color='k', linewidth=0.5)
pre = ax[0].contourf(X, Y, pred_mag, level,
                        cmap='Blues', extend='both')
fig.colorbar(pre, ax=ax[0])
pre.cmap.set_under('yellow')
pre.cmap.set_over('red')
refs = ax[1].streamplot(X[1:-1, :-1], Y[1:-1, :-1], u_true[1:-1, :-1], v_true[1:-1, :-1],
                        color='k', linewidth=0.5)
ref = ax[1].contourf(X[1:-1, :-1], Y[1:-1, :-1], true_mag[1:-1, :-1], level,
                        cmap='Blues', extend='both')
fig.colorbar(ref, ax=ax[1])
# ers = ax[2].streamplot(X[1:-1, :-1], Y[1:-1, :-1], u_er[1:-1, :-1], v_er[1:-1, :-1],
#                         color='k', linewidth=0.5)
# er = ax[2].contourf(X[1:-1, :-1], Y[1:-1, :-1], err_mag[1:-1, :-1], cmap=plt.cm.cool)
# fig.colorbar(er, ax=ax[2])
ax[0].title.set_text("PINNs solution")
ax[0].set_ylabel("Y")
ax[1].title.set_text("Numerical solution")
# ax[2].title.set_text("Error")
ax[0].set_xlabel("X")
ax[1].set_xlabel("X")
# ax[2].set_xlabel("X")
plt.savefig('output/' + 'FPC' + '.png', dpi=300)
plt.close()


u_er_act = u_er[np.nonzero(ind_sq)]
v_er_act = v_er[np.nonzero(ind_sq)]
mean_u = np.mean(u_er_act)
var_u  = np.var(u_er_act)
mean_v = np.mean(v_er_act)
var_v  = np.var(v_er_act)

print(f"Mean_u:{mean_u},Var_u:{var_u}")
print(f"Mean_v:{mean_v},Var_v{var_v}")
