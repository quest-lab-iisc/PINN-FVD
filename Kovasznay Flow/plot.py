import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import tensorflow as tf

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from utils import get_fvalues

xdisc = np.linspace(start=-0.5, stop=1., num=128)
ydisc = np.linspace(start=-0.5, stop=1.5, num=128)

X, Y = np.meshgrid(xdisc, ydisc)
grid_loc = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

test_data = [grid_loc[:, 0:1], grid_loc[:, 1:]]

nn_model_AD = tf.keras.models.load_model('output_model/AD_new.keras')
nn_model_FVD = tf.keras.models.load_model('output_model/FVD_new.keras')

u_test_AD, v_test_AD, p_test_AD = nn_model_AD(test_data)
u_test_FVD, v_test_FVD, p_test_FVD = nn_model_FVD(test_data)

u_test_AD = u_test_AD.reshape(X.shape)
v_test_AD = v_test_AD.reshape(X.shape)
p_test_AD = p_test_AD.reshape(X.shape)

u_test_FVD = u_test_FVD.reshape(X.shape)
v_test_FVD = v_test_FVD.reshape(X.shape)
p_test_FVD = p_test_FVD.reshape(X.shape)

u_true, v_true, p_true = get_fvalues(x=grid_loc[:, 0:1], y=grid_loc[:, 1:], nue=0.025)
u_true = u_true.reshape(X.shape)
v_true = v_true.reshape(X.shape)
p_true = p_true.reshape(X.shape)

true_mag = (u_true ** 2 + v_true ** 2) ** 0.5
pred_mag_AD = (u_test_AD ** 2 + v_test_AD ** 2) ** 0.5
pred_mag_FVD = (u_test_FVD ** 2 + v_test_FVD ** 2) ** 0.5


level = np.linspace(true_mag.min(), true_mag.max(), num=7)

fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharex='col', sharey='row')
fig.tight_layout()

pres = ax[1].streamplot(X, Y, u_test_AD, v_test_AD, color='k',
                            linewidth=0.5)
pre = ax[1].contourf(X, Y, pred_mag_AD, level, cmap='Blues', extend='both')
cbar = fig.colorbar(pre, ax=ax[1])
cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
pre.cmap.set_under('yellow')
pre.cmap.set_over('green')

pres_FVD = ax[0].streamplot(X, Y, u_test_FVD, v_test_FVD, color='k',
                            linewidth=0.5)
pre_FVD = ax[0].contourf(X, Y, pred_mag_FVD, level, cmap='Blues', extend='both')
cbar = fig.colorbar(pre_FVD, ax=ax[0])
cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

refs = ax[2].streamplot(X, Y, u_true, v_true, color='k',
                            linewidth=0.5)
ref = ax[2].contourf(X, Y, true_mag, level, cmap='Blues', extend='both')
cbar = fig.colorbar(ref, ax=ax[2])
cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))



ax[0].title.set_text("Pred")
ax[0].set_ylabel("Y")
ax[0].set_xlabel("X")
ax[1].set_xlabel("X")
ax[2].set_xlabel("X")
ax[0].title.set_text("PINNs FVD solution")
ax[1].title.set_text("PINNs AD solution")
ax[2].title.set_text("True solution")
plt.savefig('Compare_AD_FVD' + '.png', dpi=300)
plt.close()
      