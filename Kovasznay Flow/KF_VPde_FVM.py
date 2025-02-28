import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time



from tensorflow.python.ops.numpy_ops import np_config
from utils import get_fvalues

np_config.enable_numpy_behavior()

tf.random.set_seed(1234)
tf.keras.backend.set_floatx("float64")


class PdeModel:
    def __init__(self, inputs, outputs, get_models, loss_fn, optimizer, metrics, parameters, batches=1):
        self.inputs = inputs
        self.outputs = outputs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batches = batches

        self.nue = parameters['nue']

        # Create efficient data pipelines
        self.inner_data = self.create_data_pipeline(inputs['xin'], inputs['yin'], batch=batches)
        self.boundary_data = self.create_data_pipeline(inputs['xb'], inputs['yb'], outputs['ub'],
                                                       outputs['vb'], outputs['pb'],
                                                       batch=batches)
        self.del_x = 1.5/64.0
        self.del_y = 2.0/64.0
        self.nn_model = get_models['nn_model']

        self.loss_tracker = metrics['loss']
        self.u_loss_tracker = metrics['u_loss']
        self.v_loss_tracker = metrics['v_loss']
        self.p_loss_tracker = metrics['p_loss']
        self.bound_loss_tracker = metrics['boundary_loss']
        self.residual_loss_tracker = metrics['residual_loss']

    @staticmethod
    def create_data_pipeline(*args, batch):
        dataset = tf.data.Dataset.from_tensor_slices(args)
        #dataset = dataset.shuffle(buffer_size=len(args[0]))
        dataset = dataset.batch(np.ceil(len(args[0])/batch))
        return dataset

    @tf.function
    def Pde_residual(self, x, y ,training=True):

        u, v, P = self.nn_model([x, y], training=True)
        u = tf.reshape(u,(64, 64))
        v = tf.reshape(v,(64, 64))
        P = tf.reshape(P,(64, 64))
        u = tf.cast(u,dtype='float64')
        v = tf.cast(v,dtype='float64')
        P = tf.cast(P,dtype='float64')
       
        fe   =  (0.5*((u[1:64-1,1:64-1]+u[1:64-1,2:64])/self.del_x))
        fw   =  (0.5*((u[1:64-1,1:64-1]+u[1:64-1,0:64-2])/self.del_x)) 
        fn   =  (0.5*((v[1:64-1,1:64-1]+v[0:64-2,1:64-1])/self.del_y)) 
        fs   =  (0.5*((v[1:64-1,1:64-1]+v[2:64,1:64-1])/self.del_y))    

        # X-Momentum Equation 

        fx_x =  ((tf.multiply(u[1:64-1,1:64-1],tf.nn.relu(fe))-tf.multiply(u[1:64-1,2:64],tf.nn.relu(-fe)))
                   -(tf.multiply(u[1:64-1,0:64-2],tf.nn.relu(fw))-tf.multiply(u[1:64-1,1:64-1],tf.nn.relu(-fw)))
                   +(tf.multiply(u[1:64-1,1:64-1],tf.nn.relu(fn))-tf.multiply(u[0:64-2,1:64-1],tf.nn.relu(-fn)))
                   -(tf.multiply(u[2:64,1:64-1],tf.nn.relu(fs))-tf.multiply(u[1:64-1,1:64-1],tf.nn.relu(-fs)))
                   +(1/(2*self.del_x))*(P[1:64-1,2:64]-P[1:64-1,0:64-2]))
        
        fx_y = (self.nue)*((1/(self.del_x*self.del_x))*(u[1:64-1,2:64]-2*u[1:64-1,1:64-1]+u[1:64-1,0:64-2])
                                  +(1/(self.del_y*self.del_y))*(u[2:64,1:64-1]-2*u[1:64-1,1:64-1]+u[0:64-2,1:64-1]))
        
        fx = fx_x - (1.0)*fx_y
        
        # Y-Momentum Equation

        fy_x = ((tf.multiply(v[1:64-1,1:64-1],tf.nn.relu(fe))-tf.multiply(v[1:64-1,2:64],tf.nn.relu(-fe)))
                   -(tf.multiply(v[1:64-1,0:64-2],tf.nn.relu(fw))-tf.multiply(v[1:64-1,1:64-1],tf.nn.relu(-fw)))
                   +(tf.multiply(v[1:64-1,1:64-1],tf.nn.relu(fn))-tf.multiply(v[0:64-2,1:64-1],tf.nn.relu(-fn)))
                   -(tf.multiply(v[2:64,1:64-1],tf.nn.relu(fs))-tf.multiply(v[1:64-1,1:64-1],tf.nn.relu(-fs)))
                   +(1/(2*self.del_y))*(P[0:64-2,1:64-1]-P[2:64,1:64-1]))
        
        fy_y = (self.nue)*((1/(self.del_x*self.del_x))*(v[1:64-1,2:64]-2*v[1:64-1,1:64-1]+v[1:64-1,0:64-2])
                                 +(1/(self.del_y*self.del_y))*(v[2:64,1:64-1]-2*v[1:64-1,1:64-1]+v[0:64-2,1:64-1]))
        
        fy = fy_x - (1.0)*fy_y

        # Continuity

        div_u = ((1/(2*self.del_x))*(u[1:64-1,2:64]-u[1:64-1,0:64-2]) + (1/(2*self.del_y))*(v[0:64-2,1:64-1]-v[2:64,1:64-1]))

        residual_loss = tf.reduce_mean(tf.square(fx)) + tf.reduce_mean(tf.square(fy)) + tf.reduce_mean(tf.square(div_u))

        return residual_loss

    @tf.function
    def train_step(self, xb, yb, ub, vb, pb, xin, yin):
        with tf.GradientTape(persistent=True) as tape:
            u_pred, v_pred, p_pred = self.nn_model([xb, yb], training=True)
            residual_loss = self.Pde_residual(xin, yin, training=True)

            u_loss = self.loss_fn(ub, u_pred)
            v_loss = self.loss_fn(vb, v_pred)
            p_loss = self.loss_fn(pb, p_pred)
            bound_loss = u_loss + v_loss + p_loss
            loss = bound_loss + residual_loss

        grads = tape.gradient(loss, self.nn_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn_model.trainable_weights))

        self.loss_tracker.update_state(loss)
        self.u_loss_tracker.update_state(u_loss)
        self.v_loss_tracker.update_state(v_loss)
        self.p_loss_tracker.update_state(p_loss)
        self.bound_loss_tracker.update_state(bound_loss)
        self.residual_loss_tracker.update_state(residual_loss)

        return {"loss": self.loss_tracker.result(), "bound_loss": self.bound_loss_tracker.result(),
                "residual_loss": self.residual_loss_tracker.result(),
                "u_loss": self.u_loss_tracker.result(), "v_loss": self.v_loss_tracker.result(),
                "p_loss": self.p_loss_tracker.result()}

    def reset_metrics(self):
        self.loss_tracker.reset_state()
        self.bound_loss_tracker.reset_state()
        self.residual_loss_tracker.reset_state()
        self.u_loss_tracker.reset_state()
        self.v_loss_tracker.reset_state()
        self.p_loss_tracker.reset_state()


    def run(self, epochs, log_dir, wb=False, verbose_freq=1000, plot_freq=10000):

        history = {"loss": [], "bound_loss": [], "residual_loss": [],
                   "u_loss": [], "v_loss": [], 'p_loss': []}
        start_time = time.time()

        #self.get_model_graph(log_dir=log_dir, wb=wb)

        for epoch in range(epochs):
            self.reset_metrics()

            for j, ((xb, yb, ub, vb, pb),
                    (xin, yin)) in enumerate(zip(self.boundary_data, self.inner_data)):
                logs = self.train_step(xb, yb, ub, vb, pb, xin, yin)


            tae = time.time() - start_time
            for key, value in logs.items():
                history[key].append(value.numpy())
            if (epoch + 1) % verbose_freq == 0:
                print(f'''Epoch:{epoch + 1}/{epochs}''')
                for key, value in logs.items():
                    # history[key].append(value.numpy())
                    print(f"{key}: {value:.4f} ", end="")
                print(f"Time: {tae / 60:.4f}min")
            if (epoch + 1) % plot_freq == 0:
                self.get_plots(epoch + 1, log_dir=log_dir, wb=wb)
               

        odata = pd.DataFrame(history)
        odata.to_csv(path_or_buf=log_dir + 'history.csv')

        plt.figure()
        plt.plot(range(1, len(odata) + 1), np.log(odata['loss']))
        plt.xlabel('Epochs')
        plt.ylabel('Log_Loss')
        plt.title('log loss plot')
        plt.savefig(log_dir + '_log_loss_plt.png', dpi=300)
        
        return history

    def predictions(self, inputs):
        u_pred, v_pred, p_pred = self.nn_model.predict(inputs, batch_size=32)

        return u_pred, v_pred, p_pred

    def get_plots(self, step, log_dir, wb=False):

        xdisc = np.linspace(start=-0.5, stop=1., num=64)
        ydisc = np.linspace(start=-0.5, stop=1.5, num=64)

        X, Y = np.meshgrid(xdisc, ydisc)
        grid_loc = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

        test_data = [grid_loc[:, 0:1], grid_loc[:, 1:]]

        u_test, v_test, p_test = self.predictions(test_data)
        u_test = u_test.reshape(X.shape)
        v_test = v_test.reshape(X.shape)
        p_test = p_test.reshape(X.shape)

        u_true, v_true, p_true = get_fvalues(x=grid_loc[:, 0:1], y=grid_loc[:, 1:], nue=self.nue)
        u_true = u_true.reshape(X.shape)
        v_true = v_true.reshape(X.shape)
        p_true = p_true.reshape(X.shape)

        true_mag = (u_true ** 2 + v_true ** 2) ** 0.5
        pred_mag = (u_test ** 2 + v_test ** 2) ** 0.5
        er_mag = true_mag - pred_mag
        u_er = u_true - u_test
        v_er = v_true - v_test
        p_er = p_true - p_test

        level = np.linspace(true_mag.min(), true_mag.max(), num=7)

        fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharex='col', sharey='row')
        fig.tight_layout()

        pres = ax[0][0].streamplot(X, Y, u_test, v_test, color='k',
                                   linewidth=0.5)
        pre = ax[0][0].contourf(X, Y, pred_mag, level, cmap='cool', extend='both')
        fig.colorbar(pre, ax=ax[0, 0])
        pre.cmap.set_under('yellow')
        pre.cmap.set_over('green')

        refs = ax[0][1].streamplot(X, Y, u_true, v_true, color='k',
                                   linewidth=0.5)
        ref = ax[0][1].contourf(X, Y, true_mag, level, cmap='cool', extend='both')
        fig.colorbar(ref, ax=ax[0, 1])
        ers = ax[0, 2].streamplot(X, Y, u_er, v_er,
                                  color='k', linewidth=0.5)
        er = ax[0, 2].contourf(X, Y, er_mag, cmap=plt.cm.cool)
        fig.colorbar(er, ax=ax[0, 2])

        prep = ax[1][0].contourf(X, Y, p_test, cmap='cool')
        fig.colorbar(prep, ax=ax[1][0])
        refp = ax[1][1].contourf(X, Y, p_true, cmap='cool')
        fig.colorbar(refp, ax=ax[1, 1])
        erp = ax[1][2].contourf(X, Y, p_er, cmap='cool')

        ax[0][0].title.set_text("Pred")
        ax[0][0].set_ylabel("Y")
        ax[1][0].set_xlabel("X")
        ax[0][1].title.set_text("True")
        ax[1][1].title.set_text('Error')
        ax[1][1].set_xlabel("X")
        ax[1][2].set_xlabel("X")
        plt.savefig(log_dir + 'at_' + str(step) + '_' + '.png', dpi=300)
        plt.close()
        
