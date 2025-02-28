import numpy as np
from scipy.io import loadmat
#from pyDOE import lhs


def get_ibc_and_inner_data(start, stop, grid_size, top_velocity=1):

    # Boundary Points
    xdisc = np.linspace(start=start, stop=stop, num=grid_size)
    ydisc = np.linspace(start=stop, stop=start, num=grid_size)

    X, Y = np.meshgrid(xdisc, ydisc)
    grid_loc = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    # boundary conditions
    x_top = np.hstack((X[0, :][:, None], Y[0, :][:, None]))
    x_bottom = np.hstack((X[0, 1:-1][:, None], Y[-1, 1:-1][:, None]))
    x_left = np.hstack((X[1:, 0][:, None], Y[1:, 0][:, None]))
    x_right = np.hstack((X[1:, -1][:, None], Y[1:, 0][:, None]))

    u_top = np.ones(shape=(len(x_top), 1))*top_velocity
    v_top = np.zeros_like(u_top)

    u_bottom = np.zeros(shape=(len(x_bottom), 1))
    v_bottom = np.zeros_like(u_bottom)

    u_left = np.zeros(shape=(len(x_left), 1))
    v_left = np.zeros_like(u_left)

    u_right = np.zeros(shape=(len(x_right), 1))
    v_right = np.zeros_like(u_right)

    # Domain points
    # lb = grid_loc.min(0)
    # ub = grid_loc.max(0)
    #
    # X_domain = lb + (ub - lb) * lhs(2, num_samples)

    # idx = np.random.choice(grid_size - 2, nu, replace=False)
    xb = np.vstack((x_top[:, 0:1], x_bottom[:, 0:1], x_left[:, 0:1], x_right[:, 0:1]))
    yb = np.vstack((x_top[:, 1:2], x_bottom[:, 1:2], x_left[:, 1:2], x_right[:, 1:2]))
    u_ob = np.vstack((u_top, u_bottom, u_left, u_right))
    v_ob = np.vstack((v_top, v_bottom, v_left, v_right))

    X_boundary = np.hstack((xb, yb))

    xd = grid_loc[:, 0:1]
    yd = grid_loc[:, 1:2]

    return xb, yb, xd, yd, u_ob, v_ob


def get_boundary_data(start, stop, grid_size, top_velocity=1):

    # Boundary Points
    xdisc = np.linspace(start=start, stop=stop, num=grid_size)
    ydisc = np.linspace(start=stop, stop=start, num=grid_size)

    # for the domain points
    X, Y = np.meshgrid(xdisc, ydisc)
    grid_loc = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    # boundary conditions
    x_top = np.hstack((X[0, :][:, None], Y[0, :][:, None]))
    x_bottom = np.hstack((X[0, 1:-1][:, None], Y[-1, 1:-1][:, None]))
    x_left = np.hstack((X[1:, 0][:, None], Y[1:, 0][:, None]))
    x_right = np.hstack((X[1:, -1][:, None], Y[1:, 0][:, None]))

    u_top = np.ones(shape=(len(x_top), 1)) * top_velocity
    v_top = np.zeros_like(u_top)

    u_bottom = np.zeros(shape=(len(x_bottom), 1))
    v_bottom = np.zeros_like(u_bottom)

    u_left = np.zeros(shape=(len(x_left), 1))
    v_left = np.zeros_like(u_left)

    u_right = np.zeros(shape=(len(x_right), 1))
    v_right = np.zeros_like(u_right)

    xb = np.vstack((x_top[:, 0:1], x_bottom[:, 0:1], x_left[:, 0:1], x_right[:, 0:1]))
    yb = np.vstack((x_top[:, 1:2], x_bottom[:, 1:2], x_left[:, 1:2], x_right[:, 1:2]))
    u_ob = np.vstack((u_top, u_bottom, u_left, u_right))
    v_ob = np.vstack((v_top, v_bottom, v_left, v_right))

    X_boundary = np.hstack((xb, yb))

    xd = grid_loc[:, 0:1]
    yd = grid_loc[:, 1:2]

    # multiple boundary data
    zeros_tensor = np.zeros(shape=(len(grid_loc), grid_size))
    ones_tensor = np.ones_like(zeros_tensor)
    dfid = np.sort(np.random.rand(len(grid_loc), grid_size))
    dfbd = np.sort(np.random.rand(len(xb), grid_size))
    zc = np.zeros(shape=(len(grid_loc), 1))
    oc = np.ones_like(zc)

    xbc_in = np.concatenate((dfid[:, 1:-1], dfid[:, 1:-1], zeros_tensor, ones_tensor), axis=1)
    ybc_in = np.concatenate((ones_tensor[:, 1:-1], zeros_tensor[:, 1:-1],
                            np.concatenate((zc, dfid[:, 1:-1], oc), axis=1),
                            np.concatenate((zc, dfid[:, 1:-1], oc), axis=1)), axis=1)
    ubc_in = np.concatenate((ones_tensor[:, 1:-1]*top_velocity,
                            zeros_tensor[:, 1:-1], zeros_tensor, zeros_tensor), axis=1)
    vbc_in = np.concatenate((zeros_tensor[:, 1:-1], zeros_tensor[:, 1:-1], zeros_tensor, zeros_tensor), axis=1)

    xbc_b = np.concatenate((dfbd[:, 1:-1], dfbd[:, 1:-1], zeros_tensor[:len(dfbd)], ones_tensor[:len(dfbd)]), axis=1)
    ybc_b = np.concatenate((ones_tensor[:len(dfbd), 1:-1], zeros_tensor[:len(dfbd), 1:-1],
                            np.concatenate((zc[:len(dfbd)], dfbd[:, 1:-1], oc[:len(dfbd)]), axis=1),
                            np.concatenate((zc[:len(dfbd)], dfbd[:, 1:-1], oc[:len(dfbd)]), axis=1)), axis=1)
    ubc_b = np.concatenate((ones_tensor[:len(dfbd), 1:-1]*top_velocity,
                            zeros_tensor[:len(dfbd), 1:-1], zeros_tensor[:len(dfbd), :],
                            zeros_tensor[:len(dfbd), :]), axis=1)
    vbc_b = np.concatenate((zeros_tensor[:len(dfbd), 1:-1], zeros_tensor[:len(dfbd), 1:-1],
                            zeros_tensor[:len(dfbd), :], zeros_tensor[:len(dfbd), :]), axis=1)

    return xb, yb, xd, yd, u_ob, v_ob, xbc_in, ybc_in, ubc_in, vbc_in, xbc_b, ybc_b, ubc_b, vbc_b