import numpy as np


def get_ibc_and_inner_data(start, stop, grid_size, nue):
    # Provides the boundary coordinates and boundary conditions

    # Boundary Points
    xdisc = np.linspace(start=start[0], stop=stop[0], num=grid_size)
    ydisc = np.linspace(start=stop[1], stop=start[1], num=grid_size)

    X, Y = np.meshgrid(xdisc, ydisc)
    grid_loc = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    # boundary conditions
    x_top = np.hstack((X[0, :][:, None], Y[0, :][:, None]))
    x_bottom = np.hstack((X[0, 1:-1][:, None], Y[-1, 1:-1][:, None]))
    x_left = np.hstack((X[1:, 0][:, None], Y[1:, 0][:, None]))
    x_right = np.hstack((X[1:, -1][:, None], Y[1:, 0][:, None]))

    u_top, v_top, p_top = get_fvalues(x_top[:, 0:1], x_top[:, 1:], nue=nue)
    u_bottom, v_bottom, p_bottom = get_fvalues(x_bottom[:, 0:1], x_bottom[:, 1:], nue=nue)
    u_left, v_left, p_left = get_fvalues(x_left[:, 0:1], x_left[:, 1:], nue=nue)
    u_right, v_right, p_right = get_fvalues(x_right[:, 0:1], x_right[:, 1:], nue=nue)

    # idx = np.random.choice(grid_size - 2, nu, replace=False)
    xb = np.vstack((x_top[:, 0:1], x_bottom[:, 0:1], x_left[:, 0:1], x_right[:, 0:1]))
    yb = np.vstack((x_top[:, 1:2], x_bottom[:, 1:2], x_left[:, 1:2], x_right[:, 1:2]))
    u_ob = np.vstack((u_top, u_bottom, u_left, u_right))
    v_ob = np.vstack((v_top, v_bottom, v_left, v_right))
    p_ob = np.vstack((p_top, p_bottom, p_left, p_right))

    xd = grid_loc[:, 0:1]
    yd = grid_loc[:, 1:2]

    return xb, yb, xd, yd, u_ob, v_ob, p_ob


def get_fvalues(x, y, nue):
    zeta = (0.5 / nue) - np.sqrt((1 / (4 * nue ** 2)) + 4 * np.pi ** 2)

    u_val = 1 - np.exp(zeta * x) * np.cos(2 * np.pi * y)
    v_val = (zeta / (2 * np.pi)) * np.exp(zeta * x) * np.sin(2 * np.pi * y)
    p_val = 0.5 * (1 - np.exp(2 * zeta * x))

    return u_val, v_val, p_val


def bound_param_data(num_samples, boundary_points, xlim, ylim, nue):

    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    # left boundary
    xl = np.ones((num_samples, boundary_points)) * xlim[0]
    yl = y_range * np.random.rand(num_samples, boundary_points) + ylim[0]

    ul, vl,  pl = get_fvalues(xl, yl, nue=nue)

    # right boundary
    xr = np.ones((num_samples, boundary_points)) * xlim[1]
    yr = y_range * np.random.rand(num_samples, boundary_points) + ylim[0]

    ur, vr, pr = get_fvalues(xr, yr, nue=nue)

    # Top boundary
    yt = np.ones((num_samples, boundary_points)) * ylim[1]
    xt = x_range * np.random.rand(num_samples, boundary_points) + xlim[0]

    ut, vt, pt = get_fvalues(xt, yt, nue=nue)

    # Bottom boundary
    yb = np.ones((num_samples, boundary_points)) * ylim[0]
    xb = x_range * np.random.rand(num_samples, boundary_points) + xlim[0]

    ub, vb, pb = get_fvalues(xb, yb, nue=nue)

    xbc = np.hstack((xl, xr, xt, xb))
    ybc = np.hstack((yl, yr, yt, yb))
    ubc = np.hstack((ul, ur, ut, ub))
    vbc = np.hstack((vl, vr, vt, vb))
    pbc = np.hstack((pl, pr, pt, pb))

    return xbc, ybc, ubc, vbc, pbc
