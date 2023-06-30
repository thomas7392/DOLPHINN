# Thomas Goldman 2023
# DOLPHINN

import deepxde as dde
from deepxde.backend import tf

from .function import Function

class TwoBodyProblemNoneDimensional(Function):
    '''
    Equations of motion for the two body problem with cartesian coordinates,
    in non-dimensional units. Time has been scaled with time_scale and position
    with length_scale. Function is used by DeepXDE to construct the Dynamics
    loss term of a PINN

    State entries: [x, y, v_x, v_y]

    Args:
        x (tf.tensor): Network input, time
        y (tf.tensor): Network prediction, position, velocity

    Returns:
        loss (list): Residial of the individual equations of motion
    '''
    control = False
    coordinates = 'NDcartesian'
    entries = 4
    control_entries = 0
    loss_entries = 4
    loss_labels = ["x", "y", "v$_x$", "v$_y$"]
    entry_labels = ["x", "y", "v$_x$", "v$_y$"]

    def __init__(self, data):
        super().__init__(data)

    def call(self, x, y):

        # Unpack tensors
        x_tens  = y[:, 0:1]
        y_tens  = y[:, 1:2]
        vx_tens = y[:, 2:3]
        vy_tens = y[:, 3:4]
        r = tf.reshape(tf.norm(y[:, 0:2], axis = 1), (-1, 1))

        # Automatic differentation
        dx_dt = dde.grad.jacobian(y, x, i=0)
        dy_dt = dde.grad.jacobian(y, x, i=1)
        dvx_dt = dde.grad.jacobian(y, x, i=2)
        dvy_dt = dde.grad.jacobian(y, x, i=3)

        RHS_x  = vx_tens
        RHS_y  = vy_tens
        RHS_vx  = - (self.mu * self.time_scale**2 / self.length_scale**3) * x_tens * r**(-3)
        RHS_vy  = - (self.mu * self.time_scale**2 / self.length_scale**3) * y_tens * r**(-3)

        return [
            dx_dt - RHS_x,
            dy_dt - RHS_y,
            dvx_dt - RHS_vx,
            dvy_dt - RHS_vy
            ]



class TwoBodyProblemNonDimensionalControl(Function):
    '''
    Equations of motion for the two body problem in non-dimensional units,
    including a control term. Time has been scaled with time_scale and position
    with length_scale. Function is used by DeepXDE to construct the Dynamics
    loss term of a PINN

    State entries: [x, y, v_x, v_y, u_x, u_y]

    Args:
        x (tf.tensor): Network input, time
        y (tf.tensor): Network prediction, position, velocity

    Returns:
        loss (list): Residial of the individual equations of motion
    '''
    control = False
    entries = 6
    control_entries = 2
    loss_entries = 4
    coordinates = 'NDcartesian'
    loss_labels = ["x", "y", "v$_x$", "v$_y$"]
    entry_labels = ["x", "y", "v$_x$", "v$_y$", "u$_x$", "u$_y$" ]

    def __init__(self, data):
        super().__init__(data)

    def call(self, x, y):

        # Unpack tensors
        x_tens  = y[:, 0:1]
        y_tens  = y[:, 1:2]
        vx_tens = y[:, 2:3]
        vy_tens = y[:, 3:4]

        ux_tens = y[:, 4:5]
        uy_tens = y[:, 5:6]

        r = tf.reshape(tf.norm(y[:, 0:2], axis = 1), (-1, 1))

        # Automatic differentation
        dx_dt = dde.grad.jacobian(y, x, i=0)
        dy_dt = dde.grad.jacobian(y, x, i=1)
        dvx_dt = dde.grad.jacobian(y, x, i=2)
        dvy_dt = dde.grad.jacobian(y, x, i=3)

        RHS_x = vx_tens
        RHS_y  = vy_tens
        RHS_vx  = - (self.mu * self.time_scale**2 / self.length_scale**3) * x_tens * r**(-3) +\
              self.time_scale**2 / (self.length_scale*self.m) * ux_tens
        RHS_vy  = - (self.mu * self.time_scale**2 / self.length_scale**3) * y_tens * r**(-3) +\
              self.time_scale**2 / (self.length_scale*self.m) * uy_tens

        return [
            dx_dt - RHS_x,
            dy_dt - RHS_y,
            dvx_dt - RHS_vx,
            dvy_dt - RHS_vy,
            ]



class TwoBodyProblemRadialNonDimensional(Function):
    '''
    Equations of motion for the two body problem with radial coordinates,
    in non-dimensional units. Time has been scaled with time_scale and position
    with length_scale. Function is used by DeepXDE to construct the Dynamics
    loss term of a PINN.

    State entries: [r, v_r, v_{\theta}]

    Args:
        x (tf.tensor): Network input, time
        y (tf.tensor): Network prediction, position, velocity

    Returns:
        loss (list): Residial of the individual equations of motion
    '''
    control = False
    entries = 3
    control_entries = 0
    loss_entries = 3
    coordinates = 'radial'
    loss_labels = ["r", "v$_r$", r"v$_{\theta}$"]
    entry_labels = ["r", "v$_r$", r"v$_{\theta}$"]

    def __init__(self, data):
        super().__init__(data)

    def call(self, time, y):

        # Unpack tensors
        x1  = y[:, 0:1]
        x2  = y[:, 1:2]
        x3 =  y[:, 2:3]

        # Automatic differentation
        dx1_dt = dde.grad.jacobian(y, time, i=0)
        dx2_dt = dde.grad.jacobian(y, time, i=1)
        dx3_dt = dde.grad.jacobian(y, time, i=2)

        RHS_x1  = x2
        RHS_x2  = x3**2/x1 - (self.mu * self.time_scale**2 / self.length_scale**3) * x1**(-2)
        RHS_x3  = - (x2*x3)/x1

        return [
            dx1_dt - RHS_x1,
            dx2_dt - RHS_x2,
            dx3_dt - RHS_x3,
            ]



class TwoBodyProblemRadialNonDimensionalControl(Function):
    '''
    Equations of motion for the two body problem in non-dimensional units,
    including a control term. Time has been scaled with time_scale and position
    with length_scale. Function is used by DeepXDE to construct the Dynamics
    loss term of a PINN

    State entries: [r, v_r, v_{\theta}, u_r, u_{\theta}]

    Args:
        x (tf.tensor): Network input, time
        y (tf.tensor): Network prediction, position, velocity

    Returns:
        loss (list): Residual of the individual equations of motion
    '''

    control = True
    entries = 5
    control_entries = 2
    loss_entries = 3
    coordinates = 'radial'
    loss_labels = ["r", "v$_r$", r"v$_{\theta}$"]
    entry_labels = ["r", "v$_r$", r"v$_{\theta}$", "u$_r$", r"u$_{\theta}$"]

    def __init__(self, data):
        super().__init__(data)

    def call(self, time, y):

        # Unpack tensors
        x1 = y[:, 0:1]
        x2 = y[:, 1:2]
        x3 = y[:, 2:3]

        ur = y[:, 3:4]
        ut = y[:, 4:5]

        # Automatic differentation
        dx1_dt = dde.grad.jacobian(y, time, i=0)
        dx2_dt = dde.grad.jacobian(y, time, i=1)
        dx3_dt = dde.grad.jacobian(y, time, i=2)

        RHS_x1  = x2
        RHS_x2  = x3**2/x1 - (self.mu * self.time_scale**2 / self.length_scale**3) * x1**(-2) +\
              (self.time_scale**2/self.length_scale) * ur / self.m
        RHS_x3  = - (x2*x3)/x1 + (self.time_scale**2/self.length_scale) * ut / self.m

        return [
            dx1_dt - RHS_x1,
            dx2_dt - RHS_x2,
            dx3_dt - RHS_x3,
            ]
