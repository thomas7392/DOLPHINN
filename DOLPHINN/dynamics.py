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

    theta = False
    control = False
    mass = False
    coordinates = 'NDcartesian'
    entries = 4
    control_entries = 0
    loss_entries = 4
    loss_labels = ["x", "y", "v$_x$", "v$_y$"]
    entry_labels = ["x", "y", "v$_x$", "v$_y$"]
    on_off = False

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
    theta = False
    control = False
    mass = False
    entries = 6
    control_entries = 2
    loss_entries = 4
    coordinates = 'NDcartesian'
    loss_labels = ["x", "y", "v$_x$", "v$_y$"]
    entry_labels = ["x", "y", "v$_x$", "v$_y$", "u$_x$", "u$_y$" ]
    on_off = False

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
    theta = False
    control = False
    mass = False
    entries = 4
    control_entries = 0
    loss_entries = 3
    coordinates = 'radial'
    loss_labels = ["r", "v$_r$", r"v$_{\theta}$"]
    entry_labels = ["r",  r"$\theta$", "v$_r$", r"v$_{\theta}$"]
    on_off = False

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
    theta = False
    control = True
    mass = False
    entries = 6
    control_entries = 2
    loss_entries = 3
    coordinates = 'radial'
    loss_labels = ["r", "v$_r$", r"v$_{\theta}$"]
    entry_labels = ["r", r"$\theta$", "v$_r$", r"v$_{\theta}$", "u$_r$", r"u$_{\theta}$"]
    on_off = False

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


class TwoBodyProblemRadialNonDimensionalControl_mass(Function):
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

    theta = False
    control = True
    mass = True
    entries = 6
    control_entries = 2
    loss_entries = 4
    coordinates = 'radial'
    loss_labels = ["r", "v$_r$", r"v$_{\theta}$", "m"]
    entry_labels = ["r", r"$\theta$", "v$_r$ ", r"v$_{\theta}$", "u$_r$", r"u$_{\theta}$"]
    on_off = False

    def __init__(self, data):
        super().__init__(data)

    def call(self, time, y):

        # Unpack tensors
        x1 = y[:, 0:1]
        x2 = y[:, 1:2]
        x3 = y[:, 2:3]

        ur = y[:, 3:4]
        ut = y[:, 4:5]
        m  = y[:, 5:6]

        T = tf.reshape(tf.norm(y[:, 3:5], axis = 1), (-1, 1))

        # Automatic differentation
        dx1_dt = dde.grad.jacobian(y, time, i=0)
        dx2_dt = dde.grad.jacobian(y, time, i=1)
        dx3_dt = dde.grad.jacobian(y, time, i=2)
        dm_dt  = dde.grad.jacobian(y, time, i=5)

        RHS_x1  = x2
        RHS_x2  = x3**2/x1 - (self.mu * self.time_scale**2 / self.length_scale**3) * x1**(-2) +\
              (self.time_scale**2/self.length_scale) * ur / m
        RHS_x3  = - (x2*x3)/x1 + (self.time_scale**2/self.length_scale) * ut / m
        RHS_m = -T * self.time_scale / (self.isp * 9.81)

        return [
            dx1_dt - RHS_x1,
            dx2_dt - RHS_x2,
            dx3_dt - RHS_x3,
            dm_dt  - RHS_m,
            ]


class TwoBodyProblemRadialThetaNonDimensionalControl_mass(Function):
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

    theta = True
    control = True
    mass = True
    entries = 6
    control_entries = 2
    loss_entries = 5
    coordinates = 'radial'
    loss_labels = ["r", r"$\theta$", "v$_r$", r"v$_{\theta}$", "m"]
    entry_labels = ["r", r"$\theta$", "v$_r$ ", r"v$_{\theta}$", "u$_r$", r"u$_{\theta}$"]
    on_off = False

    def __init__(self, data):
        super().__init__(data)

    def call(self, time, y):

        # Coordinate entries
        x1    = y[:, 0:1]
        theta = y[:,1:2] # Not used, uncoupled from the others
        x2    = y[:, 2:3]
        x3    = y[:, 3:4]

        # Control entries
        ur    = y[:, 4:5]
        ut    = y[:, 5:6]

        # Mass entry
        m     = y[:, 6:7]

        # Thrust magnitude
        T = tf.reshape(tf.norm(y[:, 4:6], axis = 1), (-1, 1))

        # LHS of equations of motion (Automatic differentation)
        dx1_dt    = dde.grad.jacobian(y, time, i=0)
        dtheta_dt = dde.grad.jacobian(y, time, i=1)
        dx2_dt    = dde.grad.jacobian(y, time, i=2)
        dx3_dt    = dde.grad.jacobian(y, time, i=3)
        dm_dt     = dde.grad.jacobian(y, time, i=6)

        # RHS of equations of motion
        RHS_x1     = x2
        RHS_theta  = x3/x1
        RHS_x2     = x3**2/x1 - (self.mu * self.time_scale**2 / self.length_scale**3) * x1**(-2) +\
                      (self.time_scale**2/self.length_scale) * ur / m
        RHS_x3     = - (x2*x3)/x1 + (self.time_scale**2/self.length_scale) * ut / m
        RHS_m      = -T * self.time_scale / (self.isp * 9.81)

        # Return the residuals
        return [
            dx1_dt    - RHS_x1,
            dtheta_dt - RHS_theta,
            dx2_dt    - RHS_x2,
            dx3_dt    - RHS_x3,
            dm_dt     - RHS_m,
            ]


class TwoBodyProblemRadialThetaNonDimensionalControl2_mass(Function):
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

    theta = True
    control = True
    mass = True
    entries = 6
    control_entries = 2
    loss_entries = 5
    coordinates = 'radial'
    loss_labels = ["r", r"$\theta$", "v$_r$", r"v$_{\theta}$", "m"]
    entry_labels = ["r", r"$\theta$", "v$_r$ ", r"v$_{\theta}$", "u$_r$", r"u$_{\theta}$"]
    on_off = True

    def __init__(self, data):
        super().__init__(data)

    def call(self, time, y):

        # Coordinate entries
        x1          = y[:, 0:1]
        theta       = y[:,1:2] # Not used, uncoupled from the others
        x2          = y[:, 2:3]
        x3          = y[:, 3:4]

        # Control entries
        on_or_off   = y[:, 6:7]
        on_or_off   = tf.where(on_or_off < 0.5, tf.zeros_like(on_or_off), tf.ones_like(on_or_off))
        ur          = on_or_off* y[:, 4:5]
        ut          = on_or_off* y[:, 5:6]

        # Mass entry
        m           = y[:, 7:8]

        # Thrust magnitude
        T = tf.reshape(tf.norm(y[:, 4:6], axis = 1), (-1, 1))

        # LHS of equations of motion (Automatic differentation)
        dx1_dt    = dde.grad.jacobian(y, time, i=0)
        dtheta_dt = dde.grad.jacobian(y, time, i=1)
        dx2_dt    = dde.grad.jacobian(y, time, i=2)
        dx3_dt    = dde.grad.jacobian(y, time, i=3)
        dm_dt     = dde.grad.jacobian(y, time, i=6)

        # RHS of equations of motion
        RHS_x1     = x2
        RHS_theta  = x3/x1
        RHS_x2     = x3**2/x1 - (self.mu * self.time_scale**2 / self.length_scale**3) * x1**(-2) +\
                      (self.time_scale**2/self.length_scale) * ur / m
        RHS_x3     = - (x2*x3)/x1 + (self.time_scale**2/self.length_scale) * ut / m
        RHS_m      = -T * self.time_scale / (self.isp * 9.81)

        # Return the residuals
        return [
            dx1_dt    - RHS_x1,
            dtheta_dt - RHS_theta,
            dx2_dt    - RHS_x2,
            dx3_dt    - RHS_x3,
            dm_dt     - RHS_m,
            ]



class TwoBodyProblemNonDimensionalControl_mass(Function):
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
    theta = False
    control = True
    mass = True
    entries = 6
    control_entries = 2
    loss_entries = 5
    coordinates = 'NDcartesian'
    loss_labels = ["x", "y", "v$_x$", "v$_y$", "m"]
    entry_labels = ["x", "y", "v$_x$", "v$_y$", "u$_x$", "u$_y$" ]
    on_off = False

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

        m       = y[:, 6:7]

        T = tf.reshape(tf.norm(y[:, 4:6], axis = 1), (-1, 1))

        r = tf.reshape(tf.norm(y[:, 0:2], axis = 1), (-1, 1))

        # Automatic differentation
        dx_dt = dde.grad.jacobian(y, x, i=0)
        dy_dt = dde.grad.jacobian(y, x, i=1)
        dvx_dt = dde.grad.jacobian(y, x, i=2)
        dvy_dt = dde.grad.jacobian(y, x, i=3)
        dm_dt  = dde.grad.jacobian(y, x, i=6)

        RHS_x = vx_tens
        RHS_y  = vy_tens
        RHS_vx  = - (self.mu * self.time_scale**2 / self.length_scale**3) * x_tens * r**(-3) +\
              self.time_scale**2 / (self.length_scale*m) * ux_tens
        RHS_vy  = - (self.mu * self.time_scale**2 / self.length_scale**3) * y_tens * r**(-3) +\
              self.time_scale**2 / (self.length_scale*m) * uy_tens
        RHS_m   = -T * self.time_scale / (self.isp * 9.81)

        return [
            dx_dt - RHS_x,
            dy_dt - RHS_y,
            dvx_dt - RHS_vx,
            dvy_dt - RHS_vy,
            dm_dt  - RHS_m,
            ]

