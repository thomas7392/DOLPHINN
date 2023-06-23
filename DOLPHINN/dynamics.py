
import deepxde as dde
from deepxde.backend import tf


class Dynamics:

    def __init__(self,
                m,
                mu,
                length_scale,
                time_scale):

        self.m = m
        self.mu = mu
        self.time_scale = time_scale
        self.length_scale = length_scale
        self.dynamics_identifier = self.__class__.__name__

    def call_loss(self, x, y):
         raise NotImplementedError(
            "call_loss methods to be implemented"
        )


class TwoBodyProblemNoneDimensional(Dynamics):
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

    def __init__(self,
                 m,
                 mu,
                 length_scale,
                 time_scale):

        super().__init__(m, mu, length_scale, time_scale)

    def call_loss(self, x, y):


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

class TwoBodyProblemNonDimensionalControl(Dynamics):
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

    def __init__(self,
                m,
                mu,
                length_scale,
                time_scale):

        super().__init__(m, mu, length_scale, time_scale)

    def call_loss(self, x, y):

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
        RHS_vx  = - (self.mu * self.time_scale**2 / self.length_scale**3) * x_tens * r**(-3) + self.time_scale**2 / (self.length_scale*m) * ux_tens
        RHS_vy  = - (self.mu * self.time_scale**2 / self.length_scale**3) * y_tens * r**(-3) + self.time_scale**2 / (self.length_scale*m) * uy_tens

        return [
            dx_dt - RHS_x,
            dy_dt - RHS_y,
            dvx_dt - RHS_vx,
            dvy_dt - RHS_vy,
            ]



class TwoBodyProblemRadialNonDimensional:

    def __init__(self,
                m,
                mu,
                length_scale,
                time_scale):

        super().__init__(m, mu, length_scale, time_scale)

    def call_loss(self, time, y):
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

class TwoBodyProblemRadialNonDimensionalControl:

    def __init__(self,
                m,
                mu,
                length_scale,
                time_scale):

        super().__init__(m, mu, length_scale, time_scale)

    def call_loss(self, time, y):
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
            loss (list): Residial of the individual equations of motion
        '''

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
        RHS_x2  = x3**2/x1 - (self.mu * self.time_scale**2 / self.length_scale**3) * x1**(-2) + (self.time_scale**2/self.length_scale) * ur / m
        RHS_x3  = - (x2*x3)/x1 + (self.time_scale**2/self.length_scale) * ut / m

        return [
            dx1_dt - RHS_x1,
            dx2_dt - RHS_x2,
            dx3_dt - RHS_x3,
            ]
