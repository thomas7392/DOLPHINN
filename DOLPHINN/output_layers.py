# Thomas Goldman 2023
# DOLPHINN

import numpy as np

import deepxde as dde
from deepxde.backend import tf
from .function import Function


class InitialStateLayer(Function):
    '''
    Enforce initial condiation in a network layer
    '''

    def __init__(self, data):
        super().__init__(data)

    def call(self, t, y):
        '''
        Mulitply the networks output with a time dependent function
          which vanshises at t=0. Add the initial state.

        Arguments:
            t (tf.Tensor): network input (time)
            y (tf.Tensor): last layers output

        Returns:
            output (tf.Tensor):  This layers output

        '''
        f = 1 - tf.math.exp(-t)
        output = tf.concat([self.initial_state[i] + f*y[:,i:i+1] for i in range(len(self.initial_state))],
                           axis = 1)

        return output


class InitialFinalStateLayer_Cartesian(Function):
    '''
    Enforce initial condiation in a network layer
    '''

    def __init__(self, data):

        super().__init__(data)

    def call(self, t, y):
        '''
        Mulitply the networks output with a time dependent function
          which vanshises at t=0 and tfinal.
        Mulitply initial state with a time dependent function
          which vanshises at t>0 and tfinal.
        Mulitply final state with a time dependent function
          which vanshises at t<tfinal and tfinal.

        Arguments:
            t (tf.Tensor): network input (time)
            y (tf.Tensor): last layers output

        Returns:
            output (tf.Tensor): This layers output
        '''

        f1 = tf.math.exp(-self.a*(t-self.t0))
        f2 = 1 - tf.math.exp(-self.a*(t-self.t0)) - tf.math.exp(self.a*(t - self.tfinal))
        f3 = tf.math.exp(self.a*(t-self.tfinal))

        # Apply sigmoid to get in [0, 1], while keeping a non-zero derivative for training
        ur = tf.math.sigmoid(y[:,4:5])
        ut = tf.math.sigmoid(y[:,5:6])

        # Rescale the U_R and the U_theta to their real values
        ur = ur * self.umax
        ut = ut * 2*np.pi

        # Transform the control to cartesian coordinates
        ux = ur * tf.math.cos(ut)
        uy = ur * tf.math.sin(ut)

        output = tf.concat([f1*self.initial_state[0] + f2*y[:,0:1] + f3*self.final_state[0],
                            f1*self.initial_state[1] + f2*y[:,1:2] + f3*self.final_state[1],
                            f1*self.initial_state[2] + f2*y[:,2:3] + f3*self.final_state[2],
                            f1*self.initial_state[3] + f2*y[:,3:4] + f3*self.final_state[3],
                            ux,
                            uy], axis = 1
                            )

        return output



class InitialFinalStateLayer_Radial(Function):
    '''
    Enforce initial condiation in a network layer
    '''

    def __init__(self, data):

        super().__init__(data)

    def call(self, t, y):
        '''
        Mulitply the networks output with a time dependent function
          which vanshises at t=0 and tfinal.
        Mulitply initial state with a time dependent function
          which vanshises at t>0 and tfinal.
        Mulitply final state with a time dependent function
          which vanshises at t<tfinal and tfinal.

        Arguments:
            t (tf.Tensor): network input (time)
            y (tf.Tensor): last layers output

        Returns:
            output (tf.Tensor): This layers output

        '''

        f1 = tf.math.exp(-self.a*(t-self.t0))
        f2 = 1 - tf.math.exp(-self.a*(t-self.t0)) - tf.math.exp(self.a*(t - self.tfinal))
        f3 = tf.math.exp(self.a*(t-self.tfinal))

        # Apply sigmoid to get in [0, 1], while keeping a non-zero derivative for training
        u_norm = tf.math.sigmoid(y[:,3:4])
        u_angle = tf.math.sigmoid(y[:,4:5])

        # Rescale the U_R and the U_theta to their real values
        u_norm = u_norm * self.umax
        u_angle = u_angle * 2*np.pi

        # Transform the control to cartesian coordinates
        ur = u_norm*tf.math.sin(u_angle)
        ut = u_norm*tf.math.cos(u_angle)

        output = tf.concat([f1*self.initial_state[0] + f2*y[:,0:1] + f3*self.final_state[0],
                            f1*self.initial_state[1] + f2*y[:,1:2] + f3*self.final_state[1],
                            f1*self.initial_state[2] + f2*y[:,2:3] + f3*self.final_state[2],
                            ur,
                            ut], axis = 1
                            )

        return output


class InitialFinalStateLayer_Radial_tanh(Function):
    '''
    Enforce initial condiation in a network layer
    '''

    def __init__(self, data):

        super().__init__(data)

    def call(self, t, y):
        '''
        Mulitply the networks output with a time dependent function
          which vanshises at t=0 and tfinal.
        Mulitply initial state with a time dependent function
          which vanshises at t>0 and tfinal.
        Mulitply final state with a time dependent function
          which vanshises at t<tfinal and tfinal.

        Arguments:
            t (tf.Tensor): network input (time)
            y (tf.Tensor): last layers output

        Returns:
            output (tf.Tensor): This layers output

        '''

        f1 = tf.math.exp(-self.a*(t-self.t0))
        f2 = 1 - tf.math.exp(-self.a*(t-self.t0)) - tf.math.exp(self.a*(t - self.tfinal))
        f3 = tf.math.exp(self.a*(t-self.tfinal))

        # Apply sigmoid to get in [0, 1], while keeping a non-zero derivative for training
        u_norm = tf.math.sigmoid(y[:,3:4])
        u_angle = tf.math.tanh(y[:,4:5])

        # Rescale the U_R and the U_theta to their real values
        u_norm = u_norm * self.umax
        u_angle = u_angle * 2*np.pi

        # Transform the control to cartesian coordinates
        ur = u_norm * tf.math.sin(u_angle)
        ut = u_norm * tf.math.cos(u_angle)

        output = tf.concat([f1*self.initial_state[0] + f2*y[:,0:1] + f3*self.final_state[0],
                            f1*self.initial_state[1] + f2*y[:,1:2] + f3*self.final_state[1],
                            f1*self.initial_state[2] + f2*y[:,2:3] + f3*self.final_state[2],
                            ur,
                            ut], axis = 1
                            )

        return output

class InitialFinalStateLayer_Radial_tanh_mass(Function):
    '''
    Enforce initial condiation in a network layer
    '''

    def __init__(self, data):

        super().__init__(data)

    def call(self, t, y):
        '''
        Mulitply the networks output with a time dependent function
          which vanshises at t=0 and tfinal.
        Mulitply initial state with a time dependent function
          which vanshises at t>0 and tfinal.
        Mulitply final state with a time dependent function
          which vanshises at t<tfinal and tfinal.

        Arguments:
            t (tf.Tensor): network input (time)
            y (tf.Tensor): last layers output

        Returns:
            output (tf.Tensor): This layers output

        '''

        f1 = tf.math.exp(-self.a*(t-self.t0))
        f2 = 1 - tf.math.exp(-self.a*(t-self.t0)) - tf.math.exp(self.a*(t - self.tfinal))
        f3 = tf.math.exp(self.a*(t-self.tfinal))
        f_mass = 1 - tf.math.exp(-self.a*(t-self.t0))

        # Apply sigmoid to get in [0, 1], while keeping a non-zero derivative for training
        u_norm = tf.math.sigmoid(y[:,3:4])
        u_angle = tf.math.tanh(y[:,4:5])
        m = tf.math.sigmoid(y[:,5:6])

        # Rescale the U_R and the U_theta to their real values
        u_norm = u_norm * self.umax
        u_angle = u_angle * 2*np.pi
        m =  m * self.m

        # Transform the control to cartesian coordinates
        ur = u_norm * tf.math.sin(u_angle)
        ut = u_norm * tf.math.cos(u_angle)

        output = tf.concat([f1*self.initial_state[0] + f2*y[:,0:1] + f3*self.final_state[0],
                            f1*self.initial_state[1] + f2*y[:,1:2] + f3*self.final_state[1],
                            f1*self.initial_state[2] + f2*y[:,2:3] + f3*self.final_state[2],
                            ur,
                            ut,
                            self.m - f_mass * m], axis = 1
                            )

        return output

