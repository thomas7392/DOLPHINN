import deepxde as dde
from deepxde.backend import tf
import tensorflow_probability as tfp

from .function import Function


class OptimalFuel(Function):

    def __init__(self, data):
        super().__init__(data)

    def call(self, t, y, losses):
        '''
        Calculate consumed mass by integrating the thrust profile.
        Requires a whole batch of input/output pairs.
        '''

        # Get control and calculate norm
        t = tf.reshape(t, (1, -1))[0] * self.time_scale
        U = y[:, 4:6]
        U_norm = tf.norm(U, axis=1)

        # Sort time and control
        idx = tf.argsort(t)
        t_sorted = tf.gather(t, idx)
        U_norm_sorted = tf.gather(U_norm, idx)

        # Propellent mass
        propellent_mass = (1/self.isp/9.81) * tfp.math.trapz(U_norm_sorted, t_sorted)

        return propellent_mass