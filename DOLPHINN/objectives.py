## THomas GOldman 2023
# DOLPHINN

import numpy as np

import deepxde as dde
from deepxde.backend import tf
import tensorflow_probability as tfp

from .function import Function

class Objective(Function):

    def __init__(self, data, mass_included):
        self._entries = len(data['initial_state'])
        self.mass_included = mass_included
        super().__init__(data)

class OptimalFinalMass(Objective):

    def __init__(self,
                data,
                mass_included):

        super().__init__(data, mass_included)

    def call(self, t, y, losses):
        '''
        Calculate consumed mass by integrating the thrust profile.
        Requires a whole batch of input/output pairs.
        '''
        if self.mass_included:
            return 1/y[-1,-1]
        else:
            raise Exception("[DOLPHINN] Optimal final mass objective is selected, but mass is not propagated")

class MaximumRadius(Objective):

    def __init__(self,
                data,
                mass_included):

        super().__init__(data, mass_included)

    def call(self, t, y, losses):
        '''
        Return inverse of final radius
        '''

        return 1/y[-1,0]



class OptimalFuel(Objective):

    def __init__(self,
                 data,
                 mass_included):

        self._entries = len(data['initial_state'])

        super().__init__(data, mass_included)

    def call(self, t, y, losses):
        '''
        Calculate consumed mass by integrating the thrust profile.
        Requires a whole batch of input/output pairs.
        '''

        # Get control and calculate norm
        t = tf.reshape(t, (1, -1))[0] * self.time_scale

        if self.mass_included:
            U = y[:, self._entries:-1]
        else:
            U = y[:, self._entries:]

        U_norm = tf.norm(U, axis=1)

        # Sort time and control
        idx = tf.argsort(t)
        t_sorted = tf.gather(t, idx)
        U_norm_sorted = tf.gather(U_norm, idx)

        # Propellent mass
        propellent_mass = (1/self.isp/9.81) * tfp.math.trapz(U_norm_sorted, t_sorted)

        return propellent_mass


class OptimalTime(Objective):

    def __init__(self,
                data,
                mass_included):

        self._initial_tensor = tf.convert_to_tensor(np.array([data['final_state']]), tf.float32)
        super().__init__(data, mass_included)

    def call(self, t, y, losses):
        '''
        Calculate consumed mass by integrating the thrust profile.
        Requires a whole batch of input/output pairs.
        '''

        # Get control and calculate norm
        states = y[:, :self._entries]

        diffs = tf.subtract(states, self._initial_tensor)
        squared_diffs  = tf.square(diffs)
        column_averages = tf.reduce_mean(squared_diffs, axis=0)
        sum_averages = tf.reduce_sum(column_averages)

        return sum_averages