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


class OptimalFuelSquared(Objective):

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

        return propellent_mass**2


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

class Rastragin(Objective):

    def __init__(self,
                data,
                mass_included):

        super().__init__(data, mass_included)

    def call(self, t, y, losses):
        '''
        Calculate consumed mass by integrating the thrust profile.
        Requires a whole batch of input/output pairs.
        '''

        # Unpack tensors
        x1  = y[0, 0]
        x2  = y[0, 1]
        A = 10

        x1 -= self.x1_offset
        x2 -= self.x2_offset

        f = self.offset + A*2 + (x1**2 - A * tf.math.cos(2*np.pi * x1))\
                + (x2**2 - A * tf.math.cos(2*np.pi * x2))

        return f

class Himmelblau(Objective):

    def __init__(self,
                data,
                mass_included):

        super().__init__(data, mass_included)

    def call(self, t, y, losses):
        '''
        Calculate consumed mass by integrating the thrust profile.
        Requires a whole batch of input/output pairs.
        '''

        # Unpack tensors
        x1  = y[0, 0]
        x2  = y[0, 1]

        f = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

        return f

class Rastragin2(Objective):

    def __init__(self,
                data,
                mass_included):

        super().__init__(data, mass_included)

    def call(self, t, y, losses):
        '''
        Calculate consumed mass by integrating the thrust profile.
        Requires a whole batch of input/output pairs.
        '''
        return y[0,-1]