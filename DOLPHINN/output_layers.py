# Thomas Goldman 2023
# DOLPHINN

import numpy as np

import deepxde as dde
from deepxde.backend import tf
from .function import Function


class InitialStateLayer(Function):
    '''
    Initial state enforced
    '''
    on_off = False

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
    Cartesian Coordinates
    Initial state enforced
    Final state enforced
    Control in x and y in N
    '''
    on_off = False

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
    Radial Coordinates
    Initial state enforced
    Final state enforced
    Control in radial and tangentinal direction in N
    Control flight path angle activated with a Sigmoid
    '''
    on_off = False

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
    Radial Coordinates
    Initial state enforced
    Final state enforced
    Control in radial and tangentinal direction in N
    Control flight path angle activated with a tanh
    '''
    on_off = False

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
    Radial Coordinates
    Initial state enforced
    Final state enforced
    Control in radial and tangentinal direction in N
    Mass output in Kg
    Control flight path angle activated with a tanh
    '''
    on_off = False

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


class InitialFinalStateLayer_Radial2_tanh_mass(Function):
  '''
  Radial Coordinates
  Initial state enforced
  Final state only R and V_r enforced.
  Final V_theta is fixed to circular velocity at R
  Control output in Newton, in radial and tangentinal direction
  Mass output in KG
  '''
  on_off = False

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

      # Force final radial component velocity to be that of the circular velocity at this radius
      final_vt = (self.time_scale/self.length_scale) * tf.math.sqrt(self.mu / (y[-1,0]*self.length_scale))

      output = tf.concat([f1*self.initial_state[0] + f_mass*y[:,0:1],
                          f1*self.initial_state[1] + f2*y[:,1:2] + f3*self.final_state[1],
                          f1*self.initial_state[2] + f2*y[:,2:3] + f3*final_vt,
                          ur,
                          ut,
                          self.m - f_mass * m], axis = 1
                          )

      return output


class InitialFinalStateLayer_RadialTheta_tanh_mass(Function):
  '''
  Radial Coordinates including theta
  Initial state enforced
  Final state enforced
  Control in radial and tangentinal direction in N
  Mass output in Kg
  Control flight path angle activated with a tanh
  '''
  on_off = False

  def __init__(self, data):

      super().__init__(data)

  def call(self, t, y):
      '''
      Mulitply the networks output with a time dependent function
        which vanshises at t=0 and tfinal.
      Mulitply initial state with a time dependent function
        which vanshises at t>0.
      Mulitply final state with a time dependent function
        which vanshises at t<tfinal.
      Transforms the control outputs
      Transforms the mass output

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
      u_norm = tf.math.sigmoid(y[:,4:5])
      u_angle = tf.math.tanh(y[:,5:6])
      m = tf.math.sigmoid(y[:,6:7])
      #theta = self.final_state[1]*tf.math.sigmoid(y[:,1:2])

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
                          f1*self.initial_state[3] + f2*y[:,3:4] + f3*self.final_state[3],
                          ur,
                          ut,
                          self.m - f_mass * m], axis = 1
                          )

      return output

class InitialFinalStateLayer_RadialTheta_sin_mass(Function):
  '''
  Radial Coordinates including theta
  Initial state enforced
  Final state enforced
  Control in radial and tangentinal direction in N
  Mass output in Kg
  Control flight path angle activated with a tanh
  '''
  on_off = False

  def __init__(self, data):

      super().__init__(data)

  def call(self, t, y):
      '''
      Mulitply the networks output with a time dependent function
        which vanshises at t=0 and tfinal.
      Mulitply initial state with a time dependent function
        which vanshises at t>0.
      Mulitply final state with a time dependent function
        which vanshises at t<tfinal.
      Transforms the control outputs
      Transforms the mass output

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
      u_norm = tf.math.sigmoid(y[:,4:5])
      u_angle = tf.math.sin(y[:,5:6])
      m = tf.math.sigmoid(y[:,6:7])
      #theta = self.final_state[1]*tf.math.sigmoid(y[:,1:2])

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
                          f1*self.initial_state[3] + f2*y[:,3:4] + f3*self.final_state[3],
                          ur,
                          ut,
                          self.m - f_mass * m], axis = 1
                          )

      return output



class InitialFinalStateLayer_RadialTheta_tanh_mass2(Function):
  '''
  Radial Coordinates including theta
  Initial state enforced
  Final state enforced
  Control in radial and tangentinal direction in N
  Mass output in Kg
  Control flight path angle activated with a tanh
  '''
  on_off = False

  def __init__(self, data):
    super().__init__(data)

  def call(self, t, y):
    '''
    Mulitply the networks output with a time dependent function
      which vanshises at t=0 and tfinal.
    Mulitply initial state with a time dependent function
      which vanshises at t>0.
    Mulitply final state with a time dependent function
      which vanshises at t<tfinal.
    Transforms the control outputs
    Transforms the mass output

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
    u_norm = tf.math.sigmoid(y[:,4:5])
    u_angle = tf.math.tanh(y[:,5:6])
    m = tf.math.sigmoid(y[:,6:7])
    #theta = self.final_state[1]*tf.math.sigmoid(y[:,1:2])

    # Rescale the U_R and the U_theta to their real values
    u_norm = u_norm * self.umax
    u_angle = u_angle * 2*np.pi
    m =  m * self.max_mass

    # Transform the control to cartesian coordinates
    ur = u_norm * tf.math.sin(u_angle)
    ut = u_norm * tf.math.cos(u_angle)

    output = tf.concat([f1*self.initial_state[0] + f2*y[:,0:1] + f3*self.final_state[0],
                        f1*self.initial_state[1] + f2*y[:,1:2] + f3*self.final_state[1],
                        f1*self.initial_state[2] + f2*y[:,2:3] + f3*self.final_state[2],
                        f1*self.initial_state[3] + f2*y[:,3:4] + f3*self.final_state[3],
                        ur,
                        ut,
                        m], axis = 1
                        )

    return output


class InitialFinalStateLayer_RadialTheta_tanh_mass3(Function):
  '''
  Radial Coordinates including theta
  Initial state enforced
  Final state enforced
  Control in radial and tangentinal direction in N
  Mass output in Kg
  Control flight path angle activated with a tanh
  '''

  on_off = True

  def __init__(self, data):

      super().__init__(data)

  def call(self, t, y):
      '''
      Mulitply the networks output with a time dependent function
        which vanshises at t=0 and tfinal.
      Mulitply initial state with a time dependent function
        which vanshises at t>0.
      Mulitply final state with a time dependent function
        which vanshises at t<tfinal.
      Transforms the control outputs
      Transforms the mass output

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
      u_angle = tf.math.tanh(y[:,4:5])
      on_or_off = tf.math.sigmoid(y[:,5:6])
      #on_or_off = 1/(1 + tf.math.exp(-self.k*y[:,5:6]))
      #on_or_off = tf.where(tf.math.sigmoid(y[:,5:6]) < 0.5, tf.zeros_like(y[:,5:6]), tf.ones_like(y[:,5:6]))
      m = tf.math.sigmoid(y[:,6:7])
      #theta = self.final_state[1]*tf.math.sigmoid(y[:,1:2])

      # Rescale the U_R and the U_theta to their real values
      u_angle = u_angle * 2*np.pi
      m =  m * self.m

      # Transform the control to cartesian coordinates
      ur = self.umax * tf.math.sin(u_angle)
      ut = self.umax * tf.math.cos(u_angle)

      output = tf.concat([f1*self.initial_state[0] + f2*y[:,0:1] + f3*self.final_state[0],
                          f1*self.initial_state[1] + f2*y[:,1:2] + f3*self.final_state[1],
                          f1*self.initial_state[2] + f2*y[:,2:3] + f3*self.final_state[2],
                          f1*self.initial_state[3] + f2*y[:,3:4] + f3*self.final_state[3],
                          ur,
                          ut,
                          on_or_off,
                          self.m - f_mass * m], axis = 1
                          )

      return output



class InitialFinalStateLayer_Cartesian_tanh_mass(Function):
  '''
  Cartesian Coordinates
  Initial state enforced
  Final state enforced
  Control in x and y in N
  '''
  on_off = False

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
      ur = tf.math.sigmoid(y[:,4:5])
      ut = tf.math.tanh(y[:,5:6])
      m  = tf.math.sigmoid(y[:,6:7])

      # Rescale the U_R and the U_theta to their real values
      ur = ur * self.umax
      ut = ut * 2*np.pi
      m =  m * self.m

      # Transform the control to cartesian coordinates
      ux = ur * tf.math.cos(ut)
      uy = ur * tf.math.sin(ut)

      output = tf.concat([f1*self.initial_state[0] + f2*y[:,0:1] + f3*self.final_state[0],
                          f1*self.initial_state[1] + f2*y[:,1:2] + f3*self.final_state[1],
                          f1*self.initial_state[2] + f2*y[:,2:3] + f3*self.final_state[2],
                          f1*self.initial_state[3] + f2*y[:,3:4] + f3*self.final_state[3],
                          ux,
                          uy,
                          self.m - f_mass * m], axis = 1
                          )

      return output

class InitialFinalStateLayer_Cartesian_sin_mass(Function):
  '''
  Cartesian Coordinates
  Initial state enforced
  Final state enforced
  Control in x and y in N
  '''
  on_off = False

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
      ur = tf.math.sigmoid(y[:,4:5])
      ut = tf.math.sin(y[:,5:6])
      m  = tf.math.sigmoid(y[:,6:7])

      # Rescale the U_R and the U_theta to their real values
      ur = ur * self.umax
      ut = ut * 2*np.pi
      m =  m * self.m

      # Transform the control to cartesian coordinates
      ux = ur * tf.math.cos(ut)
      uy = ur * tf.math.sin(ut)

      output = tf.concat([f1*self.initial_state[0] + f2*y[:,0:1] + f3*self.final_state[0],
                          f1*self.initial_state[1] + f2*y[:,1:2] + f3*self.final_state[1],
                          f1*self.initial_state[2] + f2*y[:,2:3] + f3*self.final_state[2],
                          f1*self.initial_state[3] + f2*y[:,3:4] + f3*self.final_state[3],
                          ux,
                          uy,
                          self.m - f_mass * m], axis = 1
                          )

      return output

class InitialFinalStateLayer_RadialCartesian_tanh_mass(Function):
  '''
  Radial Coordinates including theta
  Initial state enforced
  Final state enforced
  Control in radial and tangentinal direction in N
  Mass output in Kg
  Control flight path angle activated with a tanh
  '''

  on_off = False

  def __init__(self, data):

      super().__init__(data)

  def call(self, t, y):
      '''
      Mulitply the networks output with a time dependent function
        which vanshises at t=0 and tfinal.
      Mulitply initial state with a time dependent function
        which vanshises at t>0.
      Mulitply final state with a time dependent function
        which vanshises at t<tfinal.
      Transforms the control outputs
      Transforms the mass output

      Arguments:
          t (tf.Tensor): network input (time)
          y (tf.Tensor): last layers output

      Returns:
          output (tf.Tensor): This layers output

      '''

      # Assume the network produced polar coordinates
      r = y[:,0:1]
      theta = y[:,1:2]
      vr = y[:,2:3]
      vtheta = y[:,3:4]

      # Convert to cartesian
      vx = tf.cos(theta) * vr + tf.sin(theta) * vtheta
      vy = -tf.sin(theta) * vr + tf.cos(theta) * vtheta
      x1 = r*tf.cos(theta)
      x2 = r*tf.sin(theta)

      # Constraint equations
      f1 = tf.math.exp(-self.a*(t-self.t0))
      f2 = 1 - tf.math.exp(-self.a*(t-self.t0)) - tf.math.exp(self.a*(t - self.tfinal))
      f3 = tf.math.exp(self.a*(t-self.tfinal))
      f_mass = 1 - tf.math.exp(-self.a*(t-self.t0))

      # Control and mass: Apply sigmoid to get in [0, 1],
      # while keeping a non-zero derivative for training
      u_norm = tf.math.sigmoid(y[:,4:5])
      u_angle = tf.math.tanh(y[:,5:6])
      m = tf.math.sigmoid(y[:,6:7])

      # Rescale the U_R, U_theta and mass to their real values
      u_norm = u_norm * self.umax
      u_angle = u_angle * 2*np.pi
      m =  m * self.m

      # Transform the control to the LVLH frame
      ur = u_norm * tf.math.sin(u_angle)
      ut = u_norm * tf.math.cos(u_angle)

      # Transform the control to inertial frame
      ux = tf.cos(theta) * ur + tf.sin(theta) * ur
      uy = -tf.sin(theta) * ut + tf.cos(theta) * ut

      output = tf.concat([f1*self.initial_state[0] + f2*x1 + f3*self.final_state[0],
                          f1*self.initial_state[1] + f2*x2 + f3*self.final_state[1],
                          f1*self.initial_state[2] + f2*vx + f3*self.final_state[2],
                          f1*self.initial_state[3] + f2*vy + f3*self.final_state[3],
                          ux,
                          uy,
                          self.m - f_mass * m], axis = 1
                          )

      return output

class InitialFinalStateLayer_TranslatedRadial_tanh_mass(Function):
  '''
  Radial Coordinates including theta
  Initial state enforced
  Final state enforced
  Control in radial and tangentinal direction in N
  Mass output in Kg
  Control flight path angle activated with a tanh
  '''
  on_off = False

  def __init__(self, data):

    super().__init__(data)

  def call(self, t, y):
    '''
    Mulitply the networks output with a time dependent function
    which vanshises at t=0 and tfinal.
    Mulitply initial state with a time dependent function
    which vanshises at t>0.
    Mulitply final state with a time dependent function
    which vanshises at t<tfinal.
    Transforms the control outputs
    Transforms the mass output

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

    r_prime      = y[:,0:1]
    theta_prime  = y[:,1:2]

    r_prime = f1*self.initial_state[0] + f2*r_prime + f3*self.final_state[0]
    theta_prime = f1*self.initial_state[1] + f2*theta_prime + f3*self.final_state[1]

    r = tf.math.sqrt(self.offset**2 + r_prime**2 - 2 * self.offset*r_prime*tf.math.cos(theta_prime))
    theta = tf.math.acos(r**2 + self.offset**2 - r_prime**2 / (2 * self.offset*r))
    mask = tf.math.less(theta_prime, 0)
    pi = tf.constant([np.pi], dtype=tf.float32)
    theta = tf.where(mask, theta + pi, theta)

    # Apply sigmoid to get in [0, 1], while keeping a non-zero derivative for training
    u_norm = tf.math.sigmoid(y[:,4:5])
    u_angle = tf.math.tanh(y[:,5:6])
    m = tf.math.sigmoid(y[:,6:7])
    #theta = self.final_state[1]*tf.math.sigmoid(y[:,1:2])

    # Rescale the U_R and the U_theta to their real values
    u_norm = u_norm * self.umax
    u_angle = u_angle * 2*np.pi
    m =  m * self.m

    # Transform the control to cartesian coordinates
    ur = u_norm * tf.math.sin(u_angle)
    ut = u_norm * tf.math.cos(u_angle)

    output = tf.concat([r,
                       theta,
                       f1*self.initial_state[2] + f2*y[:,2:3] + f3*self.final_state[2],
                       f1*self.initial_state[3] + f2*y[:,3:4] + f3*self.final_state[3],
                       ur,
                       ut,
                       self.m - f_mass * m], axis = 1
                      )

    return output


class Rastrigin(Function):
  '''
  Verification function to optimise
  '''
  on_off = False

  def __init__(self, data):

    super().__init__(data)

  def call(self, t, y):
    '''
    Mulitply the networks output with a time dependent function
    which vanshises at the edges

    Arguments:
      t (tf.Tensor): network input (time)
      y (tf.Tensor): last layers output

    Returns:
      output (tf.Tensor): This layers output

    '''

    # Unpack tensors
    x1  = 5.12*tf.math.sin(y[:, 0:1])
    x2  = 5.12*tf.math.sin(y[:, 1:2])
    f_est_tens = y[:, 2:3]

    output = tf.concat([y[:, 0:1],
                        y[:, 1:2],
                        f_est_tens
                        ], axis = 1)

    return output


class Himmelblau(Function):
  '''
  Verification function to optimise
  '''
  on_off = False

  def __init__(self, data):

    super().__init__(data)

  def call(self, t, y):
    '''
    Mulitply the networks output with a time dependent function
    which vanshises at the edges

    Arguments:
      t (tf.Tensor): network input (time)
      y (tf.Tensor): last layers output

    Returns:
      output (tf.Tensor): This layers output

    '''

    f1 = tf.math.exp(-self.a*(t-self.t0))
    f2 = 1 - tf.math.exp(-self.a*(t-self.t0)) - tf.math.exp(self.a*(t - self.tfinal))
    f3 = tf.math.exp(self.a*(t-self.tfinal))

    # Unpack tensors
    x1  = y[:, 0:1]
    x2  = y[:, 1:2]
    f_est_tens = y[:, 2:3]

    output = tf.concat([f1*(-6) + f2*x1 + f3*6,
                        f1*(-6) + f2*x2 + f3*6,
                        f_est_tens
                        ], axis = 1)

    return output

class InitialStateLayer_linear(Function):
    '''
    Initial state enforced
    '''
    on_off = False

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
        f = t
        output = tf.concat([self.initial_state[i] + f*y[:,i:i+1] for i in range(len(self.initial_state))],
                           axis = 1)

        return output