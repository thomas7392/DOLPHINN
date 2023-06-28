# Thomas Goldman 2023
# DOLPHINN

import numpy as np
import tensorflow as tf
from scipy.interpolate import CubicSpline


from .utils import get_dynamics_info_from_config, integrate_theta
from . import coordinate_transformations

from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
from tudatpy.kernel import constants
from tudatpy.util import result2array
from tudatpy.kernel.math import interpolators

spice.load_standard_kernels()


class LowThrustGuidance:

    def __init__(self, DOLPHINN):

        self.DOLPHINN = DOLPHINN
        self.is_cartesian = self.DOLPHINN.dynamics.coordinates
        if not self.is_cartesian:
            self.to_cartesian = getattr(get_dynamics_info_from_config, f"{self.DOLPHINN.dynamics.coordinates}_to_cartesian")
        self.control_entries = self.DOLPHINN.dynamics.control_entries
        self.state_entries = self.DOLPIHNN.dynamics.entries


        # Allow to retrieve theta if the NN createa radial coordinates
        if self.DOLPHINN.dynamics.coordinates == "radial":

            dummy_time = np.linspace(self.DOLPHINN.data['t0'], self.DOLPHINN.data['tf'], 1000)
            dummy_time_tensor = tf.convert_to_tensor(dummy_time, tf.float32)
            y = self.DOLPHINN.model.predict(dummy_time_tensor)
            theta = integrate_theta(dummy_time, y)

            self.theta_interpolater = CubicSpline(dummy_time, theta)

    def call_dolphinn(self, time):

        time = time/self.DOLPHINN.data['time_scale']
        time_tensor = tf.convert_to_tensor([[time]], tf.float32)
        state = self.DOLPHINN.model.predict(time_tensor)

        if self.DOLPHINN.dynamics.coordinates == "radial":
            dummy_state = np.concatenate((np.array([[time]]), state[0, 0:1], self.theta_interpolator(time), state[0,1:]), axis = 1)


        elif not self.is_cartesian:
            dummy_state = np.zeros(self.state_entries)
            dummy_state[-self.control_entries:] = control

        dummy_state_cartesian = self.to_cartesian(dummy_state)[:-self.control_entries]












class Verification:
    '''
    Create a verification numerical integration with TUDAT
    '''

    def __init__(self,
                 m,
                 t0,
                 tfinal,
                 initial_state,
                 central_body = "Sun",
                 control_nodes = None,
                 original_coordinates = "cartesian"):
        '''
        Standard initializer
        '''

        self.m = m
        self.t0 = t0
        self.tfinal = tfinal
        self.y0 = initial_state
        self.central_body = central_body
        self.control_nodes = control_nodes

        #====================
        # Create environment
        #====================

        # Create Central body
        bodies_to_create = [self.central_body]
        global_frame_origin = self.central_body
        global_frame_orientation = 'ECLIPJ2000'
        body_settings = environment_setup.get_default_body_settings(
                    bodies_to_create, global_frame_origin, global_frame_orientation)
        self.bodies = environment_setup.create_system_of_bodies(body_settings)

        # Create spacecraft
        self.bodies.create_empty_body('Vehicle')
        self.bodies.get_body('Vehicle').mass = self.m

        #====================
        # Setup Propagation
        #====================

        bodies_to_propagate = ["Vehicle"]
        central_bodies = [self.central_body]
        acceleration_settings_on_vehicle = {self.central_body: [propagation_setup.acceleration.point_mass_gravity()],
                                            "Vehicle": [  propagation_setup.acceleration.thrust_from_engine( 'MainEngine')]}
        acceleration_settings = {"Vehicle": acceleration_settings_on_vehicle}
        acceleration_models = propagation_setup.create_acceleration_models(
                self.bodies, acceleration_settings, bodies_to_propagate, central_bodies)

        #============================
        # Initial state & Termination
        #============================

        initial_state = np.zeros(6)
        initial_state[:2] = self.y0[:2]
        initial_state[3:5] = self.y0[2:]
        simulation_start_epoch = self.t0 * constants.JULIAN_DAY / (24*60*60)
        simulation_end_epoch = self.t0 + self.tfinal * constants.JULIAN_DAY / (24*60*60)
        termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

        #============================
        # Integration settings
        #============================

        control_settings = propagation_setup.integator.step_size_control_elementwise_scalar_tolerance(1.0E-10, 1.0E-10)
        integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
            initial_time_step = 1,
            coefficient_set = propagation_setup.integrator.RKCoefficientSets.rkf_78,
            step_size_control_settings = control_settings)


        # Andd its ready to go!
        self.propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            initial_state,
            simulation_start_epoch,
            integrator_settings,
            termination_settings
        )


    def convert_coordinates(self, coordinates):
        pass

    def integrate(self):

        # Create simulation object and propagate the dynamics
        self.dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                            self.bodies, self.propagator_settings
                            )

        self.states = self.dynamics_simulator.state_history
        self.states = result2array(self.states)


    @classmethod
    def from_config_states(cls,
                           states,
                           config):

        # Get the dynamics info
        central_body, control, entries,\
        control_entries, coordinates = get_dynamics_info_from_config(config)

        # Transform to cartesian
        if "cartesian" not in list(states.keys()):
            transformation = getattr(coordinate_transformations, f"{coordinates}_to_cartesian")
            cartesian_states = transformation(states[coordinates], config)
        else:
            cartesian_states = states["cartesian"]

        # Define information for TUDAT
        t0                      = cartesian_states[0,0]
        tfinal                  = cartesian_states[-1,0]
        initial_state           = cartesian_states[0,1:-control_entries]
        control                 = cartesian_states[:,-control_entries:]

        # Prepare control nodes
        if control:
            control_nodes = {key: value for key, value in zip(cartesian_states[:,0], control)}
        else:
            control_nodes = None

        return cls(config['m'],
                    t0,
                    tfinal,
                    initial_state,
                    central_body = central_body,
                    control_nodes = control_nodes,
                    original_coordinates = coordinates)

    @classmethod
    def from_DOLPHINN(cls, DOLPHINN):
        '''
        Initialize from a DOLPHINN class instance
        '''

        return cls.from_config_states(DOLPHINN.states,
                                      DOLPHINN.config)







