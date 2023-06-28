# Thomas Goldman 2023
# DOLPHINN

import numpy as np
import time
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

# class LowThrustGuidance:

#     def __init__(self, DOLPHINN):

#         self.DOLPHINN = DOLPHINN
#         self.is_cartesian = self.DOLPHINN.dynamics.coordinates
#         if not self.is_cartesian:
#             self.to_cartesian = getattr(get_dynamics_info_from_config, f"{self.DOLPHINN.dynamics.coordinates}_to_cartesian")
#         self.control_entries = self.DOLPHINN.dynamics.control_entries
#         self.state_entries = self.DOLPIHNN.dynamics.entries


#         # Allow to retrieve theta if the NN createa radial coordinates
#         if self.DOLPHINN.dynamics.coordinates == "radial":

#             dummy_time = np.linspace(self.DOLPHINN.data['t0'], self.DOLPHINN.data['tf'], 1000)
#             dummy_time_tensor = tf.convert_to_tensor(dummy_time, tf.float32)
#             y = self.DOLPHINN.model.predict(dummy_time_tensor)
#             theta = integrate_theta(dummy_time, y)

#             self.theta_interpolater = CubicSpline(dummy_time, theta)

#     def call_dolphinn(self, time):

#         time = time/self.DOLPHINN.data['time_scale']
#         time_tensor = tf.convert_to_tensor([[time]], tf.float32)
#         state = self.DOLPHINN.model.predict(time_tensor)

#         if self.DOLPHINN.dynamics.coordinates == "radial":
#             dummy_state = np.concatenate((np.array([[time]]), state[0, 0:1], self.theta_interpolator(time), state[0,1:]), axis = 1)


#         elif not self.is_cartesian:
#             dummy_state = np.zeros(self.state_entries)
#             dummy_state[-self.control_entries:] = control

#         dummy_state_cartesian = self.to_cartesian(dummy_state)[:-self.control_entries]



class LowThrustGuidance:
    '''
    Contains laws for controlling the thrust
    '''

    def __init__(self,
                 control_nodes,
                 bodies):
        '''
        Initialize the Guidance from control nodes, which consist of
        a dictionary mapping time t to control node [u1, u2]

        Args:
            control_nodes (dict):       Dictionary mapping times to control nodes
                                        in an intertial frame.

        '''

        self.control_nodes = control_nodes
        self.bodies = bodies
        self.control_interpolator = CubicSpline(np.array(list(self.control_nodes.keys())),
                                                np.array(list(self.control_nodes.values())))

    def getAngles(self, time: float):
        return np.array([0, 0, 0]).reshape(-1, 1)

    def getThrustMagnitude(self, time):
        '''
        Thrust magnitude depending on time

        Args:
            Time (float):  Time
        returns:
            magnitude (float):  Thrust magnitude
        '''

        inertial_control = self.control_interpolator(time)
        magnitude = np.linalg.norm(inertial_control)

        return magnitude


    def getThrustDirection(self, time):
        '''
        Thrust direction as a unit vector in the body fixed frame

        Args:
            Time (float):  Time
        returns:
            magnitude (np.array):  Thrust direction in the body frame
        '''


        # Get intertial control vector
        inertial_control = self.control_interpolator(time)
        if len(inertial_control) == 2:
            np.append(inertial_control, 0)
        intertial_control = intertial_control.reshape(-1, 1)

        # Get intertial to body frame rotation
        rotation_matrix = self.bodies.get("Vehicle").inertial_to_body_fixed_frame

        # Perform rotation and normalize
        bodyframe_control = rotation_matrix @ intertial_control
        direction = 1/(np.linalg.norm(bodyframe_control)) * bodyframe_control

        return  direction

class Verification:
    '''
    Create a verification numerical integration with TUDAT
    '''

    def __init__(self,
                 m,
                 t0,
                 tfinal,
                 initial_state,
                 isp,
                 central_body = "Sun",
                 control_nodes = None,
                 dolphinn_control_law = None,
                 original_coordinates = "cartesian",
                 verbose = True):
        '''
        Standard initializer
        '''

        self.m = m
        self.t0 = t0
        self.tfinal = tfinal
        self.initial_state = initial_state
        self.central_body = central_body
        self.control_nodes = control_nodes
        self.isp = isp
        self.verbose = verbose

        if self.verbose:
            print("[DOLPHINN] Setting up the TUDAT simulation")
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

        if self.control_nodes:

            if verbose:
                print("[DOLPHINN] Guidance is internal!")
            # Create guidance
            GuidanceModel = LowThrustGuidance(self.control_nodes, self.bodies)
            thrust_magnitude_function = GuidanceModel.getThrustMagnitude
            thrust_direction_funtion = GuidanceModel.getThrustDirection
            thrust_magnitude_settings = propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(thrust_magnitude_function,
                                                                                                specific_impulse = self.isp )

            # Create Engine
            environment_setup.add_variable_direction_engine_model("Vehicle",
                                                                 "MainEngine",
                                                                 thrust_magnitude_settings,
                                                                 self.bodies,
                                                                 thrust_direction_funtion)


            aerodynamic_angle_function = GuidanceModel.getAngles
            rotation_model_settings = environment_setup.rotation_model.aerodynamic_angle_based(
                         "Sun", global_frame_orientation, "Vehicle_fixed", aerodynamic_angle_function)
            environment_setup.add_rotation_model(self.bodies, "Vehicle", rotation_model_settings )

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
        # Initial time & Termination
        #============================

        simulation_start_epoch = self.t0 * constants.JULIAN_DAY / (24*60*60)
        simulation_end_epoch = simulation_start_epoch + self.tfinal * constants.JULIAN_DAY / (24*60*60)
        termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

        #============================
        # Integration settings
        #============================

        # control_settings = propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(1.0E-10, 1.0E-10)
        # integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
        #     initial_time_step = 1,
        #     coefficient_set = propagation_setup.integrator.RKCoefficientSets.rkf_78,
        #     step_size_control_settings = control_settings)

        fixed_step_size = (simulation_end_epoch - simulation_start_epoch)/2000
        fixed_step_size = 5000
        integrator_settings = propagation_setup.integrator.runge_kutta_4(fixed_step_size)

        # Andd its ready to go!
        self.propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            self.initial_state,
            simulation_start_epoch,
            integrator_settings,
            termination_settings
        )

    def integrate(self):

        if self.verbose:
            print("[DOLPHINN] Start Integrating")

        start = time.time()
        # Create simulation object and propagate the dynamics
        self.dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                            self.bodies, self.propagator_settings)
        end = time.time()

        if self.verbose:
            print(f"[DOLPHINN] Finished integrating in {end-start} s")

        self.state_history = self.dynamics_simulator.state_history
        self.states = result2array(self.state_history)


    def calculate_coordinates(self, coordinates):

        pass

    @classmethod
    def from_config_states(cls,
                           states,
                           config,
                           verbose = True):

        # Get the dynamics info
        central_body, _, entries,\
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
        control                 = cartesian_states[:,-control_entries:]
        initial_state           = cartesian_states[0,1:-control_entries].reshape(-1, 1)

        print(t0, tfinal)

        # Make 3D initial state
        if initial_state.shape[0] == 4:
            initial_state = np.concatenate((initial_state[0:2,:],
                                            np.array([[0]]),
                                            initial_state[2:4,:],
                                            np.array([[0]])),
                                            axis = 0).reshape(-1, 1)

        # Prepare control nodes
        if control_entries:
            control_nodes = {key: value for key, value in zip(cartesian_states[:,0], control)}
        else:
            control_nodes = None

        return cls(config['m'],
                    t0,
                    tfinal,
                    initial_state,
                    config['isp'],
                    central_body = central_body,
                    control_nodes = control_nodes,
                    original_coordinates = coordinates)

    @classmethod
    def from_DOLPHINN(cls, DOLPHINN):
        '''
        Initialize from a DOLPHINN class instance
        '''

        return cls.from_config_states(DOLPHINN.states,
                                      DOLPHINN.config,
                                      verbose = DOLPHINN.base_verbose)







