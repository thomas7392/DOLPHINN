# Thomas Goldman 2023
# DOLPHINN

import os, sys
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


    def getThrustMagnitude(self, time):
        '''
        Thrust magnitude depending on time

        Args:
            Time (float):  Time
        returns:
            magnitude (float):  Thrust magnitude
        '''

        # print(f"Magnitude: called with time {time}")
        if (time == time):
            inertial_control = self.control_interpolator(time)
            magnitude = np.linalg.norm(inertial_control)
            # print("Magnitude: returning ", magnitude)
            return magnitude
        else:
            # print("Magnitude: returning nan")
            return np.nan


    def getInertialThrustDirection(self, time):
        '''
        Thrust magnitude depending on time

        Args:
            Time (float):  Time
        returns:
            magnitude (float):  Thrust magnitude
        '''

        # print(f"Direction: called with time {time}")

        if (time == time):
            # Get intertial control vector
            inertial_control = self.control_interpolator(time)
            if len(inertial_control) == 2:
                inertial_control = np.append(inertial_control, 0)
            inertial_control = inertial_control.reshape(-1, 1)

            # Normalize vector (unit vector)
            direction = 1/(np.linalg.norm(inertial_control)) * inertial_control
            # print("Direction: returning ", direction)
            return  direction
        else:
            # print("Direction: returning nan")
            return np.nan


class Verification:
    '''
    Create a verification numerical integration with TUDAT
    '''

    def __init__(self,
                 m,
                 t0,
                 tfinal,
                 initial_state,
                 isp=None,
                 central_body = "Sun",
                 control_nodes = None,
                 verbose = True,
                 ref_times = None,
                 mass_rate = False,
                 integrator = None):
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
        self.ref_times = ref_times
        self.mass_rate = mass_rate

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

            # Create Thrust Magnitude/Direction guidance
            self.GuidanceModel = LowThrustGuidance(self.control_nodes, self.bodies)
            thrust_magnitude_function = self.GuidanceModel.getThrustMagnitude
            thrust_direction_function = self.GuidanceModel.getInertialThrustDirection

            # Create rototation model: aim body fixed frame at inertial thrust direction
            rotation_model_settings = environment_setup.rotation_model.custom_inertial_direction_based(thrust_direction_function,
                                                                                                        global_frame_orientation,
                                                                                                        "VehicleFixed",
                                                                                                        )
            environment_setup.add_rotation_model(self.bodies,
                                                    "Vehicle",
                                                    rotation_model_settings,
                                                )


            # Create Engine with variable thrust
            thrust_magnitude_settings = propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(thrust_magnitude_function,
                                                                                                    specific_impulse = self.isp)

            # Fix thrust in a constant direction in the body fixed frame
            environment_setup.add_engine_model("Vehicle",
                                                "MainEngine",
                                                thrust_magnitude_settings,
                                                self.bodies,
                                                body_fixed_thrust_direction = np.array([-1, 0, 0]).reshape(-1, 1))

        #====================
        # Setup Propagation
        #====================

        bodies_to_propagate = ["Vehicle"]
        central_bodies = [self.central_body]
        acceleration_settings_on_vehicle = {self.central_body: [propagation_setup.acceleration.point_mass_gravity()]}

        if control_nodes:
            acceleration_settings_on_vehicle["Vehicle"] = [propagation_setup.acceleration.thrust_from_engine('MainEngine')]

        acceleration_settings = {"Vehicle": acceleration_settings_on_vehicle}
        acceleration_models = propagation_setup.create_acceleration_models(
                self.bodies, acceleration_settings, bodies_to_propagate, central_bodies)

        #============================
        # Initial time & Termination
        #============================

        self.simulation_start_epoch = self.t0 * constants.JULIAN_DAY / (24*60*60)
        self.simulation_end_epoch = self.simulation_start_epoch + self.tfinal * constants.JULIAN_DAY / (24*60*60)
        termination_settings = propagation_setup.propagator.time_termination(self.simulation_end_epoch)


        #============================
        # Integration settings
        #============================


        if not integrator:
            current_coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_78
            integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(1.0,
                                                                            current_coefficient_set,
                                                                            1.0E-4,
                                                                            np.inf,
                                                                            1e-10,
                                                                            1e-10)
        else:
            integrator_settings = integrator


        # Andd its ready to go!
        self.propagator_settings_translational = propagation_setup.propagator.translational(
                central_bodies,
                acceleration_models,
                bodies_to_propagate,
                self.initial_state,
                self.simulation_start_epoch,
                integrator_settings,
                termination_settings
        )

        # Potentially add mass rate model
        if self.mass_rate:

            # Create mass rate model
            mass_rate_settings_on_vehicle = {'Vehicle': [propagation_setup.mass_rate.from_thrust()]}
            mass_rate_models = propagation_setup.create_mass_rate_models(self.bodies,
                                                                        mass_rate_settings_on_vehicle,
                                                                        acceleration_models)

            # Create mass propagator settings
            self.mass_propagator_settings = propagation_setup.propagator.mass(bodies_to_propagate,
                                                                        mass_rate_models,
                                                                        np.array([self.m]),
                                                                        self.simulation_start_epoch,
                                                                        integrator_settings,
                                                                        termination_settings)


            propagator_settings_list = [self.propagator_settings_translational,
                                    self.mass_propagator_settings]

            mass_variable = propagation_setup.dependent_variable.body_mass("Vehicle")
            dependent_variables_to_save = [mass_variable]

        else:
            propagator_settings_list = [self.propagator_settings_translational]
            dependent_variables_to_save = []

        # Create combination of translational and mass rate model
        self.propagator_settings = propagation_setup.propagator.multitype(propagator_settings_list,
                                                                        integrator_settings,
                                                                        self.simulation_start_epoch,
                                                                        termination_settings,
                                                                        dependent_variables_to_save)



    def integrate(self, interpolator_source = "scipy"):

        if self.verbose:
            print("[DOLPHINN] Start Integrating")

        start = time.time()
        # Propagate the dynamics
        self.dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                            self.bodies, self.propagator_settings)
        end = time.time()

        if self.verbose:
            print(f"[DOLPHINN] Finished integrating in {np.round(end-start, 5)} s")

        # Unpack the tudat propagation
        state_history = self.dynamics_simulator.state_history
        times = self.ref_times.reshape(-1, 1)

        # Prepare correct interpolator
        if interpolator_source == "tudat":
            interpolator_object = interpolators.create_one_dimensional_vector_interpolator(state_history,  interpolators.lagrange_interpolation(8))
            interpolator = lambda epoch: interpolator_object.interpolate(epoch)

        elif interpolator_source == "scipy":
            state_history_arr = result2array(state_history)
            interpolator_object = CubicSpline(state_history_arr[:,0],
                                    state_history_arr[:,1:])
            interpolator = lambda epoch: interpolator_object(epoch)[0]
        else:
            raise ValueError("Unknown interpolator source, options are: tudat, scipy")

        states = np.zeros((len(times), 7))

        if self.mass_rate:
            self.mass = np.zeros((len(times), 2))

        for i, epoch in enumerate(times):
            temp_state = interpolator(epoch)
            states[i, 0] = epoch

            if self.mass_rate:
                states[i, 1:7]  = temp_state[:6]
                self.mass[i, 0] = epoch
                self.mass[i, 1] = temp_state[6]
            else:
                states[i, 1:7] = temp_state

        # Remove the z axis
        states = np.delete(states, 6, axis=1)
        states = np.delete(states, 3, axis=1)

        # potentially add the control to the states
        if self.control_nodes:
            control = np.array(list(self.control_nodes.values()))
            states = np.concatenate((states, control), axis = 1)

        # Create the states dictionary
        self.states = {"cartesian": states}


    def calculate_coordinates(self, coordinates, config):

        transformation = getattr(coordinate_transformations, f"cartesian_to_{coordinates}")
        self.states[coordinates] = transformation(self.states['cartesian'], config)

    @classmethod
    def from_config_states(cls,
                           states,
                           config,
                           initial_mass,
                           verbose = True,
                           mass_rate = False,
                           integrator = None,
                           tfinal = None):

        # Get the dynamics info
        central_body, _, entries,\
        control_entries, coordinates = get_dynamics_info_from_config(config)

        # Transform to cartesian
        if "cartesian" not in list(states.keys()):
            transformation = getattr(coordinate_transformations, f"{coordinates}_to_cartesian")
            cartesian_states = transformation(states[coordinates], config)
        else:
            cartesian_states = states["cartesian"]

        # Define star and end
        t0       = cartesian_states[0,0]

        if not tfinal:
            tfinal   = cartesian_states[-1,0]

        # Prepare control and initial state
        if control_entries:
            control = cartesian_states[:,-control_entries:]
            control_nodes = {key: value for key, value in zip(cartesian_states[:,0], control)}
            isp = config['isp']
            initial_state = cartesian_states[0,1:-control_entries].reshape(-1, 1)

        else:
            control_nodes = None
            isp = None
            initial_state = cartesian_states[0,1:].reshape(-1, 1)

        # Make 3D initial state
        if initial_state.shape[0] == 4:
            initial_state = np.concatenate((initial_state[0:2,:],
                                            np.array([[0]]),
                                            initial_state[2:4,:],
                                            np.array([[0]])),
                                            axis = 0).reshape(-1, 1)

        return cls(initial_mass,
                    t0,
                    tfinal,
                    initial_state,
                    isp = isp,
                    central_body = central_body,
                    control_nodes = control_nodes,
                    ref_times = cartesian_states[:,0],
                    verbose = verbose,
                    mass_rate = mass_rate,
                    integrator = integrator)

    @classmethod
    def from_DOLPHINN(cls,
                      DOLPHINN,
                      integrator = None,
                      tfinal = None):
        '''
        Initialize from a DOLPHINN class instance
        '''

        if 'm' in DOLPHINN.config:
            initial_mass = DOLPHINN.config['m']
        else:
            initial_mass = DOLPHINN.mass[0,1]

        return cls.from_config_states(DOLPHINN.states,
                                      DOLPHINN.config,
                                      initial_mass,
                                      verbose = DOLPHINN.base_verbose,
                                      mass_rate = DOLPHINN.dynamics.mass,
                                      integrator = integrator,
                                      tfinal = tfinal)







