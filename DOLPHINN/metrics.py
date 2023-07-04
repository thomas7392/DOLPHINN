import numpy as np

from .function import Function
from .verification import Verification
from . import utils
from . import coordinate_transformations
from abc import ABC, abstractmethod


class Metric(ABC):
    '''
    Base class for a metric. Contains functionality to perform a verification run
    that is then stored in the DOLPHINN class.

    The idea is that metrics inherrit this class,
    and perform the verification run in their call method.

    Via "run_verification", this class will search if there is already a verification run and
    if not, perform a verifaction run.

    It sounds strange, but i believe I had a good reason for this structure.
    '''

    def __init__(self, DOLPHINN):

        self.DOLPHINN = DOLPHINN
        self.coordinates = self.DOLPHINN.dynamics.coordinates
        self.DOLPHINN._metric_verification_epoch = -1

    @abstractmethod
    def call(self):
        '''
        Enforce the existance of the call method
        '''
        pass

    def try_call(self, train_state):
        '''
        Entry point for calling the metric from inside the PINN class
        Abstracts the try/except block such that the user merely has to
        design the call method.
        '''

        try:
            return self.call(train_state)
        except Exception as e:
            print(f"[DOLPHINN] Error occured in metric: {e}")
            return np.nan


    def run_verification(self, train_state):

        if self.DOLPHINN._metric_verification_epoch != train_state.epoch:

            # Extract current test
            times = train_state.X_test
            _states = train_state.y_pred_test

            # Strip mass if it is propagated
            if self.DOLPHINN.dynamics.mass:
                self.DOLPHINN._masses_for_metric = _states[:,-1]
                _states = _states[:,:-1]

            if self.coordinates == "radial":
                theta = utils.integrate_theta(times[:,0],
                                                _states)
                _states = np.concatenate((_states[:,0:1], theta.reshape(-1, 1), _states[:,1:]), axis = 1)
            _states = np.concatenate((times, _states), axis=1)
            states = {self.coordinates: _states}

            # Verifiy
            verification = Verification.from_config_states(states,
                                                           self.DOLPHINN.config,
                                                           verbose = False,
                                                           mass_rate = self.DOLPHINN.dynamics.mass)
            verification.integrate()
            verification.calculate_coordinates("NDcartesian", self.DOLPHINN.config)

            if self.coordinates != "NDcartesian":
                transformation = getattr(coordinate_transformations, f"{self.coordinates}_to_NDcartesian")
                states['NDcartesian'] = transformation(states[self.coordinates], self.DOLPHINN.config)

            # Extract mass from propagation
            if self.DOLPHINN.dynamics.mass:
                self.DOLPHINN._verification_masses_for_metric = verification.mass[:,-1]

            # Store the verification states and the original states in the DOLPIHNN class
            self.DOLPHINN._states_for_metric = states['NDcartesian']
            self.DOLPHINN._verification_states_for_metric = verification.states['NDcartesian']
            self.DOLPHINN._metric_verification_epoch = train_state.epoch

        return self.DOLPHINN._states_for_metric, self.DOLPHINN._verification_states_for_metric,\
                self.DOLPHINN._masses_for_metric, self.DOLPHINN._verification_masses_for_metric


class FinalDr(Metric):
    '''
    Final position difference metric
    '''
    def __init__(self, DOLPHINN):
        super().__init__(DOLPHINN)

    def call(self, train_state):

        # Get the verification information
        states, verification_states, _, _ = self.run_verification(train_state)
        return np.linalg.norm(states[-1,1:3] - verification_states[-1,1:3])


class FinalDv(Metric):
    '''
    Final velocity difference metric
    '''
    def __init__(self, DOLPHINN):
        super().__init__(DOLPHINN)

    def call(self, train_state):

        # Get the verification information
        states, verification_states, _, _ = self.run_verification(train_state)
        return np.linalg.norm(states[-1,1:3] - verification_states[-1,1:3])

class FinalDm(Metric):
    '''
    Final velocity difference metric
    '''
    def __init__(self, DOLPHINN):
        super().__init__(DOLPHINN)

    def call(self, train_state):

        # Get the verification information
        _, _, masses, verification_masses = self.run_verification(train_state)
        return masses[-1] - verification_masses[-1]







