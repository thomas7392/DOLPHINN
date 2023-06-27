# Thomas Goldman 2023
# DOLPHINN

# General imports
import os
import numpy as np
import datetime
import json

# DeepXDE Imports
import deepxde as dde
from deepxde.backend import tf
from deepxde import config, optimizers

# DOLPHINN imports
from . import dynamics
from . import output_layers
from . import objectives
from . import input_layers
from . import training

from .function import Function
from .ObjectivePINN import ObjectivePINN

class DOLPHINN:
    '''
    Base class to optimize or integrate a dynamical problem using PINN
    '''

    def __init__(self,
                 data,
                 dynamics_name,
                 input_transform = None,
                 output_transform = None,
                 objective = None,
                 verbose = True,
                 train = None,
                 metrics = [],
                 callbacks = [],
                 seed = None):
        '''
        Initalizes the class

        Args:
            data (dict):    Containing al variables necesarry to define the network and
                            all parameters used in the dynamics, intput_transform, output_transform
                            and objective. Mandatory: t0, tfinal, activation, architecture, N_train,
                            N_test, N_boundary, sampler. Depending on the situation, for example:
                            initial_state, final_state, mu, m, a, umax, isp, length_scale, time_scale, etc.

            dynamics_name (Function, str):   A string with the name of an existing Equation of Motion in
                                          DOLPHINN.dynamics or a Class

            input_transform (Function, str): A layer preceding the network. A string with the name of an
                                          existing layer implementation in DOLPHINN.input_transform or a
                                          Class.

            output_transform (Function, str): A layer after the network. A string with the name of an
                                          existing layer implementation in DOLPHINN.output_transform or a
                                          Class.

            objective (Function, str):    A function describing an objective.
            verbose (bool):  Printing information toggle
            train (list):    Immidiatelly train upon creation of instance. Requires config-like dictionary
                             input

        '''

        fnames = ["dynamics", "input_transform", "output_transform", "objective"]
        functions = [dynamics_name, input_transform, output_transform, objective]

        # Create attributes of the functions that require data to work
        for fname, function in zip(fnames, functions):

            # Check if not None
            if function:

                if isinstance(function, str):
                    class_object = getattr(dynamics, dynamics_name)
                    instance = class_object(data)
                elif issubclass(function, Function):
                    instance = function(data)
                else:
                    raise TypeError(f"{fname} is of wrong type, should be a Function class or a string")

                setattr(self, fname, instance)
            else:
                setattr(self, fname, None)

        self.data = data
        self.train_procedure = 0
        self.base_verbose = verbose
        self.metrics = metrics
        self.callbacks = callbacks

        self._create_model(seed = seed, verbose = self.base_verbose)
        self._create_config()

        # Train network upon initialisation
        if train:
            for procedure in train:

                algorithm = getattr(training, procedure['name'])
                algorithm_parameters = procedure.copy()
                del algorithm_parameters['name']

                algorithm_instance = algorithm(**algorithm_parameters)
                self.train(algorithm_instance)


    @classmethod
    def from_config(cls,
                    config,
                    upload_seed = True,
                    train = False,
                    verbose = True):
        '''
        Initalizes the class from a configuration file
        '''

        # If string, treat as path and
        if isinstance(config, str):
            # Load the JSON file back into a dictionary

            if os.path.exists(config):
                with open(config, 'r') as file:
                    config_dict = json.load(file)
            else:
                raise ValueError(f"{config}: invalid path")

            config = config_dict

        # Check if dictionary
        elif not isinstance(config, dict):
            return ValueError(f"{config} invalid config argument: choose a path to a config.JSON file\
                              or directly provide the dictionary.")

        # Retrieve function and training keys
        function_keys = ['dynamics', 'output_transform', 'input_transform', 'objective']
        training_keys = [key for key in list(config.keys()) if key[:5] == "train"]

        # Make valid None values of NoneType
        for function in function_keys:
            if config[function] == "NoneType":
                config[function] = None

        # Check if
        if config["dynamics"] and not hasattr(dynamics, config["dynamics"]):
            raise ValueError(f"Dynamnics function {config[function]} is not\
                             implemented in DOLPHINN.dynamics")

        if config["input_transform"] and not hasattr(input_layers, config["input_transform"]):
            raise ValueError(f"Input_transform function {config[function]} is not\
                             implemented in DOLPHINN.input_layer")

        if config["output_transform"] and not hasattr(output_layers, config["output_transform"]):
            raise ValueError(f"Output_transform function {config[function]} is not\
                             implemented in DOLPHINN.output_layers")

        if config["objective"] and not hasattr(objectives, config["objective"]):
            raise ValueError(f"Output_transform function {config[function]} is not\
                             implemented in DOLPHINN.objectives")

        # Create data dictionary
        data = {key: value for key, value in config.items() if key not in function_keys+training_keys}
        for key in ['initial_state', 'final_state']:
            data[key] = np.array(data[key])

        # Decide if to train with procedure found in config file
        training = None
        if train:
            training = [config[key] for key in training_keys]
            if len(training) == 0:
                print("A training was requested, but config contains no training procedures")

        # Decide if to give it the seed from the config solution.
        if upload_seed:
            if "seed" not in list(config.keys()):
                raise ValueError("The config file contains no seed to upload")
            seed = config['seed']
        else:
            seed = None

        if verbose:
            print(f"Config file succesfully parsed. Initializing DOLPHINN with:")
            print()
            for key, value in config.items():
                print(f"{key}:    {value}")

        return cls(data,
                   config['dynamics'],
                   output_transform=config['output_transform'],
                   input_transform = config['input_transform'],
                   objective = config['objective'],
                   train = training,
                   seed = seed)

    def _create_model(self, seed=None, verbose = True):
        '''
        Creates the model using DeepXDE funcionality
        '''

        # Prepare nn initilisation seed
        if not seed:
            current_time = datetime.datetime.now()
            seed = int(current_time.strftime("%Y%m%d%H%M%S"))

            if verbose:
                print(f"Using time-dependent random seed: {seed}")

        # Store the current seed in the data dictionary, such that it will be
        # known what seed was used for the final solution
        self.data['seed'] = seed

        # Build the network from the config dictionary
        geom = dde.geometry.TimeDomain(self.data['t0'], self.data['tfinal'])

        data = dde.data.PDE(geom,
                            self.dynamics.call,
                            [],
                            self.data['N_train'],
                            self.data['N_boundary'],
                            num_test = self.data['N_test'],
                            train_distribution = self.data['sampler'])

        initializer = tf.keras.initializers.GlorotNormal(seed=seed)

        net = dde.nn.PFNN(self.data['architecture'],
                          self.data['activation'],
                          initializer)

        if self.input_transform:
            net.apply_feature_transform(self.input_transform.call)

        if self.output_transform:
            net.apply_output_transform(self.output_transform.call)

        if self.objective:
            net.regularizer = self.objective.call

        self.model = ObjectivePINN(data, net)
        self.model.display_progress = verbose

    def train(self, algorithm):
        '''
        Perform training
        '''

        if not isinstance(algorithm, list):
            algorithm = [algorithm]

        # Check if algortihms are valid
        for alg in algorithm:
            if not issubclass(type(alg), Function):
                raise TypeError(f"{type(alg).__name__} is not a valid training algorithm,\
                                the class should inherrit the DOLPHINN.function.Function class")

        # Iterate over training algorithms
        for alg in algorithm:
            self.train_procedure += 1
            alg.call(self)
            self._update_config(alg)

    def store(self, path, overwrite = False):
        '''
        Store the current configuration with losses and train/test results
        '''

        # Check if folder exists
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if not overwrite:
                print("Folder exists")
                return False

        # Save loss and train/test data
        dde.utils.saveplot(self.model.losshistory,
                           self.model.train_state,
                           isplot=False, output_dir = path)

        # Save weightd and biases
        self.model.save(path)

        # Save configuration
        with open(path + "config", 'w') as file:
            json.dump(self.config, file)

        print(f"Saving config file to {path}config")

    def restore(self, path):
        '''
        Update the weigths and biases of the network
        '''
        self.model.restore(path)

        if self.base_verbose:
            print(f"Restored weights at {path}")


    def _update_config(self, algorithm):
        '''
        Update the configuration file with a training
        '''

        training_overview = {f"train_{self.train_procedure}": Function.get_attributes(algorithm)}
        self.config.update(training_overview)

    def _create_config(self):
        '''
        Create configuration file
        '''

        # Functions
        self.config = {"dynamics": self.dynamics.__class__.__name__,
                        "objective": self.objective.__class__.__name__,
                        "output_transform": self.output_transform.__class__.__name__,
                        "input_transform": self.input_transform.__class__.__name__,
                        }

        # Include data
        self.config.update(self.data)

        # Transform np.arrays to lists for JSON
        for key in ['initial_state', 'final_state']:
            self.config[key] = list(self.config[key])




