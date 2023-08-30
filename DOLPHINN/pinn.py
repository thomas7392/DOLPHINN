# Thomas Goldman 2023
# DOLPHINN

# General imports
import os
import numpy as np
import datetime
import json
import time

# DeepXDE Imports
import deepxde as dde
from deepxde.backend import tf

# DOLPHINN imports
from . import dynamics
from . import output_layers
from . import objectives
from . import input_layers
from . import training
from . import utils
from . import coordinate_transformations
from . import verification
from . import metrics

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
                 train = None,
                 metrics = [],
                 callbacks = [],
                 seed = None,
                 solution = None,
                 display_every = 1000,
                 verbose = True):
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

        libraries = [dynamics, input_layers, output_layers, objectives]
        fnames = ["dynamics", "input_transform", "output_transform", "objective"]
        functions = [dynamics_name, input_transform, output_transform, objective]

        # Create attributes of the functions that require data to work
        for fname, function, library in zip(fnames, functions, libraries):

            # Check if not None
            if function:

                # Check type
                if isinstance(function, str):
                    class_object = getattr(library, function)
                elif issubclass(function, Function):
                    class_object = function
                else:
                    raise TypeError(f"[DOLPHINN] {fname} is of wrong type, should be a Function class or a string")

                # Create instance of class object
                if fname == "objective":
                    instance = class_object(data, self.dynamics.mass)
                else:
                    instance = class_object(data)

                # Set instance of function object as attribute of DOLPHINN
                setattr(self, fname, instance)
            else:
                setattr(self, fname, None)

        self.data = data
        self.train_procedure = 0
        self.base_verbose = verbose
        self.metrics = [m(self).try_call for m in metrics]
        self.callbacks = callbacks
        for c in callbacks:
            c.add_dolphinn(self)
        self.old_solution = True if solution else False
        self.display_every = display_every
        self.full_train_time = 0
        self.mass = None


        self._create_model(seed = seed, verbose = self.base_verbose)
        self._create_config()

        # Upload solution to the DOLPHINN class
        if solution:
            self._upload_solution(solution)
            self.old_solution = True

        # Train network upon initialisation
        elif train:

            train_list = []

            # Create training instances
            for procedure in train:
                algorithm = getattr(training, procedure['name'])
                algorithm_parameters = procedure.copy()
                del algorithm_parameters['name']
                algorithm_instance = algorithm(**algorithm_parameters)
                train_list.append(algorithm_instance)

            if self.base_verbose:
                print("[DOLPHINN] Starting training procedure encountered in config file")

            # Train the algorithms
            self.train(train_list)

    @classmethod
    def from_solution(cls,
                      path,
                      verbose = True):
        '''
        Upload a folder containing the solution of a previously trained DOLPHINN
        via the following files:
            path/config
            path/test.dat
            path/loss.dat
            path/<...>.ckpt.data-00000-of-00001
        '''

        config_path = path + "config"
        test_path = path + "test.dat"
        loss_path = path + "loss.dat"

        # Search weights path
        files = os.listdir(path)
        for file in files:
            if file[-5:] == "index":
                weigths_path = path + file.split(".")[0] + ".ckpt"

        if verbose:
            print(f"[DOLPHINN] Initializing the DOLPHINN from old solution at: {path}")

        return cls.from_config(config_path,
                        solution = [test_path, loss_path, weigths_path],
                        upload_seed=True,
                        train = False,
                        verbose = verbose)

    @classmethod
    def from_config(cls,
                    config,
                    solution = None,
                    upload_seed = True,
                    train = False,
                    verbose = True):
        '''
        Initalizes the class from a configuration file
        '''

        config = utils.path_or_instance(config, dict)

        # Prepare to upload a solution
        if solution:
            best_y = utils.path_or_instance(solution[0], np.ndarray)
            losshistory = utils.path_or_instance(solution[1], np.ndarray)
            weigths_path = solution[2]
            solution = [best_y, losshistory, weigths_path]
            if 'train_time' in list(config.keys()):
                solution.append(config['train_time'])


        # Retrieve function and training keys
        function_keys = ['dynamics', 'output_transform', 'input_transform', 'objective']
        training_keys = [key for key in list(config.keys()) if key[:5] == "train"]
        metric_keys = [key for key in list(config.keys()) if key[:6] == "metric"]
        metrics_ = [getattr(metrics, config[metric]) for metric in metric_keys]

        # Make valid None values of NoneType
        for function in function_keys:
            if config[function] == "NoneType":
                config[function] = None

        # Check if functions exist
        if config["dynamics"] and not hasattr(dynamics, config["dynamics"]):
            raise ValueError(f"[DOLPHINN] Dynamnics function {config[function]} is not\
                             implemented in DOLPHINN.dynamics")

        if config["input_transform"] and not hasattr(input_layers, config["input_transform"]):
            raise ValueError(f"[DOLPHINN] Input_transform function {config[function]} is not\
                             implemented in DOLPHINN.input_layer")

        if config["output_transform"] and not hasattr(output_layers, config["output_transform"]):
            raise ValueError(f"[DOLPHINN] Output_transform function {config[function]} is not\
                             implemented in DOLPHINN.output_layers")

        if config["objective"] and not hasattr(objectives, config["objective"]):
            raise ValueError(f"[DOLPHINN] Output_transform function {config[function]} is not\
                             implemented in DOLPHINN.objectives")

        # Create data dictionary
        data = {key: value for key, value in config.items() if key not in function_keys+training_keys+metric_keys}
        for key in ['initial_state', 'final_state']:
            if key in list(config.keys()):
                data[key] = np.array(data[key])

        # Add old solutions training procedure to data file in case of solutinon
        if solution:
            for key in training_keys:
                data[key] = config[key]

        # Decide if to train with procedure found in config file
        training = None
        if train:
            training = [config[key] for key in training_keys]
            if len(training) == 0:
                print("[DOLPHINN] A training was requested, but config contains no training procedures")

        # Decide if to give it the seed from the config solution.
        if upload_seed:
            if "seed" not in list(config.keys()):
                raise ValueError("The config file contains no seed to upload")
            elif upload_seed and solution:
                if verbose:
                    print("[DOLPHINN][Warning] Upload of seed requested: initialisation will be overwritten by the solution")
            seed = config['seed']
        else:
            seed = None

        if train and solution:
            raise ValueError("[DOLPHINN] Train is requested and solution is provided: choose one")

        if verbose:
            print(f"[DOLPHINN] Config file succesfully parsed. Initializing DOLPHINN with:")
            utils.print_config(config)

        return cls(data,
                   config['dynamics'],
                   output_transform = config['output_transform'],
                   input_transform = config['input_transform'],
                   objective = config['objective'],
                   train = training,
                   seed = seed,
                   solution = solution,
                   verbose = verbose,
                   metrics = metrics_)

    def _create_model(self, seed=None, verbose = True):
        '''
        Creates the model using DeepXDE funcionality
        '''

        # Prepare nn initilisation seed
        if not seed:
            current_time = datetime.datetime.now()
            seed = int(current_time.strftime("%Y%m%d%H%M%S"))

            if verbose:
                print(f"[DOLPHINN] Using time-dependent random seed: {seed}")
        else:
            if verbose:
                print(f"[DOLPHINN] Using user-defined seed: {seed}")

        # Store the current seed in the data dictionary, such that it will be
        # known what seed was used for the final solution
        self.data['seed'] = seed
        if hasattr(self, "config"):
            self.config['seed'] = seed

        # Build the network
        geom = dde.geometry.TimeDomain(self.data['t0'], self.data['tfinal'])

        data = dde.data.PDE(geom,
                            self.dynamics.call,
                            [],
                            self.data['N_train'],
                            self.data['N_boundary'],
                            num_test = self.data['N_test'],
                            train_distribution = self.data['sampler'])

        # Overide the get test data function that includes boundary points
        test_data = np.linspace(self.data['t0'],
                                self.data['tfinal'],
                                self.data['N_test'],
                                dtype=dde.config.real(np)).reshape(-1, 1)
        def new_test(self):
            return test_data, None, None
        data.test = new_test.__get__(data, dde.data.PDE)

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

        if self.old_solution:
            if self.base_verbose:
                print("[DOLPHINN] Compiling the DeepXDE model to be able to use DOLPHINN.model.predict")

            self.model.compile("adam", lr = 1e-8)

    def train(self,
              algorithm,
              additional_callbacks = []):
        '''
        Train the Dolphin!
        '''

        assert not self.old_solution, "This is an old uploaded solution, additional \
                                   training is not possible. Recreate this solution \
                                   with DOLPHINN.from_config(config, train = True, upload_seed = True) \
                                   to get (nearly) the same solution. Only difference is the random aspect \
                                   in the training data sampling. If you want to enforce additional training\
                                   set DOLPHINN.old_solution = True."

        if not isinstance(algorithm, list):
            algorithm = [algorithm]

        # Check if algortihms are valid
        for alg in algorithm:
            if not issubclass(type(alg), Function):
                raise TypeError(f"{type(alg).__name__} is not a valid training algorithm,\
                                the training class should inherrit the\
                                DOLPHINN.function.Function class")

        start_train = time.time()

        #Iterate over training algorithms
        for alg in algorithm:
            if self.base_verbose:
                print(f"\n[DOLPHINN] Training with procedure: {alg.name}\n")
            alg.call(self, additional_callbacks) #Train the DOLPHINN

        end_train = time.time()
        training_time = end_train - start_train
        self.full_train_time += training_time

        self._update_config(algorithm)

        if self.base_verbose:
            print(f'[DOLPHINN] This training {[alg.name for alg in algorithm]} took {np.round(training_time, 2)} s')
            print(f"[DOLPHINN] The entire training so far took: {np.round(self.full_train_time, 2)} s")

        # Create best states and loss in DOLPHINN class instance
        self._create_states_and_loss(self.model.train_state.X_test,
                                    self.model.train_state.best_y,
                                    np.array(self.model.losshistory.loss_train),
                                    np.array(self.model.losshistory.loss_test),
                                    np.array(self.model.losshistory.steps),
                                    np.array(self.model.losshistory.metrics_test))

    def _create_states_and_loss(self,
                                time,
                                best_y,
                                loss_train,
                                loss_test,
                                steps,
                                metrics):
        '''
        Stores the loss history and best test solution in the
        DOLPHINN instance. Converts states to Non-Dimensional cartesian.
        '''

        # Relevant results
        self.loss_train = loss_train
        self.loss_test = loss_test
        self.steps = steps
        self.metrics_test = metrics

        # Store mass seperately and then strip mass from the states
        # Assume mass is always the final entry
        if self.dynamics.mass:
            self.mass = np.concatenate((time, best_y[:,-1].reshape(-1, 1)), axis = 1)
            best_y = best_y[:,:-1]

        # Retrieve theta if radial coordinates, then add time and initial state
        if self.dynamics.coordinates == "radial" and not self.dynamics.theta:
            theta = utils.integrate_theta(time[:,0],
                                          best_y)
            states = np.concatenate((time, best_y[:,0:1], theta.reshape(-1, 1), best_y[:,1:]), axis = 1)
        else:
            states = np.concatenate((time, best_y), axis = 1)

        # If on/off structure, strip the onn off entry
        if self.dynamics.on_off:
            states = states[:,:-1]

        # Create state attribute
        self.states = {self.dynamics.coordinates: states}

        # Potentially convert states to Non-Dimensional Cartesian states
        if self.dynamics.coordinates != "NDcartesian":
            transformation = getattr(coordinate_transformations,
                                      f"{self.dynamics.coordinates}_to_NDcartesian")
            self.states["NDcartesian"] = transformation(self.states[self.dynamics.coordinates],
                                                        self.config)

    def calculate_coordinates(self, coordinates):

        existing_coordiantes = self.dynamics.coordinates
        transformation = getattr(coordinate_transformations, f"{existing_coordiantes}_to_{coordinates}")
        self.states[coordinates] = transformation(self.states[existing_coordiantes], self.config)


    def _upload_solution(self, solution):


        best_y_arr = solution[0]
        losshistory_arr = solution[1]
        weigths_path = solution[2]

        self.restore(weigths_path)

        if len(solution) > 3:
            self.full_train_time = solution[-1]

        time = best_y_arr[:,0:1]
        best_y = best_y_arr[:,1:]

        loss_entries = self.dynamics.loss_entries + int(bool(self.objective))
        loss_train = losshistory_arr[:,1:1+loss_entries]
        loss_test = losshistory_arr[:,1+loss_entries:1+2*loss_entries]
        metrics_test = losshistory_arr[:,1+2*loss_entries:]
        steps = losshistory_arr[:,0]

        self._create_states_and_loss(time,
                                    best_y,
                                    loss_train,
                                    loss_test,
                                    steps,
                                    metrics_test)


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

        print(f"[DOLPHINN] Saving config file to {path}config")

    def restore(self, path):
        '''
        Update the weigths and biases of the network
        '''
        self.model.restore(path)

        if self.base_verbose:
            print(f"[DOLPHINN] Restored weights at {path}")


    def _update_config(self, algorithms):
        '''
        Update the configuration file with a training
        '''

        for i, algorithm in enumerate(algorithms):
            training_overview = {f"train_{self.train_procedure + i}": Function.get_attributes(algorithm)}
            self.config.update(training_overview)

        self.train_procedure += len(algorithms)

        training_time = {"train_time": self.full_train_time}
        self.config.update(training_time)

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

        self.config.update({f"metric_{i+1}": metric.__self__.__class__.__name__ for i, metric in enumerate(self.metrics)})

        # Include data
        self.config.update(self.data)

        # Transform np.arrays to lists for JSON
        for key in ['initial_state', 'final_state']:
            if key in list(self.config.keys()):
                self.config[key] = list(self.config[key])


    def verify(self):
        '''
        Perform a verification of a trained network by integrating the dynamics,
        initial state and control profile for the same time span using a
        traditional numerical integrator
        '''

        self.bench = verification.Verification.from_DOLPHINN(self)
        self.bench.integrate()
        self.bench.calculate_coordinates("NDcartesian", self.config)

    def print_config(self):
        utils.print_config(self.config)


    def print_metrics(self, additional_metrics = []):

        metrics_to_print = self.metrics

        if len(additional_metrics) != 0 and self.old_solution:
            raise Exception("[DOLPHINN] Can't include new metrics in old solution.")

        for m in additional_metrics:

            if isinstance(m, str):
                m = getattr(metrics, m)
            elif issubclass(m, metrics.Metric):
                pass
            else:
                raise TypeError(f"[DOLPHINN] Metric {m} is of wrong type, should be str of DOLPHINN.metrics.Metric")

            metrics_to_print.append(m(self))

        if len(metrics_to_print):
            print("---- Metrics ----")
        else:
            print("[DOLPHINN] No metrics to print")

        for m in metrics_to_print:
            name = m.__class__.__name__
            print(f"{name.ljust(30)} {m.call(self.model.train_state)}")






