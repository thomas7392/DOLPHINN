# Thomas Goldman 2023
# DOLPHINN

import os
import numpy as np
import json

from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp

from . import dynamics

def get_dynamics_info_from_config(config):
    '''
    Gather information about the dynamics
    descibred in a config dictionary

    Args:
        config (dict):  The configuration dictionary, must include the
                        gravitional paramter "mu" and the name of the
                        dynamics class "dynamics"
    '''

    if config['mu'] > 1.32e20 and config['mu'] < 1.33e20:
        central_body =  "Sun"

    if config['mu'] < 4e14 and config['mu'] > 3.9e14:
        central_body = "Earth"

    class_object = getattr(dynamics, config['dynamics'])
    control = class_object.control
    entries = class_object.entries
    control_entries = class_object.control_entries
    coordinates = class_object.coordinates

    return central_body, control, entries, control_entries, coordinates


def integrate_theta(time,
                    best_y,
                    theta_0 = 0):
    '''
    Integrate to retrieve theta
    '''


    # Interpolate r, vr and vt
    cs = CubicSpline(time, best_y[:,:3])

    # ODE for theta
    def fun(time, state):

        r, _, vt = cs(time)
        dtheta_dt = vt/r

        return np.array([dtheta_dt])

    # Integrate theta
    result = solve_ivp(fun,
                        (time[0], time[-1]),
                        np.array([theta_0]),
                        atol = 1e-10,
                        rtol = 1e-10)

    # Extract results, interpolate and get theta at original times
    time2 = result.t
    y = result.y.T
    cs2 = CubicSpline(time2, y)
    thetas = cs2(time)

    return thetas


def path_or_instance(x,
                     inst_type):

    if isinstance(x, str):
        if os.path.exists(x):

            if inst_type == np.ndarray:
                product = np.loadtxt(x)
            elif inst_type == dict:
                with open(x, 'r') as file:
                    product = json.load(file)
            else:
                raise ValueError(f"path_or_instance can only check for inst_type dict or np.ndarray")

        else:
            raise ValueError(f"{x}: invalid path")

        return product

    elif not isinstance(x, inst_type):
        return TypeError(f"{x} is of invalid type ({type(x)}): choose a path (str) to a config.JSON file\
                            or directly provide a dictionary or array.")

    return x


def print_config(config):

    config = path_or_instance(config, dict)

    function_keys = ['dynamics', 'input_transform', 'output_transform', 'objective']
    training_keys = [key for key in list(config.keys()) if key[:5] == "train"]
    network_keys = ["architecture", "activation", "sampler", "N_train", "N_boundary", "N_test", "seed"]
    metric_keys = [key for key in list(config.keys()) if key[:6] == "metric"]
    problem_keys = [key for key in list(config.keys()) if key not in function_keys+training_keys+network_keys+metric_keys]

    keys = {"Functions": function_keys,
            "Problem": problem_keys,
            "Network": network_keys,
            "Metrics": metric_keys,
            "Training": training_keys,
            }

    print(f"\n {'='*20} Config Content {'='*20}\n")

    for cat_name, keys in keys.items():

        print(f"{'-'*10}{cat_name}{'-'*10}")
        for key in keys:

            try:
                value = config[key]

                if isinstance(value, dict):
                    print(f"{key.ljust(30)} {value['name']}")
                    for key, value2 in value.items():
                        if key != 'name':
                         print(f"{' ' * 4}{key.ljust(26)} {value2}")
                    print()
                else:
                    print(f"{key.ljust(30)} {value}")
            except:
                pass

        print()
    print(f"{'='*55}\n")

