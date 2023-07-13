# Thomas Goldman 2023
# DOLPHINN

import numpy as np
import matplotlib.pyplot as plt


def plot_loss(case,
              oweigth = None,
              objective_zoom = False):

    steps = case.steps
    loss_train = case.loss_train
    loss_test = case.loss_test

    if oweigth == None:
        try:
            counter = -1
            for key in list(case.config.keys()):
                if key.split("_")[0] == 'train' and key.split("_")[1] != "time":
                    counter += 1
            oweigth = case.config[f'train_{counter}']['loss_weigths'][-1]
        except:
            oweigth = None

    loss_labels = case.dynamics.loss_labels

    if case.objective:
        loss_labels += [f"Objective (fuel)\nWeight = {oweigth}"]

    fig, axes = plt.subplots(1, 2 + int(objective_zoom), figsize = (21, 6))

    axes[0].plot(steps, np.sum(loss_train, axis = 1), label = "Train loss")
    axes[0].plot(steps, np.sum(loss_test, axis = 1), "r--", dashes = (4, 4),  label = "Test loss")
    axes[0].set_title("Total Loss", fontsize = 20)
    axes[1].set_title("Individual losses", fontsize = 20)

    n_loss_entries = case.dynamics.loss_entries
    if case.objective:
        n_loss_entries += 1

    for i in range(n_loss_entries):
        axes[1].plot(steps, np.array(loss_train)[:,i], label = loss_labels[i])

    if objective_zoom:
        axes[2].set_title("Objective Loss Zoom-In", fontsize = 20)
        axes[2].plot(steps[steps > 10000], np.array(loss_train)[steps > 10000,-1], label = "Train")
        axes[2].plot(steps[steps > 10000], np.array(loss_test)[steps > 10000,-1],
                    label = f"Test")
        #axes[2].set_yscale("log")

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")

    for ax in axes:
        ax.legend()
        ax.grid()
        ax.set_ylabel("Loss", fontsize = 20)
        ax.set_xlabel("Iterations", fontsize = 20)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2)
        ax.tick_params(labelsize=13)
        ax.tick_params(axis="both",direction="in",which="both", length=4, width = 1.2)
        ax.tick_params(bottom=True, top=True, left=True, right=True)




def plot_transfer(case,
                  thrust = True,
                  velocity = False,
                  bench = False,
                  thrust_scale = 0.1,
                  velocity_scale = 1,
                  N_arrows = 50,
                  r_target = 1.5,
                  r_start = 1,
                  lim = None,
                  grid = False):

    # Retrieve relevant data from the DOLPHINN class instance

    if bench:
        y = case.bench.states['NDcartesian'][:,1:]
    else:
        y = case.states['NDcartesian'][:,1:]


    theta = np.linspace(0, 2*np.pi, 1000)

    if lim==None:
        if r_start==None and r_target == None:
            lim = 1.5*np.max(np.linalg.norm(y[:,:2], axis = 1))
        else:
            lim = 1.5*max(r_target, r_start, np.max(np.linalg.norm(y[:,:2], axis = 1)))

    fig, ax = plt.subplots(1, figsize = (10, 10))

    if bench:
        title = "Physically-Informed Neural Network Solution\n---Verification Numerical Intergation---"
    else:
        title = "Physically-Informed Neural Network Solution\n---DOLPHINN---"

    fig.suptitle(title, fontsize = 25, y=0.97)
    plt.plot(y[:,0], y[:,1], label = "transfer trajectory")

    if r_target:
        plt.plot(r_target*np.cos(theta), r_target*np.sin(theta), label = "Target orbit")
    if r_start:
        plt.plot(r_start*np.cos(theta), r_start*np.sin(theta), label = "Start orbit")

    if thrust:
        arrow_indices = np.linspace(0, len(y[:,0])-1, N_arrows, dtype = int)
        scale = thrust_scale

        for i in arrow_indices:
            plt.arrow(y[i,0],
                      y[i,1],
                      y[i,4]*0.5/scale,
                      y[i,5]*0.5/scale, width=0.008, color='red')
        plt.arrow(-0.9*lim, 0.87*lim, 0.5, 0, width = 0.008, color = 'Red')
        plt.text(-0.9*lim, 0.90*lim, f"Thrust scale: {scale} N")

    if velocity:
        arrow_indices = np.linspace(0, len(y[:,0])-1, N_arrows, dtype = int)
        scale = velocity_scale
        for i in arrow_indices:
            plt.arrow(y[i,0],
                      y[i,1],
                      y[i,2]*0.5/scale,
                      y[i,3]*0.5/scale, width=0.008, color='Blue')
        plt.arrow(-0.9*lim, 0.77*lim, 0.5, 0, width = 0.008, color = 'Blue')
        plt.text(-0.9*lim, 0.80*lim, f"Velocity scale: {scale} N")

    plt.plot(0, 0, 'yo', markersize = 10, label = "Sun")
    plt.plot(1, 0, 'bo', markersize = 10, label = "Departure")
    plt.plot(y[-1,0], y[-1,1], "go", markersize = 10, label = "Arrival")

    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    ax.legend(bbox_to_anchor=(1.3, 1))
    plt.xlabel("x [A.U.]", fontsize=20)
    plt.ylabel("y [A.U.]", fontsize=20)
    ax.set_aspect('equal', adjustable='box')

    if grid:
        ax.grid()

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(labelsize=16)
    ax.tick_params(axis="both", direction="in", which="both", length=4, width = 1.5)
    ax.tick_params(bottom=True, top=True, left=True, right=True)



def plot_coordinates(DOLPHINN,
                     coordinates = "NDcartesian",
                     bench = False,
                     plot_control = True,
                     custom_labels = None,
                     custom_control_labels = None):

    # Check if plot is possible
    if plot_control and not DOLPHINN.dynamics.control:
        raise ValueError("[DOLPHINN] Requested control profile plot but solution did not produce control")

    if bench:
        if coordinates not in list(DOLPHINN.bench.states.keys()):
            raise ValueError(f"[DOLPHINN] Coordinates ({coordinates}) not available in DOLPHINN.states.bench, please calculate first")
    else:
        if coordinates not in list(DOLPHINN.states.keys()):
            raise ValueError(f"[DOLPHINN] Coordinates ({coordinates}) not available in DOLPHINN.states, please calculate first")

    if bench:
        time = DOLPHINN.bench.states[coordinates][:,0]
        states = DOLPHINN.bench.states[coordinates][:,1:1+DOLPHINN.dynamics.entries-DOLPHINN.dynamics.control_entries]
        control = DOLPHINN.bench.states[coordinates][:,-DOLPHINN.dynamics.control_entries:]
    else:
        time = DOLPHINN.states[coordinates][:,0]
        states = DOLPHINN.states[coordinates][:,1:1+DOLPHINN.dynamics.entries-DOLPHINN.dynamics.control_entries]
        control = DOLPHINN.states[coordinates][:,-DOLPHINN.dynamics.control_entries:]

    if plot_control:
        fig, axes = plt.subplots(DOLPHINN.dynamics.entries - DOLPHINN.dynamics.control_entries,
                                 2,
                                 figsize = (12, 7),
                                 sharex = True)
    else:
        fig, axes = plt.subplots(DOLPHINN.dynamics.entries - DOLPHINN.dynamics.control_entries,
                                 1,
                                 figsize = (8, 7),
                                 sharex = True)


    fig.subplots_adjust(hspace=0.2)
    if bench:
        title = f"Coordinates [{coordinates}] time evolution\n--Verification numerical solution--"
    else:
        title = f"Coordinates [{coordinates}] time evolution\n--DOLPHINN solution--"
    fig.suptitle(title, fontsize = 20, y=0.98)

    if plot_control:
        for i in range(DOLPHINN.dynamics.entries - DOLPHINN.dynamics.control_entries):
            axes[i, 0].plot(time, states[:,i])
            if custom_labels:
                axes[i, 0].set_ylabel(custom_labels[i] , fontsize = 16)
            else:
                axes[i, 0].set_ylabel(DOLPHINN.dynamics.entry_labels[i], fontsize = 16)
        axes[i, 0].set_xlabel("Time [-]", fontsize = 16)

        for i in range(DOLPHINN.dynamics.control_entries):
            axes[i, 1].plot(time, control[:,i])
            if custom_control_labels:
                axes[i, 1].set_ylabel(custom_control_labels[i] , fontsize = 16)
            else:
                axes[i, 1].set_ylabel(f"$u_{i+1}$" , fontsize = 16)

        axes[i, 1].set_xlabel("Time [-]", fontsize = 16)

        for i in range(DOLPHINN.dynamics.entries - 2* DOLPHINN.dynamics.control_entries):
            axes[DOLPHINN.dynamics.control_entries+i,1].axis('off')
    else:
        for i in range(DOLPHINN.dynamics.entries - DOLPHINN.dynamics.control_entries):
            axes[i].plot(time, states[:,i])
            if custom_labels:
                axes[i].set_ylabel(custom_labels[i] , fontsize = 16)
            else:
                axes[i].set_ylabel(DOLPHINN.dynamics.entry_labels[i] , fontsize = 16)
        axes[i].set_xlabel("Time [-]", fontsize = 16)

    for ax in axes.flat:
        ax.grid()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2)
        ax.tick_params(labelsize=13)
        ax.tick_params(axis="both", direction="in", which="both", length=4, width = 1.2)
        ax.tick_params(bottom=True, top=True, left=True, right=True)



def compare(DOLPHINN,
            coordinates = "NDcartesian",
            custom_labels = None,
            log = True):

    if coordinates not in list(DOLPHINN.states.keys()):
        raise ValueError(f"[DOLPHINN] DOLPHINN solution does not contain {coordinates} ephemeris")

    if coordinates not in list(DOLPHINN.bench.states.keys()):
        raise ValueError(f"[DOLPHINN] Verification solution does not contain {coordinates} ephemeris")

    time = DOLPHINN.states[coordinates][:,0]
    states = DOLPHINN.states[coordinates][:,1:1+DOLPHINN.dynamics.entries-DOLPHINN.dynamics.control_entries]
    states_bench = DOLPHINN.bench.states[coordinates][:,1:1+DOLPHINN.dynamics.entries-DOLPHINN.dynamics.control_entries]

    fig, axes = plt.subplots(2, 2, figsize = (14, 8), gridspec_kw={'height_ratios': [5, 2]}, sharex = True)
    fig.subplots_adjust(hspace=0, wspace = 0.3)
    fig.suptitle(f"Compare verification to DOLPHINN [{coordinates}]", fontsize = 20, y = 0.98)

    for i in range(2):
        if custom_labels:
            label = custom_labels[i]
        else:
            label = f"x$_{i+1}$"

        axes[0,0].plot(time, states[:,i], label = f"DOLPHINN [{label}]")
        axes[0,0].plot(time, states_bench[:,i], linestyle = '--', label = f"Verification [{label}]")

        axes[1,0].plot(time, np.abs(states[:,+i] - states_bench[:,i]), label = f"[{label}]")


    for i in range(2):
        if custom_labels:
            label = custom_labels[2+i]
        else:
            label = f"x$_{i+3}$"

        axes[0,1].plot(time, states[:,2+i], label = f"DOLPHINN [{label}]")
        axes[0,1].plot(time, states_bench[:,2+i], linestyle = '--',  label = f"Verification [{label}]")

        axes[1,1].plot(time, np.abs(states[:,2+i] - states_bench[:,2+i]), label = f"[{label}]")


    axes[1,1].set_xlabel("Time [-]", fontsize = 20)
    axes[1,0].set_xlabel("Time  [-]", fontsize = 20)

    axes[0,0].set_ylabel("Position [-]", fontsize = 20)
    axes[0,1].set_ylabel("Velocity [-]", fontsize = 20)

    axes[1,0].set_ylabel("Residuals [-]", fontsize = 16)
    axes[1,1].set_ylabel("Residuals [-]", fontsize = 16)

    if log:
        axes[1,0].set_yscale("log")
        axes[1,1].set_yscale("log")

    for ax in axes.flat:
        ax.legend()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2)
        ax.tick_params(labelsize=16)
        ax.tick_params(axis="both", direction="in", which="both", length=4, width = 1.2)
        ax.tick_params(bottom=True, top=True, left=True, right=True)


def compare_mass(case):


    if not hasattr(case, "mass"):
        raise AttributeError("[DOLPHINN] Given DOLPHINN has not mass attribute")
    if not hasattr(case.bench, "mass"):
        raise AttributeError("[DOLPHINN] Given TUDAT verification has not mass attribute")

    fig, axes = plt.subplots(2, 1, figsize = (7, 8), gridspec_kw={'height_ratios': [5, 2]}, sharex = True)
    fig.subplots_adjust(hspace=0.1)
    fig.suptitle("Spacecraft mass evolution", fontsize = 18, y = 0.96)

    axes[0].plot(case.mass[:,0], case.mass[:,1], label = "DOLPHINN mass")
    axes[0].plot(case.bench.mass[:,0]/case.data['time_scale'], case.bench.mass[:,1], linestyle = '--', label = "TUDAT Mass")
    axes[0].set_ylabel("Mass [kg]", fontsize = 16)
    axes[0].legend()

    axes[1].plot(case.mass[:,0], case.mass[:,1] - case.bench.mass[:,1])
    axes[1].set_ylabel("Mass Residual [kg]", fontsize = 16)
    axes[1].set_xlabel("Time [-]", fontsize = 16)

    for ax in axes.flat:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2)
        ax.tick_params(labelsize=16)
        ax.tick_params(axis="both", direction="in", which="both", length=4, width = 1.2)
        ax.tick_params(bottom=True, top=True, left=True, right=True)


def plot_metrics(problem):


    fig, axes = plt.subplots(len(problem.metrics), 1,
                             figsize = (8, 2*len(problem.metrics)),
                             sharex = True)

    fig.subplots_adjust(hspace=0.1)
    fig.suptitle("Metrics vs Iterations", fontsize = 18, y = 0.95)

    metrics_test = np.array(problem.metrics_test)
    for i, ax in enumerate(axes):
        metric_name = problem.metrics[i].__self__.__class__.__name__
        metric_values = metrics_test[:,i]

        if metric_name == "FinalDm":
            metric_values = np.abs(metric_values)

        ax.plot(problem.steps,
                metric_values,
                label = f"Final value: {np.round(metrics_test[-1,i], 4)}")
        ax.set_ylabel(metric_name, fontsize = 14)

    axes[-1].set_xlabel("Epoch [-]", fontsize = 14)

    for i, ax in enumerate(axes):
        metric_name = problem.metrics[i].__self__.__class__.__name__
        if metric_name not in ['Fuel', 'FinalRadius']:
            ax.set_yscale("log")
        ax.legend()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2)
        ax.tick_params(labelsize=13)
        ax.tick_params(axis="both",direction="in",which="both", length=4, width = 1.2)
        ax.tick_params(bottom=True, top=True, left=True, right=True)

