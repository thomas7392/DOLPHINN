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
            oweigth = case.config['train_1']['loss_weigths'][-1]
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
    for i in range(len(loss_train[0])):
        axes[1].plot(steps, np.array(loss_train)[:,i], label = loss_labels[i])

    if objective_zoom:
        axes[2].set_title("Objective Loss Zoom-In", fontsize = 20)
        axes[2].plot(steps[steps > 10000], np.array(loss_train)[steps > 10000,-1], label = "Train")
        axes[2].plot(steps[steps > 10000], np.array(loss_test)[steps > 10000,-1],
                    label = f"Test\nFinal Fuel mass = {np.round(loss_test[-1, -1]/oweigth, 2)} kg")
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






def plot_transfer(case, thrust = True, velocity = False):

    # Retrieve relevant data from the DOLPHINN class instance
    y = case.states['NDcartesian'][:,1:]
    final_state = case.data['final_state']
    initial_state = case.data['initial_state']

    theta = np.linspace(0, 2*np.pi, 1000)
    r_inner = np.abs(final_state[0])
    r_outer = np.abs(initial_state[0])

    lim = max(r_inner, r_outer)

    fig, ax = plt.subplots(1, figsize = (10, 10))
    fig.suptitle("Physically Constrained Neural Network Solution", fontsize = 25, y=0.95)
    plt.plot(y[:,0], y[:,1], label = "transfer trajectory")

    plt.plot(r_inner*np.cos(theta), r_inner*np.sin(theta), label = "Target orbit")
    plt.plot(r_outer*np.cos(theta), r_outer*np.sin(theta), label = "Earth orbit")

    if thrust:
        arrow_indices = np.linspace(0, len(y[:,0])-1, 50, dtype = int)
        scale = 0.02
        for i in arrow_indices:
            plt.arrow(y[i,0],
                      y[i,1],
                      y[i,4]*0.5/scale,
                      y[i,5]*0.5/scale, width=0.004, color='red')
        plt.arrow(-1.5, lim, 0.5, 0, width = 0.004, color = 'Red')
        plt.text(-1.5, 1.02*lim, f"Thrust scale: {scale} N")

    if velocity:
        arrow_indices = np.linspace(0, len(y[:,0])-1, 50, dtype = int)
        scale = 10
        for i in arrow_indices:
            plt.arrow(y[i,0],
                      y[i,1],
                      scale * y[i,2],
                      scale * y[i,3], width=0.004, color='red')
        plt.arrow(-1, lim, scale/10, 0, width = 0.004, color = 'Red')
        plt.text(-1, 0.95, "Velocity scale: 0.1 ")

    plt.plot(0, 0, 'yo', markersize = 10, label = "Sun")
    plt.plot(1, 0, 'bo', markersize = 10, label = "Departure")
    plt.plot(y[-1,0], y[-1,1], "go", markersize = 10, label = "Arrival")

    plt.xlim(-1.5*lim, 1.5*lim)
    plt.ylim(-1.5*lim, 1.5*lim)
    ax.legend(bbox_to_anchor=(1.3, 1))
    plt.xlabel("x [A.U.]", fontsize=20)
    plt.ylabel("y [A.U.]", fontsize=20)
    ax.set_aspect('equal', adjustable='box')

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(labelsize=16)
    ax.tick_params(axis="both", direction="in", which="both", length=4, width = 1.5)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
