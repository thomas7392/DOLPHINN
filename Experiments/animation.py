import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import json
import os, sys

from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

#Get the absolute path of the current scriptf
current_path = os.path.dirname(os.path.abspath(__file__))
dolphinn_path = os.path.join(current_path, '..')
sys.path.append(dolphinn_path)

from DOLPHINN.coordinate_transformations import radial_to_NDcartesian
#from utils import integrate_theta

run_from_file = True
base_path = "../Data/Optimisation/LVLH/mars_1_5_revolv2/"
coordinates = "radial"

def create_animation(path):
    # TODO
    pass


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

if run_from_file:

    # Settings
    final_position = np.array([-1.5, 0])
    initial_position = np.array([1, 0])

    r_initial = 1
    r_final = 1.5
    lim = 1.3 * max(r_initial, r_final)

    # Import data
    with open(base_path + "animation_data/data.pickle", "rb") as handle:
        data = pickle.load(handle)

    # Import config
    with open(base_path + "config", 'r') as file:
        config = json.load(file)

    # Remove epochs with nan values
    nan_epochs = []
    for key, value in data['y_pred_test'].items():
        if np.isnan(np.array(value)).any():
            nan_epochs.append(key)

    for cat in ['x_train', 'y_pred_test', 'y_pred_train', 'loss_train', 'loss_test', 'loss_metrics', 'lr']:
        for key in nan_epochs:
            del data[cat][key]

    if coordinates == "radial":
        states = {}
        for epoch, _states in data['y_pred_test'].items():
            _time = data['x_test']
            _theta = integrate_theta(_time.reshape(1, -1)[0], _states)
            _states_with_time = np.concatenate((_time, _states[:,0:1], _theta.reshape(-1, 1), _states[:,1:]), axis = 1)
            _states = radial_to_NDcartesian(_states_with_time, config)[:,1:]
            states[epoch] = _states
    else:
        states = data['y_pred_test']


    epochs = np.array(list(data['y_pred_test'].keys()))

    iloss_train = np.array(list(data['loss_train'].values()))
    loss_train = np.sum(np.array(list(data['loss_train'].values())), axis = 1)
    loss_test = np.sum(np.array(list(data['loss_train'].values())), axis = 1)
    metrics = np.array(list(data['loss_metrics'].values()))
    lr = np.array(list(data['lr'].values()))
    n_epochs = epochs[-1]

    # Lay out settings
    base_title = "DOLPHINN transfer solution"
    specific_title = f"Fixed-time: {int(config['tfinal']*config['time_scale']/(24*3600))} days"
    N_THRUST_VECTORS = 50



    fig = plt.figure(figsize = (14, 7))
    gs = GridSpec(4, 2, left=0.05, right=0.95, wspace=0.2)

    ax_left = fig.add_subplot(gs[:,0])
    ax_loss = fig.add_subplot(gs[0,1])
    ax_iloss = fig.add_subplot(gs[1:3,1])
    ax_metrics = fig.add_subplot(gs[3,1])

    axes = [ax_left, ax_loss, ax_metrics, ax_iloss]

    #plt.subplots_adjust(right = 0.9, left = 0.2)
    fig.suptitle(base_title + "\n" + specific_title, fontsize = 16, y = 0.97)

    epoch_text = axes[0].text(0.5, 0.95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                    transform=axes[0].transAxes, ha="center")

    lr_text = axes[0].text(0.85, 0.95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                    transform=axes[0].transAxes, ha="center")

    # Empty line objects for the animation
    line_trajectory, = axes[0].plot([], [], lw=2)
    line_train_loss, = axes[1].plot([], [], lw=2, label = "Train loss")
    line_test_loss, = axes[1].plot([], [], lw=2, label = "Test Loss", linestyle = '--', zorder = 100)
    line_x_metric, = axes[2].plot([], [], lw = 2, label = "Position")
    line_v_metric, = axes[2].plot([], [], lw = 2, label = "Velocity")

    iloss_lines = []
    for i in range(len(iloss_train[0])):
        line, = axes[3].plot([], [], label = f"Loss index {i}")
        iloss_lines.append(line)


    # Prepare thrust vectors in heliocentric frame
    thrust_vectors = []
    for i in range(N_THRUST_VECTORS):
        arrow = axes[0].arrow(0, 0, 0, 0, color='red', width = 0.004)
        thrust_vectors.append(arrow)

    # Plot the planets
    theta = np.linspace(0, 2*np.pi, 1000)
    axes[0].plot(r_final*np.cos(theta), r_final*np.sin(theta), label = "Target orbit")
    axes[0].plot(r_initial*np.cos(theta), r_initial*np.sin(theta), label = "Initial orbit")
    axes[0].plot(0, 0, 'yo', markersize = 10, label = "Sun")
    axes[0].plot(1, 0, 'bo', markersize = 10, label = "Departure")
    line_arrival, = axes[0].plot([], [], 'go', markersize = 10, label = "Arrival")

    # Style trajectory
    thrust_scale = 0.03

    axes[0].arrow(-0.95*lim, 0.87*lim, 0.5, 0, width = 0.008, color = 'Red')
    axes[0].text(-0.95*lim, 0.90*lim, f"Thrust scale: {thrust_scale} N")
    axes[0].set_xlim(-1.2*lim, 1.2*lim)  # Set x-axis limits
    axes[0].set_ylim(-1.2*lim, 1.2*lim)  # Set y-axis limits
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlabel("x [A.U.]", fontsize=14)
    axes[0].set_ylabel("y [A.U.]", fontsize=14)
    axes[0].legend(loc = "lower right")

    # Style Loss
    axes[1].set_xlim(0, n_epochs)
    axes[1].set_ylim(1e-7, 1e5)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Loss", fontsize=14)
    axes[1].set_xlabel("Epochs [-]", fontsize=14)
    axes[1].grid()

    axes[1].legend()

    # Style metrics
    axes[2].set_xlim(0, n_epochs)
    axes[2].set_ylim(1e-5, 1e2)
    axes[2].set_yscale("log")
    axes[2].grid()
    axes[2].legend()
    axes[2].set_ylabel("Final error", fontsize=14)
    axes[2].set_xlabel("Epochs [-]", fontsize=14)

    # Style metrics
    axes[3].set_xlim(0, n_epochs)
    axes[3].set_ylim(1e-8, 1e2)
    axes[3].set_yscale("log")
    axes[3].grid()
    axes[3].legend()
    axes[3].set_ylabel("Loss", fontsize=14)
    axes[3].set_xlabel("Epochs [-]", fontsize=14)


    # Universal style
    for ax in axes:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.tick_params(labelsize=14)
        ax.tick_params(axis="both", direction="in", which="both", length=4, width = 1.5)
        ax.tick_params(bottom=True, top=True, left=True, right=True)

    # =========================================
    # Animate
    # =========================================
    vertical_lines = []

    def init():

        line_trajectory.set_data([], [])  # Clear the line
        line_train_loss.set_data([], [])  # Clear the line
        line_test_loss.set_data([], [])  # Clear the line

        return line_trajectory, line_train_loss, line_test_loss


    def updateTrajectory(epoch):


        y_pred_test = states[epoch]

        x = y_pred_test[:,0]
        y = y_pred_test[:,1]

        line_trajectory.set_data(x, y)
        line_arrival.set_data([x[-1]], [y[-1]])

        arrow_indices = np.linspace(0, len(y_pred_test[:,0])-1, N_THRUST_VECTORS, dtype = int)
        for arrow_i, i in enumerate(arrow_indices):

            thrust_vectors[arrow_i].set_data(x=y_pred_test[i,0],
                                            y=y_pred_test[i,1],
                                            dx = y_pred_test[i,4]*0.5/thrust_scale,
                                            dy = y_pred_test[i,5]*0.5/thrust_scale)

        return [line_trajectory, line_arrival, *thrust_vectors]

    def updateLosses(frame):

        line_test_loss.set_data(epochs[:frame], loss_test[:frame])
        line_train_loss.set_data(epochs[:frame], loss_train[:frame])

        for i, line in enumerate(iloss_lines):
            line.set_data(epochs[:frame], iloss_train[:frame, i])

        if lr[frame] != lr[frame - 1] and frame > 0:
            for ax in np.array(axes)[1:]:
                vertical_line = ax.axvline(x=epochs[frame], color='k', linestyle = '--')  # Adjust the color as needed
                vertical_lines.append(vertical_line)

        return [line_test_loss, line_train_loss, *iloss_lines, *vertical_lines]

    def updateMetric(frame):


        line_x_metric.set_data(epochs[:frame], metrics[:frame,0])
        line_v_metric.set_data(epochs[:frame], metrics[:frame,1])

        lines_to_return = [line_x_metric, line_v_metric]

        return lines_to_return

    def update(frame):

        epoch = epochs[frame]
        new_title = f"Epoch: {epoch}"
        epoch_text.set_text(new_title)
        lr_text.set_text(f"lr = {lr[frame]}")


        lines1 = updateTrajectory(epoch)
        lines2 = updateLosses(frame)
        lines3 = updateMetric(frame)

        return [*lines1, *lines2, *lines3, epoch_text, lr_text]


    ani = FuncAnimation(fig,
                        update,
                        frames=np.arange(0, len(epochs) - len(nan_epochs)),
                        init_func=init,
                        blit=True,
                        interval = 50)

    # def callback(frame, total_frame):
    #     print(f"Progress: {100* frame/total_frame} %", end = "\r")


    # ani.save(base_path + 'animation.gif',
    #          writer='imagemagick',
    #          fps=1000/50,
    #          progress_callback=callback)

    # print(f"Saved gif at {base_path}/animation.gif")

    plt.show()