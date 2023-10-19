from functools import partial
from itertools import product

import numpy as np
from dataclasses import dataclass

from scipy.integrate import solve_ivp

from sklearn.metrics.pairwise import polynomial_kernel as sklearn_polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf_kernel

import matplotlib.pyplot as plt


@dataclass
class SpringMassParams:
    """Spring-mass-damper parameters."""

    mass: float = 1.0
    """Mass of the object."""

    spring_constant: float = 1.0
    """Spring constant."""

    damping_coefficient: float = 0.5
    """Damping coefficient."""


def spring_mass_dynamics(t: float, x: np.ndarray, params):
    q, p = x

    q_dot = p / params.mass
    p_dot = -params.spring_constant * q

    return np.array([q_dot, p_dot])


def spring_mass_damper_dynamics(t: float, x: np.ndarray, params):
    q, p = x

    q_dot = p / params.mass
    p_dot = -params.spring_constant * q - params.damping_coefficient / params.mass * p

    return np.array([q_dot, p_dot])


def unbiased_prediction(
    t: float,
    x0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    kernel: callable,
):
    """Compute the unbiased prediction."""

    trajectory = []
    trajectory.append(x0)

    for i in range(len(t) - 1):
        # Compute the kernel vector.
        k = kernel(x, trajectory[-1].reshape(1, -1))

        # Compute the prediction.
        prediction = k.T @ W @ y

        # Append the prediction to the trajectory.
        trajectory.append(np.squeeze(prediction))

    return np.array(trajectory)


def biased_prediction(
    t: float,
    x0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    kernel: callable,
    dynamics: callable,
):
    """Compute the biased prediction."""

    trajectory = []
    trajectory.append(x0)

    f = []
    for i in range(len(x)):
        # f.append(odeint(dynamics, x[i], np.array([0.0, sampling_time]))[-1])
        f.append(solve_ivp(dynamics, np.array([0.0, sampling_time]), x[i]).y[:, -1])

    f = np.array(f)

    for i in range(len(t) - 1):
        # Compute the kernel vector.
        k = kernel(x, trajectory[-1].reshape(1, -1))

        # Compute the prediction.
        prediction = k.T @ W @ y

        # Compute the bias.
        bias = k.T @ W @ f

        # Compute the correction.
        # correction = odeint(dynamics, trajectory[i], np.array([0.0, sampling_time]))[-1]
        correction = solve_ivp(
            dynamics, np.array([0.0, sampling_time]), trajectory[i]
        ).y[:, -1]

        # Append the prediction to the trajectory.
        trajectory.append(np.squeeze(prediction - bias) + correction)

    return np.array(trajectory)


sampling_time = 0.1

# Undamped system dynamics.
params_undamped = SpringMassParams(
    mass=1.0,
    spring_constant=1.0,
    damping_coefficient=0.0,
)
dynamics_undamped = partial(spring_mass_dynamics, params=params_undamped)

# Damped system dynamics.
params_damped = SpringMassParams(
    mass=1.0,
    spring_constant=1.0,
    damping_coefficient=0.5,
)
dynamics_damped = partial(spring_mass_damper_dynamics, params=params_damped)

# Define the initial condition.
x0 = np.array([0.1, 0.1])

# Define the polynomial kernel.
polynomial_kernel = partial(sklearn_polynomial_kernel, degree=2, coef0=1.0)

# Define the RBF kernel.
sigma = 0.3
gamma = 1.0 / (2 * sigma**2)
rbf_kernel = partial(sklearn_rbf_kernel, gamma=gamma)


# For the first experiment, we have two sets of cases.
# We want to test under different sample sizes, sample_size = 10, 50, 100, 500.
# We also want to test under the different samples, one over the entire region of
# interest and one over a smaller region.

# Create a figure.
fig, ax = plt.subplots(2, 4, figsize=(7, 4), dpi=300)

# Set the font size to 8pt.
plt.rcParams.update({"font.size": 8})
# Reduce the spacing between subplots.
plt.subplots_adjust(wspace=0.05, hspace=0.05)

fig.supxlabel("$q$")
fig.supylabel("$\dot{q}$")

# The top row of the figure will be the partial dataset.
# The bottom row of the figure will be the entire dataset.
# Going from left to right, the size of the dataset will increase.
# Each plot will show the undamped dynamics, the true dynamics, the purely data-driven
# estiamte, the biased estimate using therbf kernel, and the biased estimate with the
# polynoimial kernel.

# Set the numpy rng seed.
np.random.seed(42)


# Compute the true (undamped) dynamics and plot them as a dashed gray line in each plot.
t = np.linspace(0.0, 2 * np.pi, 100)
traj_undamped = solve_ivp(dynamics_undamped, np.array([0.0, 2 * np.pi]), x0, t_eval=t)
traj_undamped = traj_undamped.y.T

# Compute the true (damped) dynamics and plot them as a solid black line in each plot.
t = np.linspace(0, 10, 100)
traj_damped = solve_ivp(dynamics_damped, np.array([0.0, 10.0]), x0, t_eval=t)
traj_damped = traj_damped.y.T

# Iterate over all rows and columns of the figure.
for row, col in product(range(ax.shape[0]), range(ax.shape[1])):
    h_undamped = ax[row, col].plot(
        traj_undamped[:, 0],
        traj_undamped[:, 1],
        color="gray",
        linestyle=":",
        label="Undamped Dynamics",
    )

    h_damped = ax[row, col].plot(
        traj_damped[:, 0],
        traj_damped[:, 1],
        color="black",
        label="True Dynamics",
    )

    ax[row, col].grid()
    ax[row, col].set_xlim([-0.15, 0.15])
    ax[row, col].set_ylim([-0.15, 0.15])


# Generate a sample from the system.
# Sample x uniformly within the region [-0.15, 0.15] x [-0.15, 0.15]
# and then secondly from [0, 0.15] x [0, 0.15].
sample_size = 500

x_entire = np.random.uniform(-0.15, 0.15, size=(sample_size, 2))
x_partial = np.random.uniform(0.0, 0.15, size=(sample_size, 2))
y_entire = []
y_partial = []
for i in range(sample_size):
    y_entire.append(
        solve_ivp(dynamics_damped, np.array([0.0, sampling_time]), x_entire[i]).y[:, -1]
        + np.random.normal(0.0, 0.0001, size=(2,))
    )
    y_partial.append(
        solve_ivp(dynamics_damped, np.array([0.0, sampling_time]), x_partial[i]).y[
            :, -1
        ]
        + np.random.normal(0.0, 0.0001, size=(2,)),
    )

y_entire = np.array(y_entire)
y_partial = np.array(y_partial)


regularization_parameter = 1 / (sample_size**2)


for i, M in enumerate([10, 50, 100, 500]):
    idx = np.random.choice(sample_size, size=M, replace=False)

    x_partial_subset = x_partial[idx]
    x_entire_subset = x_entire[idx]
    y_partial_subset = y_partial[idx]
    y_entire_subset = y_entire[idx]

    # Plot the dataset as transparent gray dots.
    h_data = ax[0, i].scatter(
        x_partial_subset[:, 0],
        x_partial_subset[:, 1],
        color="gray",
        alpha=0.2,
        label="Data",
        s=5,
    )
    ax[1, i].scatter(
        x_entire_subset[:, 0],
        x_entire_subset[:, 1],
        color="gray",
        alpha=0.2,
        s=5,
    )

    # Compute the kernel matrix for the rbf kernel.
    K_rbf_partial = rbf_kernel(x_partial_subset, x_partial_subset)
    # Compute the kernel matrix for the polynomial kernel.
    K_poly_partial = polynomial_kernel(x_partial_subset, x_partial_subset)

    # Compute the kernel matrix inverse.
    W_rbf_partial = np.linalg.inv(K_rbf_partial + regularization_parameter * np.eye(M))
    W_poly_partial = np.linalg.inv(
        K_poly_partial + regularization_parameter * np.eye(M)
    )

    # Compute the unbiased prediction.
    x0 = np.array([0.1, 0.1])
    traj_unbiased_rbf_partial = unbiased_prediction(
        t, x0, x_partial_subset, y_partial_subset, W_rbf_partial, rbf_kernel
    )

    x0 = np.array([0.1, 0.1])
    traj_unbiased_poly_partial = unbiased_prediction(
        t, x0, x_partial_subset, y_partial_subset, W_poly_partial, polynomial_kernel
    )

    # Compute the biased prediction.
    x0 = np.array([0.1, 0.1])
    traj_biased_rbf_partial = biased_prediction(
        t,
        x0,
        x_partial_subset,
        y_partial_subset,
        W_rbf_partial,
        rbf_kernel,
        dynamics_undamped,
    )

    x0 = np.array([0.1, 0.1])
    traj_biased_poly_partial = biased_prediction(
        t,
        x0,
        x_partial_subset,
        y_partial_subset,
        W_poly_partial,
        polynomial_kernel,
        dynamics_undamped,
    )

    # Plot the unbiased prediction.
    ax[0, i].plot(
        traj_unbiased_rbf_partial[:, 0],
        traj_unbiased_rbf_partial[:, 1],
        color="C0",
        linestyle="--",
    )

    ax[0, i].plot(
        traj_unbiased_poly_partial[:, 0],
        traj_unbiased_poly_partial[:, 1],
        color="C1",
        linestyle="--",
    )

    # Plot the biased prediction.
    ax[0, i].plot(
        traj_biased_rbf_partial[:, 0],
        traj_biased_rbf_partial[:, 1],
        color="C0",
    )

    ax[0, i].plot(
        traj_biased_poly_partial[:, 0],
        traj_biased_poly_partial[:, 1],
        color="C1",
    )

    # Compute the kernel matrix for the rbf kernel.
    K_rbf_entire = rbf_kernel(x_entire_subset, x_entire_subset)
    # Compute the kernel matrix for the polynomial kernel.
    K_poly_entire = polynomial_kernel(x_entire_subset, x_entire_subset)

    # Compute the kernel matrix inverse.
    W_rbf_entire = np.linalg.inv(K_rbf_entire + regularization_parameter * np.eye(M))
    W_poly_entire = np.linalg.inv(K_poly_entire + regularization_parameter * np.eye(M))

    # Compute the unbiased prediction.
    x0 = np.array([0.1, 0.1])
    traj_unbiased_rbf_entire = unbiased_prediction(
        t, x0, x_entire_subset, y_entire_subset, W_rbf_entire, rbf_kernel
    )

    x0 = np.array([0.1, 0.1])
    traj_unbiased_poly_entire = unbiased_prediction(
        t, x0, x_entire_subset, y_entire_subset, W_poly_entire, polynomial_kernel
    )

    # Compute the biased prediction.
    x0 = np.array([0.1, 0.1])
    traj_biased_rbf_entire = biased_prediction(
        t,
        x0,
        x_entire_subset,
        y_entire_subset,
        W_rbf_entire,
        rbf_kernel,
        dynamics_undamped,
    )

    x0 = np.array([0.1, 0.1])
    traj_biased_poly_entire = biased_prediction(
        t,
        x0,
        x_entire_subset,
        y_entire_subset,
        W_poly_entire,
        polynomial_kernel,
        dynamics_undamped,
    )

    # Plot the unbiased prediction.
    h_unbiased_rbf = ax[1, i].plot(
        traj_unbiased_rbf_entire[:, 0],
        traj_unbiased_rbf_entire[:, 1],
        color="C0",
        linestyle="--",
        label="Unbiased Prediction (RBF)",
    )

    h_unbiased_poly = ax[1, i].plot(
        traj_unbiased_poly_entire[:, 0],
        traj_unbiased_poly_entire[:, 1],
        color="C1",
        linestyle="--",
        label="Unbiased Prediction (Polynomial)",
    )

    # Plot the biased prediction.
    h_biased_rbf = ax[1, i].plot(
        traj_biased_rbf_entire[:, 0],
        traj_biased_rbf_entire[:, 1],
        color="C0",
        label="Biased Prediction (RBF)",
    )

    h_biased_poly = ax[1, i].plot(
        traj_biased_poly_entire[:, 0],
        traj_biased_poly_entire[:, 1],
        color="C1",
        label="Biased Prediction (Polynomial)",
    )


# Only show x axis labels and ticks for the bottom row. Remove them for the top row.
for i in range(ax.shape[1]):
    ax[0, i].set_xticks([-0.1, 0.0, 0.1])
    ax[0, i].set_xticklabels([-0.1, 0.0, 0.1])

    ax[1, i].set_xticks([-0.1, 0.0, 0.1])
    ax[1, i].set_xticklabels([-0.1, 0.0, 0.1])

    # ax[0, i].set_xticks([])
    ax[0, i].set_xticklabels([])

    # Remove the x tick marks, but don't set it to empty.
    ax[0, i].tick_params(axis="x", which="both", bottom=False, top=False)


# Only show y axis labels and ticks for the left column. Remove them for all others.
for i in range(ax.shape[0]):
    ax[i, 0].set_yticks([-0.1, 0.0, 0.1])
    ax[i, 0].set_yticklabels([-0.1, 0.0, 0.1])

    ax[i, 1].set_yticks([-0.1, 0.0, 0.1])
    ax[i, 2].set_yticks([-0.1, 0.0, 0.1])
    ax[i, 3].set_yticks([-0.1, 0.0, 0.1])

    ax[i, 1].set_yticklabels([])

    ax[i, 2].set_yticklabels([])

    ax[i, 3].set_yticklabels([])

    # Remove the y tick marks, but don't set it to empty.
    ax[i, 1].tick_params(axis="y", which="both", left=False, right=False)
    ax[i, 2].tick_params(axis="y", which="both", left=False, right=False)
    ax[i, 3].tick_params(axis="y", which="both", left=False, right=False)

# Display a legend above the figure that has three columns of labels.
fig.legend(
    handles=[
        h_data,
        h_undamped[0],
        h_damped[0],
        h_unbiased_rbf[0],
        h_unbiased_poly[0],
        h_biased_rbf[0],
        h_biased_poly[0],
    ],
    loc="outside upper center",
    ncol=3,
    frameon=False,
)

plt.subplots_adjust(
    top=0.85, bottom=0.1, left=0.1, right=0.95, wspace=0.07, hspace=0.07
)
fig.supxlabel("$q$", x=0.525)


plt.savefig("spring_mass_damper.pdf", bbox_inches="tight")
