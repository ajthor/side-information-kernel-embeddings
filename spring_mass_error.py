# In this file, we compare the error of various kernel approximations.
# We compare the purely data-driven kernel embedding to the true dynamics,
# we compare the biased kernel embedding with a Gaussian kernel to the true dynamics,
# and we compare the biased kernel embedding with a polynomial kernel to the true dynamics.
# We vary the sample size from 100 to 5000, and we average over 100 trials.

from functools import partial
from itertools import product

import numpy as np
from dataclasses import dataclass

from scipy.integrate import solve_ivp

from sklearn.metrics.pairwise import polynomial_kernel as sklearn_polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf_kernel

from tqdm import tqdm

import matplotlib.pyplot as plt

sampling_time = 0.1


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
        correction = solve_ivp(
            dynamics, np.array([0.0, sampling_time]), trajectory[i]
        ).y[:, -1]

        # Append the prediction to the trajectory.
        trajectory.append(np.squeeze(prediction - bias) + correction)

    return np.array(trajectory)


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


# Define the polynomial kernel.
polynomial_kernel = partial(sklearn_polynomial_kernel, degree=2, coef0=1.0)

# Define the RBF kernel.
sigma = 0.3
gamma = 1.0 / (2 * sigma**2)
rbf_kernel = partial(sklearn_rbf_kernel, gamma=gamma)


# Define the experiment parameters.
# sample_sizes = np.linspace(10, 5000, 10, dtype=int)
sample_sizes = np.logspace(1, 3, 20, dtype=int)
num_trials = 100

results = np.zeros((len(sample_sizes), num_trials, 4))

# Set the random seed.
np.random.seed(0)


def generate_dataset(sample_size):
    """Generate a sample from the spring-mass-damper system."""
    x = np.random.uniform(-0.15, 0.15, size=(sample_size, 2))
    y = np.zeros((sample_size, 2))
    for i in range(sample_size):
        y[i] = solve_ivp(dynamics_damped, np.array([0.0, sampling_time]), x[i]).y[
            :, -1
        ] + np.random.normal(0.0, 0.0001, size=(2,))

    return x, y


def compute_prediction_error(y, prediction):
    """Compute the prediction error."""
    return np.linalg.norm(y - prediction, axis=1).sum()


def debug_plot_dataset(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x[:, 0], x[:, 1], color="C0")
    ax.scatter(y[:, 0], y[:, 1], color="C1")

    plt.show()


def debug_plot_prediction(t, y, prediction):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(t, y[:, 0], color="C0")
    ax.plot(t, prediction[:, 0], color="C1")

    plt.show()


def debug_plot_phase_portrait(y, prediction):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(y[:, 0], y[:, 1], color="C0")
    ax.plot(prediction[:, 0], prediction[:, 1], color="C1")

    plt.show()


def debug_plot_gram_matrix(K):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(K)

    plt.show()


for i, sample_size in tqdm(enumerate(sample_sizes), total=len(sample_sizes)):
    # Define the regularization parameter.
    regularization_parameter = 1 / (sample_size**2)

    for j in range(num_trials):
        # Generate the dataset.
        x, y = generate_dataset(sample_size)

        # Compute the kernel matrix.
        K_polynomial = polynomial_kernel(x, x)
        K_rbf = rbf_kernel(x, x)

        # Compute the kernel matrix inverse.
        K_polynomial[np.diag_indices_from(K_polynomial)] += regularization_parameter
        # Try to invert. If it fails, use the pseudo-inverse.
        try:
            K_polynomial_inv = np.linalg.inv(K_polynomial)
        except np.linalg.LinAlgError:
            K_polynomial_inv = np.linalg.pinv(K_polynomial)

        K_rbf[np.diag_indices_from(K_rbf)] += regularization_parameter
        K_rbf_inv = np.linalg.inv(K_rbf)

        x0 = np.random.uniform(-0.1, 0.1, size=(2,))

        t = np.linspace(0.0, 10.0, 100)
        true_y = solve_ivp(dynamics_damped, np.array([0.0, 10.0]), x0, t_eval=t).y.T

        # Compute the unbiased prediction.
        traj_unbiased_rbf = unbiased_prediction(
            np.linspace(0.0, 10.0, 100),
            x0,
            x,
            y,
            K_rbf_inv,
            rbf_kernel,
        )

        traj_unbiased_polynomial = unbiased_prediction(
            np.linspace(0.0, 10.0, 100),
            x0,
            x,
            y,
            K_polynomial_inv,
            polynomial_kernel,
        )

        # Compute the biased prediction.
        traj_biased_polynomial = biased_prediction(
            np.linspace(0.0, 10.0, 100),
            x0,
            x,
            y,
            K_polynomial_inv,
            polynomial_kernel,
            dynamics_undamped,
        )

        traj_biased_rbf = biased_prediction(
            np.linspace(0.0, 10.0, 100),
            x0,
            x,
            y,
            K_rbf_inv,
            rbf_kernel,
            dynamics_undamped,
        )

        # Compute the prediction error.
        results[i, j, 0] = compute_prediction_error(true_y, traj_unbiased_rbf)
        results[i, j, 1] = compute_prediction_error(true_y, traj_unbiased_polynomial)
        results[i, j, 2] = compute_prediction_error(true_y, traj_biased_polynomial)
        results[i, j, 3] = compute_prediction_error(true_y, traj_biased_rbf)

# Save the results to an npz file.
np.savez(
    "spring_mass_error_data.npz",
    sample_sizes=sample_sizes,
    results=results,
)
