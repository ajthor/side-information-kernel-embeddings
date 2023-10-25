from functools import partial

import numpy as np
from dataclasses import dataclass

from scipy.integrate import solve_ivp

from sklearn.metrics.pairwise import polynomial_kernel as sklearn_polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf_kernel

from tqdm import tqdm

import matplotlib.pyplot as plt


def check_array(
    array: np.ndarray,
    dtype: np.dtype = np.float64,
    order=None,
    copy: bool = False,
    ensure_2d: bool = True,
    ensure_finite: bool = True,
):
    """Validate array.

    Performs checks to ensure that the array is valid.

    Note:

        This function is intended as a simple replacement for the
        :py:func:`sklearn.metrics.pairwise.check_array` function. Unlike
        :py:mod:`sklearn`, this function does not check sparse input data, and does not
        do sophisticated type checking or upcasting.

    Args:
        array: The array to be validated.
        dtype: The data type of the resulting array.
        order: The memory layout of the resulting array.
        copy: Whether to create a forced copy of ``array``.
        ensure_2d: Whether to raise an error if the array is not 2D.
        ensure_finite: Whether to raise an error if the array is not finite.

    Returns:
        The validated array.

    """

    array_orig = array

    # Convert array to numpy array.
    array = np.asarray(array, dtype=dtype, order=order)

    # Raise an error if the array is not 2D.
    if ensure_2d is True:
        if array.ndim == 0:
            raise ValueError(
                "Expected 2D array, got scalar array instead.\n"
                "Reshape the data using array.reshape(-1, 1) "
                "if the data has only a single dimension "
                "or array.reshape(1, -1) if there is only a single sample."
            )
        elif array.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead.\n"
                "Reshape the data using array.reshape(-1, 1) "
                "if the data has only a single dimension "
                "or array.reshape(1, -1) if there is only a single sample."
            )

    # Raise an error if the array elements are not finite.
    if ensure_finite is True:
        if array.dtype.kind in "fc":
            if not np.isfinite(array).all():
                raise ValueError("Input contains infinity or NaN.")

    # Create a copy if requested.
    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    return array


def check_pairwise_arrays(
    X,
    Y=None,
    dtype: np.dtype = np.float64,
    ensure_finite: bool = True,
    copy: bool = False,
):
    """Check pairwise arrays.

    Note:

        This function is intended as a simple replacement for the
        :py:func:`sklearn.metrics.pairwise.check_pairwise_arrays` function. Unlike
        :py:mod:`sklearn`, this function does not check sparse input data, and does not
        do sophisticated type checking or upcasting.

    Args:
        X: A 2D array with observations oganized in ROWS.
        Y: A 2D array with observations oganized in ROWS.
        dtype: The data type of the resulting array.
        copy: Whether to create a forced copy of ``array``.
        ensure_finite: Whether to raise an error if the array is not finite.

    Returns:
        The validated arrays ``X`` and ``Y``.

    """

    if Y is None:
        X = check_array(X, dtype=dtype, ensure_finite=ensure_finite, copy=copy)
        Y = X

    else:
        X = check_array(X, dtype=dtype, ensure_finite=ensure_finite, copy=copy)
        Y = check_array(Y, dtype=dtype, ensure_finite=ensure_finite, copy=copy)

    return X, Y


def periodic_kernel(X, Y, sigma, period):
    """Periodic kernel.

    k(x, y) = exp(-2 * sin^2(pi * ||x - y|| / period) / sigma^2)

    """

    D = np.squeeze(np.subtract.outer(Y, X)).T

    return np.exp(-np.sin(np.pi * D / period) ** 2 / (2 * sigma**2))


@dataclass
class NonholonomicParams:
    pass


def nonholonomic_dynamics_inertial(t: float, x: np.ndarray, u: np.ndarray, params):
    """Nonholonomic dynamics in the inertial frame."""

    x1, x2, x3 = x
    u1, u2 = u

    x1_dot = u1 * np.cos(x3)
    x2_dot = u1 * np.sin(x3)
    x3_dot = u2

    return np.array([x1_dot, x2_dot, x3_dot])


def nonholonomic_dynamics_body_fixed(t: float, x: np.ndarray, u: np.ndarray, params):
    """Nonholonomic dynamics in the body-fixed frame."""

    x1, x2, x3 = x
    u1, u2 = u

    x1_dot = u1
    x2_dot = 0.0
    x3_dot = u2

    return np.array([x1_dot, x2_dot, x3_dot])


# Time horizon.
time_horizon = 10.0

# Sampling time.
sampling_time = 0.1

# Define a series of time_horizon/sampling_time waypoints that follow a sinusoidal pattern with a very long wavelength.
amplitude = 0.5
period = 5.0
waypoints = np.array(
    [
        [
            (x * sampling_time) - 1.0,
            4
            * amplitude
            / period
            * np.abs(
                (((((x * sampling_time) - 1.0) - period / 2) % period) + period)
                % period
                - period / 2
            )
            - amplitude,
        ]
        for x in range(int(time_horizon / sampling_time))
    ]
)


# Generate sample data.
sample_size = 1000
A_sample_size = 100

# Generate a sample from the inertial frame dynamics.
x_lim = np.array([[-1.0, 1.0], [-1.0, 1.0], [-np.pi, np.pi]])
u_lim = np.array([[0.1, 1.1], [-10.1, 10.1]])

X_inertial = np.random.uniform(x_lim[:, 0], x_lim[:, 1], size=(sample_size, 3))
U_inertial = np.random.uniform(u_lim[:, 0], u_lim[:, 1], size=(sample_size, 2))
A_inertial = np.random.uniform(u_lim[:, 0], u_lim[:, 1], size=(A_sample_size, 2))
Y_inertial = np.zeros_like(X_inertial)
for i in range(sample_size):
    Y_inertial[i] = solve_ivp(
        nonholonomic_dynamics_inertial,
        np.array([0.0, sampling_time]),
        X_inertial[i],
        args=(U_inertial[i], None),
    ).y[:, -1]

# Generate a sample from the body frame dynamics.
X_body = np.random.uniform([0.0, 0.0, -np.pi], [0.0, 0.0, np.pi], size=(sample_size, 3))
U_body = U_inertial
A_body = A_inertial
Y_body = np.zeros_like(X_body)
for i in range(sample_size):
    Y_body[i] = solve_ivp(
        nonholonomic_dynamics_inertial,
        np.array([0.0, sampling_time]),
        X_body[i],
        args=(U_body[i], None),
    ).y[:, -1]


def tracking_cost_invariant(state, waypoint):
    """Tracking cost."""
    # return np.linalg.norm(state[:2] - waypoint, axis=0) ** 2

    _Y = Y_body

    # Rotation.
    theta = state[2]
    _Y[:, 2] += theta
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    _Y = _Y @ R

    # Translation.
    _Y[:, :2] += state[:2]

    # Compute the cost.
    return np.linalg.norm(_Y[:, :2] - waypoint, axis=1) ** 2


# Define the polynomial kernel.
polynomial_kernel = partial(sklearn_polynomial_kernel, degree=2, coef0=1.0)

# Define the RBF kernel.
sigma = 0.3
gamma = 1.0 / (2 * sigma**2)
rbf_kernel = partial(sklearn_rbf_kernel, gamma=gamma)

# Define the periodic kernel.
period = 2.0
periodic_kernel = partial(periodic_kernel, sigma=sigma, period=period)

# Define the kernel function.
kernel = rbf_kernel

regularization_parameter = 1e-3

# We compare three cases.
# The first case is the unbiased predictor with the inertial frame dynamics.
# The second case is the biased predictor with the inertial frame dynamics.
# The third case is the biased predictor with the body frame dynamics.


# Compute the unbiased inertial frame prediction.
K = kernel(X_inertial, X_inertial) * kernel(U_inertial, U_inertial)
K[np.diag_indices_from(K)] += regularization_parameter

W = np.linalg.inv(K)


# Compute the biased inertial frame prediction.
K = kernel(X_inertial, X_inertial) * kernel(U_inertial, U_inertial)
K[np.diag_indices_from(K)] += regularization_parameter

W = np.linalg.inv(K)


# Compute the biased body frame prediction.
G = kernel(U_body, U_body)
G[np.diag_indices_from(G)] += regularization_parameter
b = np.linalg.solve(G, kernel(U_body, A_body))

trajectory_unbiased = []
# Set the initial condition.
initial_condition = np.array([-0.8, 0.0, 0.0])
trajectory_unbiased.append(initial_condition)

for i in range(int(time_horizon / sampling_time)):
    # Compute the predicted cost.
    C = np.atleast_2d(tracking_cost_invariant(trajectory_unbiased[i], waypoints[i])) @ b
    idx = np.argmin(C)

    action = A_inertial[idx]

    # Compute the next state.
    next_state = solve_ivp(
        nonholonomic_dynamics_inertial,
        np.array([0.0, sampling_time]),
        trajectory_unbiased[i],
        args=(action, None),
    ).y[:, -1]

    trajectory_unbiased.append(next_state)


trajectory_unbiased = np.array(trajectory_unbiased)

# Plot the trajectory.
fig = plt.figure(figsize=(3.4, 2))
ax = fig.add_subplot(111)

ax.plot(waypoints[:, 0], waypoints[:, 1], color="C0", linewidth=0.5)
ax.plot(trajectory_unbiased[:, 0], trajectory_unbiased[:, 1], color="C1")

plt.show()
