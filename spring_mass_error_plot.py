import numpy as np

import matplotlib.pyplot as plt

# Plot the error.
# There should be a line denoting the mean error over all 100 trials,
# and a shaded region showing the standard deviation.

# Load the data.
data = np.load("spring_mass_error_data.npz")
sample_sizes = data["sample_sizes"]
results = data["results"]

# Compute the mean and standard deviation.
mean = np.mean(results, axis=1)
std = np.std(results, axis=1)


fig = plt.figure(figsize=(3.4, 2))
ax = fig.add_subplot(111)

ax.set_xscale("log")
ax.set_yscale("log")

# Show grid.
ax.grid(True, which="both", axis="both", linestyle="-", alpha=0.5)

# Set the x-axis limits.
ax.set_xlim(np.min(sample_sizes), np.max(sample_sizes))

# Set the x-axis ticks as being [1, 100, 1000, 5000].
# ax.set_xticks([1, 100, 1000, 5000])
# ax.set_xticklabels(["1", "100", "1000", "5000"])

ax.set_xlabel("Sample Size")
ax.set_ylabel("Prediction Error")

# Plot the standard deviation of the unbiased RBF as a shaded blue region.
ax.fill_between(
    sample_sizes,
    mean[:, 0] - std[:, 0],
    mean[:, 0] + std[:, 0],
    alpha=0.2,
    color="C0",
    linewidth=0.0,
)

# Plot the standard deviation of the biased polynomial as a shaded orange region.
ax.fill_between(
    sample_sizes,
    mean[:, 1] - std[:, 1],
    mean[:, 1] + std[:, 1],
    alpha=0.2,
    color="C1",
    linewidth=0.0,
)

# Plot the standard deviation of the biased RBF as a shaded blue region.
ax.fill_between(
    sample_sizes,
    mean[:, 2] - std[:, 2],
    mean[:, 2] + std[:, 2],
    alpha=0.2,
    color="C0",
    linewidth=0.0,
)

# Plot the mean error of the unbiased RBF as a dashed blue line with small circles.
ax.plot(
    sample_sizes,
    mean[:, 0],
    label="Unbiased Error (RBF)",
    color="C0",
    linestyle="--",
    marker=".",
    clip_on=False,
)

# Plot the mean error of the biased polynomial as a solid orange line.
ax.plot(
    sample_sizes,
    mean[:, 1],
    label="Biased Error (Polynomial)",
    color="C1",
    linestyle="-",
    marker="o",
    clip_on=False,
)

# Plot the mean error of the biased RBF as a solid blue line.
ax.plot(
    sample_sizes,
    mean[:, 2],
    label="Biased Error (RBF)",
    color="C0",
    linestyle="-",
    marker="o",
    clip_on=False,
)


plt.savefig("spring_mass_error_plot.pdf", bbox_inches="tight")
